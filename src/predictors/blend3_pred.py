import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
import warnings

# Warning filters
warnings.filterwarnings('ignore')

class CFG:
    """A minimal config for prediction."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NFOLDS = 5

class Blend3Predictor:
    @staticmethod
    def set_seed(seed=42):
        """Sets the seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    class FTTransformerV2(nn.Module):
        def __init__(self, num_continuous, embed_dim, num_heads, num_layers, dropout):
            super().__init__()
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.cont_embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_continuous)])
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
            self.output_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

        def forward(self, x_continuous):
            bs = x_continuous.size(0)
            continuous_tokens = torch.stack([embed(x_continuous[:, i].unsqueeze(1)) for i, embed in enumerate(self.cont_embeddings)], dim=1)
            x = torch.cat([self.cls_token.expand(bs, -1, -1), continuous_tokens], dim=1)
            x = self.transformer_encoder(x)
            return self.output_head(x[:, 0, :])

    @staticmethod
    def create_advanced_features(df):
        df_fe = df.copy()
        for i in range(1, 6):
            for j in range(1, 11):
                df_fe[f'weighted_prop_{i}_{j}'] = df_fe[f'Component{i}_fraction'] * df_fe[f'Component{i}_Property{j}']
        for j in range(1, 11):
            prop_cols = [f'Component{i}_Property{j}' for i in range(1, 6)]
            df_fe[f'prop_mean_{j}'] = df_fe[prop_cols].mean(axis=1)
            df_fe[f'prop_std_{j}'] = df_fe[prop_cols].std(axis=1)
        epsilon = 1e-6
        for i in range(1, 6):
            for j in range(i + 1, 6):
                frac_i, frac_j = df_fe[f'Component{i}_fraction'], df_fe[f'Component{j}_fraction']
                df_fe[f'frac_ratio_{i}_{j}'] = frac_i / (frac_j + epsilon)
                df_fe[f'frac_diff_{i}_{j}'] = frac_i - frac_j
        df_fe.fillna(0, inplace=True)
        df_fe.replace([np.inf, -np.inf], 0, inplace=True)
        return df_fe

    @staticmethod
    def predict(test_filepath):
        Blend3Predictor.set_seed()

        print("🚀 Starting prediction workflow...")
        try:
            # Define paths
            base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-3')
            SCALER_PATH = os.path.join(base_dir, 'scaler.gz')
            FEATURES_PATH = os.path.join(base_dir, 'top_features.list')
            PARAMS_PATH = os.path.join(base_dir, 'best_params.json')
            MODEL_PATHS = [os.path.join(base_dir, f'final_model_Final_Model_Fixed_Params_fold{i}.pth') for i in range(1, CFG.NFOLDS + 1)]

            # Load assets
            if not os.path.exists(SCALER_PATH) or not os.path.exists(FEATURES_PATH) or not os.path.exists(PARAMS_PATH):
                raise FileNotFoundError("Missing one or more required files in Blend-3 directory")
            scaler = joblib.load(SCALER_PATH)
            top_features = joblib.load(FEATURES_PATH)
            with open(PARAMS_PATH, 'r') as f:
                best_params = json.load(f)
            print("✅ Loaded scaler, feature list, and hyperparameters.")

            # Load and preprocess the new data
            if not os.path.exists(test_filepath):
                raise FileNotFoundError(f"Test file not found: {test_filepath}")
            new_data_df = pd.read_csv(test_filepath)
            print(f"\n✅ Successfully loaded {len(new_data_df)} rows from '{test_filepath}'.")
            if new_data_df.empty:
                raise ValueError("Test data is empty")

            print("🧪 Applying feature engineering...")
            processed_features = Blend3Predictor.create_advanced_features(new_data_df)

            for col in top_features:
                if col not in processed_features.columns:
                    processed_features[col] = 0
            X_new = processed_features[top_features]
            if X_new.empty:
                raise ValueError("Processed features DataFrame is empty")

            print("🔧 Scaling features...")
            X_new_scaled = scaler.transform(X_new)

            new_data_tensor = torch.FloatTensor(X_new_scaled).to(CFG.DEVICE)
            all_predictions = []

            for path in MODEL_PATHS:
                print(f"-> Loading model from: {path}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")
                base_model = Blend3Predictor.FTTransformerV2(
                    num_continuous=len(top_features),
                    embed_dim=best_params['embed_dim'],
                    num_heads=best_params['num_heads'],
                    num_layers=best_params['num_layers'],
                    dropout=best_params['dropout']
                )

                state_dict = torch.load(path, map_location=CFG.DEVICE)
                is_swa_model = any(key.startswith('module.') for key in state_dict.keys())

                if is_swa_model:
                    print("   (Detected SWA model, using AveragedModel wrapper)")
                    from torch.optim.swa_utils import AveragedModel
                    model = AveragedModel(base_model)
                else:
                    print("   (Detected regular model)")
                    model = base_model

                model.load_state_dict(state_dict)
                model.to(CFG.DEVICE)
                model.eval()

                with torch.no_grad():
                    preds = model(new_data_tensor).cpu().numpy().flatten()
                    all_predictions.append(preds)

            final_predictions = np.mean(all_predictions, axis=0)
            if len(final_predictions) != len(new_data_df):
                raise ValueError(f"Final prediction length ({len(final_predictions)}) does not match test data length ({len(new_data_df)})")

            # Return predictions as a Series with ID
            result = pd.Series(final_predictions, index=new_data_df['ID'], name='BlendProperty3')
            if result.empty:
                raise ValueError("Prediction Series is empty")
            return result
        except Exception as e:
            print(f"Error in Blend3Predictor: {str(e)}")
            return None
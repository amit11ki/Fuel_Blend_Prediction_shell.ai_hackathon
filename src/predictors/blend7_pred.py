import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import gc
import joblib
import pickle
from pathlib import Path
from collections import OrderedDict

class Blend7Predictor:
    class CFG:
        """Configuration class for all parameters and file paths."""
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        NFOLDS = 5
        SEED = 42
        TTA_STEPS = 10
        TTA_NOISE_LEVEL = 0.015

    class BlendDataset(Dataset):
        """PyTorch Dataset for blend features."""
        def __init__(self, features, labels=None):
            self.features = torch.FloatTensor(features)  # Directly use NumPy array
            self.labels = None  # Labels are not needed for prediction
        def __len__(self): return len(self.features)
        def __getitem__(self, idx): return self.features[idx], 0

    class FTTransformerV2(nn.Module):
        """A simplified FT-Transformer for tabular data."""
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
    def set_seed(seed=CFG.SEED):
        """Sets the seed for consistent results."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        Blend7Predictor.set_seed()
        print("Starting prediction for BlendProperty7...")
        try:
            # Load data
            if not os.path.exists(test_filepath):
                raise FileNotFoundError(f"Test file not found: {test_filepath}")
            test_df = pd.read_csv(test_filepath)
            if test_df.empty:
                raise ValueError("Test data is empty")

            # Feature engineering
            processed_features = Blend7Predictor.create_advanced_features(test_df)
            if processed_features.empty:
                raise ValueError("Feature engineering resulted in empty DataFrame")

            # Load scaler and top features
            base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-7')
            scaler_path = os.path.join(base_dir, 'scaler.joblib')
            features_path = os.path.join(base_dir, 'feature_cols.pkl')
            if not os.path.exists(scaler_path) or not os.path.exists(features_path):
                raise FileNotFoundError("Missing scaler or top features file")
            scaler = joblib.load(scaler_path)
            top_features = joblib.load(features_path)

            # Prepare data
            for col in top_features:
                if col not in processed_features.columns:
                    processed_features[col] = 0
            X_test = processed_features[top_features]
            if X_test.empty:
                raise ValueError("Prepared test data is empty")
            X_test_scaled = scaler.transform(X_test)

            # Model prediction with TTA
            final_predictions = np.zeros(len(X_test_scaled))
            model_base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-7')
            for fold in range(Blend7Predictor.CFG.NFOLDS):
                model_path = os.path.join(model_base_dir, f'model_Stage2_Final_fold{fold+1}.pt')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                model = Blend7Predictor.FTTransformerV2(
                    num_continuous=len(top_features),
                    embed_dim=256,
                    num_heads=4,
                    num_layers=6,
                    dropout=0.09887100000886359
                ).to(Blend7Predictor.CFG.DEVICE)

                saved_state_dict = torch.load(model_path, map_location=Blend7Predictor.CFG.DEVICE)
                new_state_dict = OrderedDict()
                for k, v in saved_state_dict.items():
                    if k.startswith('module.'):
                        name = k[7:]
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                if 'n_averaged' in new_state_dict:
                    del new_state_dict['n_averaged']
                model.load_state_dict(new_state_dict)
                model.eval()

                fold_tta_preds = np.zeros(len(X_test_scaled))
                for i in range(Blend7Predictor.CFG.TTA_STEPS):
                    print(f"\r   ...TTA step {i+1}/{Blend7Predictor.CFG.TTA_STEPS}", end="")
                    noise = np.random.normal(0, Blend7Predictor.CFG.TTA_NOISE_LEVEL, X_test_scaled.shape)
                    test_loader = DataLoader(
                        Blend7Predictor.BlendDataset(X_test_scaled + noise),
                        batch_size=32,
                        shuffle=False
                    )
                    step_preds = []
                    with torch.no_grad():
                        for features, _ in test_loader:
                            features = features.to(Blend7Predictor.CFG.DEVICE)
                            outputs = model(features)
                            step_preds.append(outputs.cpu().numpy())
                    fold_tta_preds += np.concatenate(step_preds).flatten()
                fold_tta_preds /= Blend7Predictor.CFG.TTA_STEPS
                final_predictions += fold_tta_preds
                print("\n   Fold prediction complete.")

            final_predictions /= Blend7Predictor.CFG.NFOLDS

            # Return predictions as a Series with ID
            result = pd.Series(final_predictions, index=test_df['ID'], name='BlendProperty7')
            if result.empty:
                raise ValueError("Prediction Series is empty")
            return result
        except Exception as e:
            print(f"Error in Blend7Predictor: {str(e)}")
            return None
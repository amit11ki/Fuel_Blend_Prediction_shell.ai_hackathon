import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import warnings
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Warning filters
warnings.filterwarnings('ignore', category=UserWarning, message="The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.")
warnings.filterwarnings('ignore', category=FutureWarning, message="`torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.")
warnings.filterwarnings('ignore', category=FutureWarning, message="`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.")

class Blend2Predictor:
    @staticmethod
    def create_features_from_script_1(df):
        """Features designed for linear models, Gaussian Processes, and potentially Transformers."""
        df_fe = df.copy()
        df_fe.columns = df_fe.columns.str.strip()

        component_fraction_cols = [f'Component{i}_fraction' for i in range(1, 6)]
        component_property_cols = {f'Component{i}_Property{j}' for i in range(1, 6) for j in range(1, 11)}

        for i in range(1, 6):
            for j in range(1, 11):
                if f'Component{i}_fraction' in df_fe.columns and f'Component{i}_Property{j}' in df_fe.columns:
                    df_fe[f'Interaction_C{i}_P{j}'] = df_fe[f'Component{i}_fraction'] * df_fe[f'Component{i}_Property{j}']
                else:
                    df_fe[f'Interaction_C{i}_P{j}'] = 0.0

        for j in range(1, 11):
            prop_cols = [f'Component{i}_Property{j}' for i in range(1, 6) if f'Component{i}_Property{j}' in df_fe.columns]
            if prop_cols:
                df_fe[f'Mean_Property{j}'] = df_fe[prop_cols].mean(axis=1)
                df_fe[f'Std_Property{j}'] = df_fe[prop_cols].std(axis=1).fillna(0)
                df_fe[f'Min_Property{j}'] = df_fe[prop_cols].min(axis=1)
                df_fe[f'Max_Property{j}'] = df_fe[prop_cols].max(axis=1)
            else:
                df_fe[f'Mean_Property{j}'] = 0.0
                df_fe[f'Std_Property{j}'] = 0.0
                df_fe[f'Min_Property{j}'] = 0.0
                df_fe[f'Max_Property{j}'] = 0.0

        blend_composition_cols = [col for col in component_fraction_cols if col in df_fe.columns]
        if len(blend_composition_cols) > 0:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(df_fe[blend_composition_cols])
            poly_feature_names = poly.get_feature_names_out(blend_composition_cols)
            df_poly_frac = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_fe.index)
            new_poly_cols = [col for col in poly_feature_names if col not in df_fe.columns]
            df_fe = pd.concat([df_fe, df_poly_frac[new_poly_cols]], axis=1)

        for i in range(1, 5):
            for j in range(i + 1, 6):
                frac_i = f'Component{i}_fraction'
                frac_j = f'Component{j}_fraction'
                if frac_i in df_fe.columns and frac_j in df_fe.columns:
                    df_fe[f'Ratio_C{i}_C{j}'] = df_fe[frac_i] / (df_fe[frac_j] + 1e-6)
                    df_fe[f'Diff_C{i}_C{j}'] = df_fe[frac_i] - df_fe[frac_j]

        for j in range(1, 11):
            weighted_prop_sum = pd.Series(0.0, index=df_fe.index)
            for i in range(1, 6):
                frac_col = f'Component{i}_fraction'
                prop_col = f'Component{i}_Property{j}'
                if frac_col in df_fe.columns and prop_col in df_fe.columns:
                    weighted_prop_sum += df_fe[frac_col] * df_fe[prop_col]
            df_fe[f'WeightedSum_Property{j}'] = weighted_prop_sum

        df_fe.fillna(0, inplace=True)
        return df_fe

    class FeatureTokenizer(nn.Module):
        def __init__(self, n_features, d_token):
            super().__init__()
            self.n_features = n_features
            self.d_token = d_token
            self.linear = nn.Linear(1, d_token)
            self.pos_embedding = nn.Parameter(torch.randn(1, n_features, d_token))

        def forward(self, x_num):
            x_num = x_num.unsqueeze(-1)
            embeddings = self.linear(x_num)
            embeddings = embeddings + self.pos_embedding
            return embeddings

    class TransformerBlock(nn.Module):
        def __init__(self, d_token, n_heads, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(d_token)
            self.ffn = nn.Sequential(
                nn.Linear(d_token, 4 * d_token),
                nn.GELU(),
                nn.Linear(4 * d_token, d_token)
            )
            self.norm2 = nn.LayerNorm(d_token)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            attn_output, _ = self.attn(x, x, x)
            x = self.norm1(x + self.dropout(attn_output))
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))
            return x

    class FTTransformer(nn.Module):
        def __init__(self, n_features, d_token, n_blocks, n_heads, dropout):
            super().__init__()
            self.feature_tokenizer = Blend2Predictor.FeatureTokenizer(n_features, d_token)
            self.transformer_blocks = nn.ModuleList([
                Blend2Predictor.TransformerBlock(d_token, n_heads, dropout) for _ in range(n_blocks)
            ])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, d_token // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token // 2, 1)
            )

        def forward(self, x_num):
            x = self.feature_tokenizer(x_num)
            cls_token_expanded = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token_expanded, x], dim=1)
            for block in self.transformer_blocks:
                x = block(x)
            cls_output = x[:, 0, :]
            output = self.mlp_head(cls_output)
            return output.squeeze(-1)

    @staticmethod
    def predict(test_filepath):
        # GPU setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load data
        print("Loading test data...")
        try:
            test_df = pd.read_csv(test_filepath)
            if test_df.empty:
                raise ValueError("Test data is empty")
        except Exception as e:
            print(f"Error loading test data: {str(e)}")
            return None

        # Apply feature engineering
        print("Applying feature engineering to test data...")
        try:
            test_fe_df = Blend2Predictor.create_features_from_script_1(test_df.copy())
            if test_fe_df.empty:
                raise ValueError("Feature engineering resulted in empty DataFrame")
        except Exception as e:
            print(f"Error in feature engineering: {str(e)}")
            return None

        # Define features
        feature_cols = [col for col in test_fe_df.columns if col != 'ID']
        feature_cols = sorted(feature_cols)
        X_test = test_fe_df[feature_cols]
        test_ids = test_df['ID']

        print(f"Number of features after engineering: {X_test.shape[1]}")

        # Load scaler and top_features
        scaler_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-2', 'scaler.pkl')
        top_features_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-2', 'top_features.pkl')
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(top_features_path, 'rb') as f:
                top_features = pickle.load(f)
            if not top_features:
                raise ValueError("Top features list is empty")
        except Exception as e:
            print(f"Error loading scaler or top_features: {str(e)}")
            return None

        # Scale test data
        try:
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
        except Exception as e:
            print(f"Error scaling test data: {str(e)}")
            return None

        # Select top features
        available_top_features = [f for f in top_features if f in X_test_scaled_df.columns]
        if len(available_top_features) < len(top_features):
            print(f"Warning: {len(top_features) - len(available_top_features)} features missing in test data. Using available ones.")
        X_test_selected = X_test_scaled_df[available_top_features]

        print(f"Number of features after selection: {X_test_selected.shape[1]}")

        # Model Hyperparameters
        d_token = 400
        n_blocks = 8
        n_heads = 8
        dropout = 0.002
        n_splits = 3

        # Load models and predict
        fold_test_predictions = []
        model_base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-2')
        for fold in range(1, n_splits + 1):
            model_path = os.path.join(model_base_dir, f'best_fttransformer_model_fold_{fold}.pth')
            try:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                model = Blend2Predictor.FTTransformer(X_test_selected.shape[1], d_token, n_blocks, n_heads, dropout).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
            except Exception as e:
                print(f"Error loading model {model_path}: {str(e)}")
                return None

            try:
                X_test_tensor = torch.tensor(X_test_selected.values, dtype=torch.float32).to(device)
                if X_test_tensor.dim() != 2:
                    raise ValueError(f"X_test_tensor has incorrect dimensions: {X_test_tensor.dim()}")
                fold_test_preds = []
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        test_preds_tensor = model(X_test_tensor)
                    fold_test_preds = test_preds_tensor.cpu().numpy()
                if len(fold_test_preds) != len(test_ids):
                    raise ValueError(f"Prediction length ({len(fold_test_preds)}) does not match test IDs length ({len(test_ids)})")
                fold_test_predictions.append(fold_test_preds)
            except Exception as e:
                print(f"Error during prediction for fold {fold}: {str(e)}")
                return None

        # Average Test Predictions
        try:
            final_test_predictions = np.mean(fold_test_predictions, axis=0)
            if len(final_test_predictions) != len(test_ids):
                raise ValueError(f"Final prediction length ({len(final_test_predictions)}) does not match test IDs length ({len(test_ids)})")
        except Exception as e:
            print(f"Error averaging predictions: {str(e)}")
            return None

        # Return predictions as a Series with ID
        try:
            result = pd.Series(final_test_predictions, index=test_ids, name='BlendProperty2')
            if result.empty:
                raise ValueError("Prediction Series is empty")
            return result
        except Exception as e:
            print(f"Error creating output Series: {str(e)}")
            return None
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb  # Import LightGBM
import os
import warnings
import pickle
from pathlib import Path

# Warning filters remain unchanged
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message="The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate."
)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message="`torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead."
)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message="`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead."
)

class Blend4Trainer:
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
            self.feature_tokenizer = Blend4Trainer.FeatureTokenizer(n_features, d_token)
            self.transformer_blocks = nn.ModuleList([
                Blend4Trainer.TransformerBlock(d_token, n_heads, dropout) for _ in range(n_blocks)
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
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    @staticmethod
    def calculate_mape(y_true, y_pred, epsilon=1e-8):
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    @staticmethod
    def select_top_features(X, y, feature_cols, n_features_to_select=50):
        """Select top N features using LightGBM feature importance."""
        print("Performing feature selection with LightGBM...")

        # Prepare LightGBM dataset
        lgb_train = lgb.Dataset(X, label=y, feature_name=feature_cols)

        # LightGBM parameters for feature selection
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 42,
            'verbose': -1
        }

        # Train LightGBM model
        lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_train],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )

        # Get feature importance
        importance = lgb_model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        # Select top N features
        top_features = feature_importance['Feature'].head(n_features_to_select).tolist()
        print(f"Selected top {len(top_features)} features.")

        return top_features, feature_importance

    @staticmethod
    def train():
        # GPU setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load data from data directory
        data_dir = Path(__file__).parent.parent.parent / 'data'
        print("Loading training data...")
        try:
            train_df = pd.read_csv(data_dir / 'train.csv')
        except FileNotFoundError:
            print(f"Error: train.csv not found in {data_dir}. Please ensure the file exists.")
            return

        # Apply feature engineering
        print("Applying feature engineering to training data...")
        train_fe_df = Blend4Trainer.create_features_from_script_1(train_df.copy())

        # Define target and features
        target_column = 'BlendProperty4'
        feature_cols = [col for col in train_fe_df.columns if not col.startswith('BlendProperty') and col != 'ID']
        feature_cols = sorted(feature_cols)  # Sort alphabetically for consistency

        X = train_fe_df[feature_cols]
        y = train_fe_df[target_column]

        print(f"Number of features after engineering: {X.shape[1]}")

        # Feature Selection
        # Scale features before feature selection for LightGBM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        # Select top 50 features using LightGBM
        top_features, feature_importance = Blend4Trainer.select_top_features(X_scaled_df, y, feature_cols, n_features_to_select=50)

        # Update features to only include top 50
        X_selected = X_scaled_df[top_features]

        print(f"Number of features after selection: {X_selected.shape[1]}")

        # Save scaler and top_features
        model_dir = Path(__file__).parent.parent.parent / 'models' / 'Blend-4'
        model_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = model_dir / 'scaler.pkl'
        top_features_path = model_dir / 'top_features.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(top_features_path, 'wb') as f:
            pickle.dump(top_features, f)
        print(f"Scaler saved to {scaler_path}, Top features saved to {top_features_path}")

        # K-Fold Cross-Validation Setup
        n_splits = 3
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_predictions = np.zeros(len(X))

        # Model Hyperparameters
        n_features = X_selected.shape[1]  # Update to number of selected features
        d_token = 400
        n_blocks = 8
        n_heads = 8
        dropout = 0.002
        batch_size = 128
        n_epochs = 500
        patience = 125
        learning_rate = 1e-4
        weight_decay = 1e-5
        grad_clip_norm = 1.0

        print(f"\nTraining with {n_splits}-Fold Cross-Validation...")

        for fold, (train_index, val_index) in enumerate(kf.split(X_selected, y)):
            print(f"\n--- Fold {fold+1}/{n_splits} ---")

            X_train_fold, X_val_fold = X_selected.iloc[train_index], X_selected.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_fold.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_fold.values, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_fold.values, dtype=torch.float32)

            # Create DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            # Model, Optimizer, Loss
            model = Blend4Trainer.FTTransformer(n_features, d_token, n_blocks, n_heads, dropout).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
            scaler_amp = torch.amp.GradScaler('cuda')

            best_val_loss = float('inf')
            epochs_no_improve = 0
            model_path = model_dir / f'best_fttransformer_model_fold_{fold+1}.pth'

            for epoch in range(n_epochs):
                model.train()
                running_loss = 0.0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    running_loss += loss.item() * inputs.size(0)

                train_loss = running_loss / len(train_dataset)

                # Validation
                model.eval()
                val_preds = []
                val_true = []
                val_loss_sum = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            val_loss_sum += criterion(outputs, targets).item() * inputs.size(0)
                        val_preds.extend(outputs.cpu().numpy())
                        val_true.extend(targets.cpu().numpy())

                val_loss = val_loss_sum / len(val_dataset)
                val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
                val_mae = mean_absolute_error(val_true, val_preds)
                val_mape = Blend4Trainer.calculate_mape(np.array(val_true), np.array(val_preds))

                print(f"  Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val MAPE: {val_mape:.2f}%")

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"  Early stopping triggered for Fold {fold+1} after {patience} epochs without improvement.")
                        break

            print(f"Loading best model for Fold {fold+1} for OOF predictions.")
            model.load_state_dict(torch.load(model_path))

            # OOF predictions
            model.eval()
            fold_val_preds = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                    fold_val_preds.extend(outputs.cpu().numpy())
            oof_predictions[val_index] = np.array(fold_val_preds)

        print("\n--- K-Fold Training Complete ---")

        # Final Evaluation
        print("\n--- Overall Out-of-Fold (OOF) Evaluation ---")
        overall_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        overall_oof_mae = mean_absolute_error(y, oof_predictions)
        overall_oof_mape = Blend4Trainer.calculate_mape(y.values, oof_predictions)

        print(f"Overall OOF RMSE: {overall_oof_rmse:.4f}")
        print(f"Overall OOF MAE: {overall_oof_mae:.4f}")
        print(f"Overall OOF MAPE: {overall_oof_mape:.2f}%")

        print("\nTraining complete. Models, scaler, and top_features saved.")

if __name__ == "__main__":
    Blend4Trainer.train()
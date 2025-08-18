import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
import os
import gc
import random
import joblib
import json
import lightgbm as lgb
from pathlib import Path

class Blend8Trainer:
    class CFG:
        """Configuration class for STATIC hyperparameters."""
        # Workflow
        TARGET_COL = 'BlendProperty8'
        NFOLDS = 5
        SEED = 42
        NUM_TOP_FEATURES = 28
        
        # ADDED: Define a single output directory for all files
        OUTPUT_DIR = Path(__file__).parent.parent.parent / 'models' / 'Blend-8'

        # Training Epochs/Patience for the FINAL model
        FINAL_MODEL_EPOCHS = 300
        FINAL_MODEL_PATIENCE = 35
        SWA_LEARNING_RATE = 1e-4
        SWA_START_EPOCH_RATIO = 0.7
        TTA_STEPS = 10
        TTA_NOISE_LEVEL = 0.015
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class EarlyStopping:
        def __init__(self, patience=7, verbose=False, path='checkpoint.pt'):
            self.patience, self.verbose, self.path = patience, verbose, path
            self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.inf

        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None or score > self.best_score:
                if self.verbose: print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
                torch.save(model.state_dict(), self.path)
                self.best_score, self.val_loss_min = score, val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose and self.counter % 5 == 0: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience: self.early_stop = True

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

    class BlendDataset(Dataset):
        def __init__(self, features, labels=None):
            self.features = torch.FloatTensor(features.values)
            self.labels = torch.FloatTensor(labels.values).view(-1, 1) if labels is not None else None

        def __len__(self): return len(self.features)

        def __getitem__(self, idx):
            if self.labels is not None:
                return self.features[idx], self.labels[idx]
            return self.features[idx], torch.zeros(1)

    @staticmethod
    def set_seed(seed=CFG.SEED):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
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
    def run_training_workflow(X, y, X_test, scaler, params, n_folds, epochs, patience, run_name=""):
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        cv_tta_preds = np.zeros(len(X_test))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=Blend8Trainer.CFG.SEED)
        
        fold_model_paths = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
            print(f"\n--- {run_name} Fold {fold + 1}/{n_folds} ---")
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_loader = DataLoader(Blend8Trainer.BlendDataset(X_train, y_train), batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(Blend8Trainer.BlendDataset(X_val, y_val), batch_size=params['batch_size'], shuffle=False)
            
            model = Blend8Trainer.FTTransformerV2(
                len(X.columns), params['embed_dim'], params['num_heads'], params['num_layers'], params['dropout']
            ).to(Blend8Trainer.CFG.DEVICE)
            
            loss_fn = nn.HuberLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            
            # Save checkpoints to the output directory
            checkpoint_filename = f'checkpoint_{run_name}_fold{fold+1}.pt'
            checkpoint_path = Blend8Trainer.CFG.OUTPUT_DIR / checkpoint_filename
            early_stopping = Blend8Trainer.EarlyStopping(patience=patience, verbose=False, path=checkpoint_path)
            
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=Blend8Trainer.CFG.SWA_LEARNING_RATE)
            swa_start_epoch = int(epochs * Blend8Trainer.CFG.SWA_START_EPOCH_RATIO)
            
            for epoch in range(epochs):
                model.train()
                for features, labels in train_loader:
                    features, labels = features.to(Blend8Trainer.CFG.DEVICE), labels.to(Blend8Trainer.CFG.DEVICE)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip_norm'])
                    optimizer.step()
                
                if epoch < swa_start_epoch:
                    scheduler.step()
                else:
                    if epoch == swa_start_epoch: print("--- SWA training started ---")
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    val_loss = sum(loss_fn(model(features.to(Blend8Trainer.CFG.DEVICE)), labels.to(Blend8Trainer.CFG.DEVICE)).item() for features, labels in val_loader)
                avg_val_loss = val_loss / len(val_loader)
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping on epoch {epoch+1}")
                    break
            
            final_model_for_prediction = swa_model if epoch >= swa_start_epoch else model
            if epoch < swa_start_epoch:
                final_model_for_prediction.load_state_dict(torch.load(checkpoint_path))
            
            torch.optim.swa_utils.update_bn(train_loader, final_model_for_prediction, device=Blend8Trainer.CFG.DEVICE)
            final_model_for_prediction.eval()
            
            fold_tta_preds = np.zeros(len(X_test_scaled))
            for _ in range(Blend8Trainer.CFG.TTA_STEPS):
                test_loader = DataLoader(Blend8Trainer.BlendDataset(X_test_scaled + np.random.normal(0, Blend8Trainer.CFG.TTA_NOISE_LEVEL, X_test_scaled.shape)), batch_size=params['batch_size'], shuffle=False)
                with torch.no_grad():
                    step_preds = np.concatenate([final_model_for_prediction(features.to(Blend8Trainer.CFG.DEVICE)).cpu().numpy() for features, _ in test_loader])
                fold_tta_preds += step_preds.flatten()
            cv_tta_preds += fold_tta_preds / Blend8Trainer.CFG.TTA_STEPS / n_folds

            # Save final models to the output directory
            final_model_filename = f'final_model_{run_name}_fold{fold+1}.pth'
            final_model_path = Blend8Trainer.CFG.OUTPUT_DIR / final_model_filename
            torch.save(final_model_for_prediction.state_dict(), final_model_path)
            fold_model_paths.append(final_model_path)
            print(f"💾 Model for fold {fold+1} saved to {final_model_path}")
        
        return cv_tta_preds, fold_model_paths

    @staticmethod
    def train():
        print(f"🚀 Starting DIRECT training workflow for target '{Blend8Trainer.CFG.TARGET_COL}' on device: {Blend8Trainer.CFG.DEVICE}")
        Blend8Trainer.CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📂 All output files will be saved to the '{Blend8Trainer.CFG.OUTPUT_DIR}/' folder.")

        # Load Data
        data_dir = Path(__file__).parent.parent.parent / 'data'
        try:
            train_df = pd.read_csv(data_dir / 'train.csv')
            test_df = pd.read_csv(data_dir / 'test.csv')
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure train.csv and test.csv are in {data_dir}.")
            return

        # Feature Engineering, Selection, and Scaler fitting
        print("🧪 Creating advanced feature sets...")
        train_features_full = Blend8Trainer.create_advanced_features(train_df)
        test_features_full = Blend8Trainer.create_advanced_features(test_df)
        all_blend_properties = [f'BlendProperty{i}' for i in range(1, 11)]
        feature_cols = [col for col in train_features_full.columns if col not in all_blend_properties and col != 'ID']
        X = train_features_full[feature_cols].copy()
        y = train_features_full[[Blend8Trainer.CFG.TARGET_COL]].copy()
        X_test = test_features_full[feature_cols].copy()
        print("\n🚀 Starting feature selection with LightGBM...")
        lgbm = lgb.LGBMRegressor(random_state=Blend8Trainer.CFG.SEED, n_estimators=200, learning_rate=0.05)
        lgbm.fit(X, y.values.ravel())
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': lgbm.feature_importances_
        }).sort_values('importance', ascending=False)
        top_features = feature_importances.head(Blend8Trainer.CFG.NUM_TOP_FEATURES)['feature'].tolist()
        print(f"✅ Selected top {len(top_features)} features.")
        X = X[top_features]
        X_test = X_test[top_features]
        print("\n🔧 Fitting the feature scaler on selected features...")
        scaler = StandardScaler()
        scaler.fit(X)

        # Use pre-defined optimal hyperparameters
        best_params = {
            'embed_dim': 512, 
            'num_layers': 6, 
            'num_heads': 8, 
            'dropout': 0.10853875455024573, 
            'batch_size': 32, 
            'learning_rate': 0.00012093134774024999, 
            'weight_decay': 1.6104723248832257e-05, 
            'gradient_clip_norm': 0.8321252164223925
        }
        print("\n⚙ Using pre-defined optimal hyperparameters:")
        print(best_params)

        # Train the final model
        print(f"\n{'='*20} TRAINING FINAL MODEL ON SELECTED FEATURES {'='*20}")
        final_predictions, final_model_paths = Blend8Trainer.run_training_workflow(
            X, y, X_test, scaler, best_params,
            Blend8Trainer.CFG.NFOLDS, Blend8Trainer.CFG.FINAL_MODEL_EPOCHS, Blend8Trainer.CFG.FINAL_MODEL_PATIENCE,
            "Final_Model_Fixed_Params"
        )
        print("✅ Final model training complete.")
        print(f"Models saved at: {final_model_paths}")

        # Save all necessary assets for prediction to the output directory
        print(f"\n{'='*20} SAVING PREDICTION ASSETS {'='*20}")
        joblib.dump(scaler, Blend8Trainer.CFG.OUTPUT_DIR / 'scaler.gz')
        print(f"🔧 Scaler saved to {Blend8Trainer.CFG.OUTPUT_DIR / 'scaler.gz'}")
        joblib.dump(top_features, Blend8Trainer.CFG.OUTPUT_DIR / 'top_features.list')
        print(f"📊 Top features list saved to {Blend8Trainer.CFG.OUTPUT_DIR / 'top_features.list'}")
        with open(Blend8Trainer.CFG.OUTPUT_DIR / 'best_params.json', 'w') as f:
            json.dump(best_params, f)
        print(f"⚙ Best hyperparameters saved to {Blend8Trainer.CFG.OUTPUT_DIR / 'best_params.json'}")

        # Create Final Submission in the output directory
        print(f"\n{'='*20} CREATING FINAL SUBMISSION FOR {Blend8Trainer.CFG.TARGET_COL} {'='*20}")
        submission_df = pd.DataFrame({'ID': test_df['ID'], Blend8Trainer.CFG.TARGET_COL: final_predictions})
        submission_path = Blend8Trainer.CFG.OUTPUT_DIR / f'submission_{Blend8Trainer.CFG.TARGET_COL}.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"\n🎉 Submission file created successfully at: {submission_path}")

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    Blend8Trainer.train()
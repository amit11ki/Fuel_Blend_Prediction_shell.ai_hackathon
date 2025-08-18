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
import pickle
from pathlib import Path

class Blend7Trainer:
    class CFG:
        """Configuration class for all parameters and file paths."""
        # Core Workflow
        TARGET_COL = 'BlendProperty7'
        NFOLDS = 5
        SEED = 42
        
        # Paths & Directories
        OUTPUT_DIR = Path(__file__).parent.parent.parent / 'models' / 'Blend-7'
        
        # Optimal Hyperparameters (Hardcoded from your Optuna run)
        FINAL_PARAMS = {
            'embed_dim': 256,
            'num_layers': 6,
            'num_heads': 4,
            'dropout': 0.09887100000886359,
            'batch_size': 32,
            'learning_rate': 0.0002929119804481019,
            'weight_decay': 0.0009221757765332223,
            'gradient_clip_norm': 1.674065474287529
        }

        # Training Stage Settings
        STAGE_1_EPOCHS = 150
        STAGE_1_PATIENCE = 25
        STAGE_2_EPOCHS = 80
        STAGE_2_PATIENCE = 15

        # Advanced Techniques
        SWA_LEARNING_RATE = 1e-4
        SWA_START_EPOCH_RATIO = 0.7
        TTA_STEPS = 10
        TTA_NOISE_LEVEL = 0.015

        # System
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
        """Sets the seed for consistent results."""
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
    def run_training_workflow(X, y, X_test, scaler, run_name, params, epochs, patience):
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        cv_tta_preds = np.zeros(len(X_test))
        kf = KFold(n_splits=Blend7Trainer.CFG.NFOLDS, shuffle=True, random_state=Blend7Trainer.CFG.SEED)
        
        fold_model_paths = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
            print(f"\n--- {run_name} Fold {fold + 1}/{Blend7Trainer.CFG.NFOLDS} ---")
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_loader = DataLoader(Blend7Trainer.BlendDataset(X_train, y_train), batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(Blend7Trainer.BlendDataset(X_val, y_val), batch_size=params['batch_size'], shuffle=False)
            
            model = Blend7Trainer.FTTransformerV2(
                len(X.columns), params['embed_dim'], params['num_heads'], params['num_layers'], params['dropout']
            ).to(Blend7Trainer.CFG.DEVICE)
            
            loss_fn = nn.HuberLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            
            checkpoint_filename = f'checkpoint_{run_name}_fold{fold+1}.pt'
            checkpoint_path = Blend7Trainer.CFG.OUTPUT_DIR / checkpoint_filename
            early_stopping = Blend7Trainer.EarlyStopping(patience=patience, verbose=False, path=checkpoint_path)
            
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=Blend7Trainer.CFG.SWA_LEARNING_RATE)
            swa_start_epoch = int(epochs * Blend7Trainer.CFG.SWA_START_EPOCH_RATIO)
            
            for epoch in range(epochs):
                model.train()
                for features, labels in train_loader:
                    features, labels = features.to(Blend7Trainer.CFG.DEVICE), labels.to(Blend7Trainer.CFG.DEVICE)
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
                    val_loss = sum(loss_fn(model(features.to(Blend7Trainer.CFG.DEVICE)), labels.to(Blend7Trainer.CFG.DEVICE)).item() for features, labels in val_loader)
                avg_val_loss = val_loss / len(val_loader)
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping on epoch {epoch+1}")
                    break
            
            final_model_for_prediction = swa_model if epoch >= swa_start_epoch else model
            if epoch < swa_start_epoch:
                final_model_for_prediction.load_state_dict(torch.load(checkpoint_path))
            
            torch.optim.swa_utils.update_bn(train_loader, final_model_for_prediction, device=Blend7Trainer.CFG.DEVICE)
            final_model_for_prediction.eval()
            
            fold_tta_preds = np.zeros(len(X_test_scaled))
            for _ in range(Blend7Trainer.CFG.TTA_STEPS):
                test_loader = DataLoader(Blend7Trainer.BlendDataset(X_test_scaled + np.random.normal(0, Blend7Trainer.CFG.TTA_NOISE_LEVEL, X_test_scaled.shape)), batch_size=params['batch_size'], shuffle=False)
                with torch.no_grad():
                    step_preds = np.concatenate([final_model_for_prediction(features.to(Blend7Trainer.CFG.DEVICE)).cpu().numpy() for features, _ in test_loader])
                fold_tta_preds += step_preds.flatten()
            cv_tta_preds += fold_tta_preds / Blend7Trainer.CFG.TTA_STEPS / Blend7Trainer.CFG.NFOLDS

            final_model_filename = f'final_model_{run_name}_fold{fold+1}.pth'
            final_model_path = Blend7Trainer.CFG.OUTPUT_DIR / final_model_filename
            torch.save(final_model_for_prediction.state_dict(), final_model_path)
            fold_model_paths.append(final_model_path)
            print(f"💾 Model for fold {fold+1} saved to {final_model_path}")
        
        return cv_tta_preds, fold_model_paths

    @staticmethod
    def train():
        print(f"🚀 Starting DIRECT training workflow for target '{Blend7Trainer.CFG.TARGET_COL}' on device: {Blend7Trainer.CFG.DEVICE}")
        Blend7Trainer.CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📂 All output files will be saved to the '{Blend7Trainer.CFG.OUTPUT_DIR}/' folder.")

        # Load data
        data_dir = Path(__file__).parent.parent.parent / 'data'
        try:
            train_df = pd.read_csv(data_dir / 'train.csv')
            test_df = pd.read_csv(data_dir / 'test.csv')
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure train.csv and test.csv are in {data_dir}.")
            return

        # Feature engineering
        print("🧪 Creating advanced feature sets...")
        train_fe = Blend7Trainer.create_advanced_features(train_df)
        test_fe = Blend7Trainer.create_advanced_features(test_df)

        # Prepare data and save artifacts for inference
        all_blend_props = [f'BlendProperty{i}' for i in range(1, 11)]
        feature_cols = [col for col in train_fe.columns if col not in all_blend_props and col != 'ID']
        X, y = train_fe[feature_cols], train_fe[[Blend7Trainer.CFG.TARGET_COL]]
        X_test = test_fe[feature_cols]

        print("💾 Saving feature columns and scaler for inference...")
        with open(Blend7Trainer.CFG.OUTPUT_DIR / 'feature_cols.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, Blend7Trainer.CFG.OUTPUT_DIR / 'scaler.joblib')
        
        # === STAGE 1: Generate Pseudo-Labels ===
        print(f"\n{'='*20} STAGE 1: Generating Pseudo-Labels {'='*20}")
        pseudo_labels = Blend7Trainer.run_training_workflow(
            X, y, X_test, scaler, "Stage1", Blend7Trainer.CFG.FINAL_PARAMS, Blend7Trainer.CFG.STAGE_1_EPOCHS, Blend7Trainer.CFG.STAGE_1_PATIENCE
        )
        min_target, max_target = y[Blend7Trainer.CFG.TARGET_COL].min(), y[Blend7Trainer.CFG.TARGET_COL].max()
        pseudo_labels = np.clip(pseudo_labels, min_target, max_target)
        print("✅ Stage 1 complete. Pseudo-labels generated.")

        # === STAGE 2: Train on Combined Data ===
        print(f"\n{'='*20} STAGE 2: Training on Combined Data {'='*20}")
        pseudo_df = X_test.copy()
        pseudo_df[Blend7Trainer.CFG.TARGET_COL] = pseudo_labels
        X_combined = pd.concat([X, pseudo_df[feature_cols]], ignore_index=True)
        y_combined = pd.concat([y, pseudo_df[[Blend7Trainer.CFG.TARGET_COL]]], ignore_index=True)
        print(f"Created combined dataset with {len(X_combined)} samples.")

        final_predictions = Blend7Trainer.run_training_workflow(
            X_combined, y_combined, X_test, scaler, "Stage2_Final", Blend7Trainer.CFG.FINAL_PARAMS, Blend7Trainer.CFG.STAGE_2_EPOCHS, Blend7Trainer.CFG.STAGE_2_PATIENCE
        )
        print("✅ Stage 2 complete.")

        # === Final Submission ===
        print(f"\n{'='*20} CREATING FINAL SUBMISSION {'='*20}")
        submission_df = pd.DataFrame({'ID': test_df['ID'], Blend7Trainer.CFG.TARGET_COL: final_predictions})
        submission_path = Blend7Trainer.CFG.OUTPUT_DIR / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"\n🎉 Workflow finished! Submission file '{submission_path}' and all artifacts are ready.")

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    Blend7Trainer.train()
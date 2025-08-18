import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

class Blend5Trainer:
    @staticmethod
    def apply_feature_engineering(df):
        df_fe = df.copy()
        df_fe.columns = df_fe.columns.str.strip()
        component_fraction_cols = [f'Component{i}_fraction' for i in range(1, 6)]

        # Interaction terms
        for i in range(1, 6):
            for j in range(1, 11):
                if f'Component{i}_fraction' in df_fe.columns and f'Component{i}_Property{j}' in df_fe.columns:
                    df_fe[f'Interaction_C{i}_P{j}'] = df_fe[f'Component{i}_fraction'] * df_fe[f'Component{i}_Property{j}']
                else:
                    df_fe[f'Interaction_C{i}_P{j}'] = 0.0

        # Property stats
        for j in range(1, 11):
            prop_cols = [f'Component{i}_Property{j}' for i in range(1, 6) if f'Component{i}_Property{j}' in df_fe.columns]
            if prop_cols:
                df_fe[f'Mean_Property{j}'] = df_fe[prop_cols].mean(axis=1)
                df_fe[f'Std_Property{j}'] = df_fe[prop_cols].std(axis=1).fillna(0)
                df_fe[f'Min_Property{j}'] = df_fe[prop_cols].min(axis=1)
                df_fe[f'Max_Property{j}'] = df_fe[prop_cols].max(axis=1)

        # Polynomial features
        if component_fraction_cols:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(df_fe[component_fraction_cols])
            poly_feature_names = poly.get_feature_names_out(component_fraction_cols)
            df_poly_frac = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_fe.index)
            new_poly_cols = [col for col in poly_feature_names if col not in df_fe.columns]
            df_fe = pd.concat([df_fe, df_poly_frac[new_poly_cols]], axis=1)

        df_fe.fillna(0, inplace=True)
        return df_fe

    @staticmethod
    def train():
        # Load data from data directory
        data_dir = Path(__file__).parent.parent.parent / 'data'
        try:
            df_train = pd.read_csv(data_dir / 'train.csv')
        except FileNotFoundError:
            print(f"Error: train.csv not found in {data_dir}. Please ensure the file exists.")
            return

        # Apply feature engineering
        df_train_fe = Blend5Trainer.apply_feature_engineering(df_train)

        # Define features & target
        features_cols = [col for col in df_train_fe.columns if col not in [f'BlendProperty{i}' for i in range(1, 11)] and col != 'ID']
        X_full_train = df_train_fe[features_cols]
        y_full_train = df_train_fe['BlendProperty5']

        # Step 1: Initial training to get top features
        print("--- 📊 Step 1: Training with all features to find top 8 ---")
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_full_train, y_full_train, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train_split, y_train_split)
        val_preds = model.predict(X_val)

        mape_score = mean_absolute_percentage_error(y_val, val_preds) * 100
        rmse_score = np.sqrt(mean_squared_error(y_val, val_preds))
        mae_score = mean_absolute_error(y_val, val_preds)

        print(f"MAPE: {mape_score:.4f}% | RMSE: {rmse_score:.4f} | MAE: {mae_score:.4f}")

        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = 8
        top_features = [features_cols[indices[i]] for i in range(top_n)]

        print("\n🔝 Top 8 Features:")
        for i, feat in enumerate(top_features, 1):
            print(f"{i}. {feat} - {importances[indices[i-1]]:.6f}")

        # Save top features
        model_dir = Path(__file__).parent.parent.parent / 'models' / 'Blend-5'
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(top_features, model_dir / 'top_features.pkl')

        # Plot
        plt.figure(figsize=(8, 5))
        plt.barh(top_features[::-1], [importances[indices[i]] for i in range(top_n)][::-1], color='skyblue')
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Top 8 Features for BlendProperty5")
        plt.tight_layout()
        plt.savefig(model_dir / 'feature_importance.png')
        plt.close()

        # Step 2: Retrain with only top features
        print("\n--- 🚀 Step 2: Retraining with top 8 features only ---")
        X_full_train_top = X_full_train[top_features]
        final_model = RandomForestRegressor(random_state=42)
        final_model.fit(X_full_train_top, y_full_train)

        # Save the final model
        joblib.dump(final_model, model_dir / 'final_model.pkl')

        print("\n✅ Model and top features saved successfully!")

if __name__ == "__main__":
    Blend5Trainer.train()
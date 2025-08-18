import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import os
class Blend5Predictor:
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
    def predict(test_filepath):
        print("Starting prediction for BlendProperty5...")
        try:
            # Load data
            df_test = pd.read_csv(test_filepath)
            if df_test.empty:
                raise ValueError("Test data is empty")

            # Apply feature engineering
            df_test_fe = Blend5Predictor.apply_feature_engineering(df_test)
            if df_test_fe.empty:
                raise ValueError("Feature engineering resulted in empty DataFrame")

            # Load saved top features and model
            top_features = joblib.load(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-5', 'top_features.pkl'))
            final_model = joblib.load(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-5', 'final_model.pkl'))
            if not top_features or final_model is None:
                raise ValueError("Top features or model file is invalid")

            # Prepare test data with top features
            X_test = df_test_fe.reindex(columns=top_features, fill_value=0)
            if X_test.empty:
                raise ValueError("Prepared test data is empty")

            # Make predictions
            submission_predictions = final_model.predict(X_test)
            if len(submission_predictions) != len(df_test):
                raise ValueError(f"Prediction length ({len(submission_predictions)}) does not match test data length ({len(df_test)})")

            # Return predictions as a Series with ID
            result = pd.Series(submission_predictions, index=df_test['ID'], name='BlendProperty5')
            if result.empty:
                raise ValueError("Prediction Series is empty")
            return result
        except Exception as e:
            print(f"Error in Blend5Predictor: {str(e)}")
            return None
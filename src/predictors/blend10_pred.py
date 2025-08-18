import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class Blend10Predictor:
    @staticmethod
    def apply_feature_engineering(df):
        """
        Applies feature engineering to the raw data.
        This function should be identical in all training and prediction scripts.
        """
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

    @staticmethod
    def predict(test_filepath):
        print("Starting prediction for BlendProperty10...")
        try:
            # Load the trained model
            model_filename = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Blend-10', 'model_BlendProperty10.joblib')
            if not os.path.exists(model_filename):
                raise FileNotFoundError(f"Model file not found: {model_filename}")
            model = joblib.load(model_filename)

            # Load new data for prediction
            if not os.path.exists(test_filepath):
                raise FileNotFoundError(f"Test file not found: {test_filepath}")
            df_test = pd.read_csv(test_filepath)
            if df_test.empty:
                raise ValueError("Test data is empty")

            # Apply feature engineering
            df_test_fe = Blend10Predictor.apply_feature_engineering(df_test)
            if df_test_fe.empty:
                raise ValueError("Feature engineering resulted in empty DataFrame")

            # Align test set columns with training set columns
            features_cols = [col for col in df_test_fe.columns if not col.startswith('BlendProperty') and col != 'ID']
            X_test = df_test_fe.reindex(columns=model.named_steps['scaler'].feature_names_in_, fill_value=0)
            if X_test.empty:
                raise ValueError("Aligned test data is empty")

            # Make predictions
            predictions = model.predict(X_test)
            if len(predictions) != len(df_test):
                raise ValueError(f"Prediction length ({len(predictions)}) does not match test data length ({len(df_test)})")

            # Return predictions as a Series with ID
            result = pd.Series(predictions, index=df_test['ID'], name='BlendProperty10')
            if result.empty:
                raise ValueError("Prediction Series is empty")
            return result
        except Exception as e:
            print(f"Error in Blend10Predictor: {str(e)}")
            return None
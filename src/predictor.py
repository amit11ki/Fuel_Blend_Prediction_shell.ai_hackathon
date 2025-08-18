# import sys
# import os
# import pandas as pd
# from pathlib import Path
# from predictors.blend1_pred import Blend1Predictor
# from predictors.blend2_pred import Blend2Predictor
# from predictors.blend3_pred import Blend3Predictor
# from predictors.blend4_pred import Blend4Predictor
# from predictors.blend5_pred import Blend5Predictor
# from predictors.blend6_pred import Blend6Predictor
# from predictors.blend7_pred import Blend7Predictor
# from predictors.blend8_pred import Blend8Predictor
# from predictors.blend9_pred import Blend9Predictor
# from predictors.blend10_pred import Blend10Predictor

# # Adjust sys.path to include the project root
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

# def run_central_predictor(test_filepath):
#     predictors = {
#         1: Blend1Predictor,
#         2: Blend2Predictor,
#         3: Blend3Predictor,
#         4: Blend4Predictor,
#         5: Blend5Predictor,
#         6: Blend6Predictor,
#         7: Blend7Predictor,
#         8: Blend8Predictor,
#         9: Blend9Predictor,
#         10: Blend10Predictor
#     }
#     results = {}
#     for blend, predictor_class in predictors.items():
#         print(f"\nRunning prediction for BlendProperty{blend}...")
#         results[blend] = predictor_class.predict(test_filepath)

#     # Combine results into a single DataFrame
#     failed_blends = []
#     if all(results[blend] is not None for blend in results):
#         combined_df = pd.DataFrame(index=results[1].index)  # Use IDs from Blend1 as the index
#         for blend, preds in results.items():
#             combined_df[f'BlendProperty{blend}'] = preds
#         combined_df.index.name = 'ID'
#         combined_df.reset_index(inplace=True)

#         # Save combined predictions
#         output_filename = project_root / "predictions_all_blends.csv"
#         combined_df.to_csv(output_filename, index=False)
#         print(f"\n🎉 Combined predictions saved to '{output_filename}'!")
#         print("\n--- Combined DataFrame with Predictions ---")
#         print(combined_df.head())
#     else:
#         for blend, result in results.items():
#             if result is None:
#                 failed_blends.append(blend)
#         print(f"\n❌ Error: Predictions failed for BlendProperty{', BlendProperty'.join(map(str, failed_blends))}. Check the logs for details.")

#     return combined_df if all(results[blend] is not None for blend in results) else None

# if __name__ == "__main__":
#     # Use the project root to construct the test file path
#     base_path = Path(__file__).parent.parent
#     test_filepath = base_path / "data" / "test.csv"
#     run_central_predictor(test_filepath)

# import sys
# import os
# import pandas as pd
# from pathlib import Path

# # Add project root to sys.path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from src.predictors.blend1_pred import Blend1Predictor
# from src.predictors.blend2_pred import Blend2Predictor
# from src.predictors.blend3_pred import Blend3Predictor
# from src.predictors.blend4_pred import Blend4Predictor
# from src.predictors.blend5_pred import Blend5Predictor
# from src.predictors.blend6_pred import Blend6Predictor
# from src.predictors.blend7_pred import Blend7Predictor
# from src.predictors.blend8_pred import Blend8Predictor
# from src.predictors.blend9_pred import Blend9Predictor
# from src.predictors.blend10_pred import Blend10Predictor

# def run_central_predictor(test_filepath):
#     predictors = {
#         1: Blend1Predictor,
#         2: Blend2Predictor,
#         3: Blend3Predictor,
#         4: Blend4Predictor,
#         5: Blend5Predictor,
#         6: Blend6Predictor,
#         7: Blend7Predictor,
#         8: Blend8Predictor,
#         9: Blend9Predictor,
#         10: Blend10Predictor
#     }
#     results = {}
#     for blend, predictor_class in predictors.items():
#         print(f"\nRunning prediction for BlendProperty{blend}...")
#         results[blend] = predictor_class.predict(test_filepath)

#     # Combine results into a single DataFrame
#     failed_blends = []
#     if all(results[blend] is not None for blend in results):
#         combined_df = pd.DataFrame(index=results[1].index)  # Use IDs from Blend1 as the index
#         for blend, preds in results.items():
#             combined_df[f'BlendProperty{blend}'] = preds
#         combined_df.index.name = 'ID'
#         combined_df.reset_index(inplace=True)

#         # Save combined predictions
#         output_filename = project_root / "predictions_all_blends.csv"
#         combined_df.to_csv(output_filename, index=False)
#         print(f"\n🎉 Combined predictions saved to '{output_filename}'!")
#         print("\n--- Combined DataFrame with Predictions ---")
#         print(combined_df.head())
#     else:
#         for blend, result in results.items():
#             if result is None:
#                 failed_blends.append(blend)
#         print(f"\n❌ Error: Predictions failed for BlendProperty{', BlendProperty'.join(map(str, failed_blends))}. Check the logs for details.")

#     return combined_df if all(results[blend] is not None for blend in results) else None

# if __name__ == "__main__":
#     # Use the project root to construct the test file path
#     base_path = Path(__file__).parent.parent
#     test_filepath = base_path / "data" / "test.csv"
#     run_central_predictor(test_filepath)
import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.predictors.blend1_pred import Blend1Predictor
from src.predictors.blend2_pred import Blend2Predictor
from src.predictors.blend3_pred import Blend3Predictor
from src.predictors.blend4_pred import Blend4Predictor
from src.predictors.blend5_pred import Blend5Predictor
from src.predictors.blend6_pred import Blend6Predictor
from src.predictors.blend7_pred import Blend7Predictor
from src.predictors.blend8_pred import Blend8Predictor
from src.predictors.blend9_pred import Blend9Predictor
from src.predictors.blend10_pred import Blend10Predictor

def run_central_predictor(test_filepath, stream_progress=False):
    predictors = {
        1: Blend1Predictor,
        2: Blend2Predictor,
        3: Blend3Predictor,
        4: Blend4Predictor,
        5: Blend5Predictor,
        6: Blend6Predictor,
        7: Blend7Predictor,
        8: Blend8Predictor,
        9: Blend9Predictor,
        10: Blend10Predictor
    }
    results = {}
    total_blends = len(predictors)
    combined_df = None

    for blend, predictor_class in predictors.items():
        print(f"\nRunning prediction for BlendProperty{blend}...")
        try:
            # Instantiate predictor and run prediction
            predictor = predictor_class()
            predictions = predictor.predict(test_filepath)
            results[blend] = predictions

            # Yield progress for SSE if stream_progress is True
            if stream_progress:
                progress = (blend / total_blends) * 100
                yield {'progress': progress, 'blend': f'BlendProperty{blend}'}
        except Exception as e:
            print(f"Error in Blend{blend}Predictor: {str(e)}")
            results[blend] = None
            if stream_progress:
                yield {'error': f'Prediction failed for BlendProperty{blend}: {str(e)}'}

    # Combine results into a single DataFrame
    failed_blends = [blend for blend, result in results.items() if result is None]
    if not failed_blends:
        combined_df = pd.DataFrame(index=results[1].index)  # Use IDs from Blend1 as the index
        for blend, preds in results.items():
            combined_df[f'BlendProperty{blend}'] = preds
        combined_df.index.name = 'ID'
        combined_df.reset_index(inplace=True)

        # Save combined predictions
        output_filename = project_root / "predictions_all_blends.csv"
        combined_df.to_csv(output_filename, index=False)
        print(f"\n🎉 Combined predictions saved to '{output_filename}'!")
        print("\n--- Combined DataFrame with Predictions ---")
        print(combined_df.head())
    else:
        print(f"\n❌ Error: Predictions failed for BlendProperty{', BlendProperty'.join(map(str, failed_blends))}. Check the logs for details.")

    if stream_progress:
        if not failed_blends:
            yield {'progress': 100, 'blend': 'Complete', 'result': combined_df.to_dict(orient='records')}
        else:
            yield {'error': f"Predictions failed for BlendProperty{', BlendProperty'.join(map(str, failed_blends))}"}
    else:
        return combined_df if not failed_blends else None

if __name__ == "__main__":
    # Use the project root to construct the test file path
    base_path = Path(__file__).parent.parent
    test_filepath = base_path / "data" / "test.csv"
    
    # Initialize the generator
    prediction_generator = run_central_predictor(test_filepath, stream_progress=True)
    
    final_result = None
    
    # Iterate over the generator to get the final result
    for data in prediction_generator:
        # You can print the progress here if you want
        if 'progress' in data:
            print(f"Progress: {data['progress']}%, Blend: {data['blend']}")
        elif 'error' in data:
            print(f"Error: {data['error']}")
            final_result = None
            break
        
        # The 'result' key will only be in the final yielded dictionary
        if 'result' in data:
            final_result = pd.DataFrame.from_records(data['result'])
            print("\nSuccessfully received combined DataFrame:")
            print(final_result.head())
            
    if final_result is not None:
        # You can now save or use the final_result DataFrame
        print("\nFinal DataFrame successfully captured.")
import pandas as pd
from sklearn.metrics import accuracy_score
import os

# ==========================================================
# Configuration
# ==========================================================
REAL_DATA_CSV = "traffic_dataset.csv"
PREDICTION_CSV = "prediction_output.csv"

# ==========================================================
# Comparison Script
# ==========================================================
def compare_accuracy_on_predicted_data(real_csv_path, prediction_csv_path):
    """
    Loads real and predicted data, filters the real data to match the scope
    of the predictions, and calculates the accuracy percentage.
    """
    print("--- Starting Comparison ---")
    try:
        # Load the ground truth data and the model's predictions
        df_real = pd.read_csv(real_csv_path)
        df_pred = pd.read_csv(prediction_csv_path)
        print("Successfully loaded both CSV files.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required CSV file. {e}")
        return

    # Use an 'inner' merge. This is the key step.
    # It automatically filters both dataframes, keeping only the rows
    # where 'time' and 'hex_id' exist in BOTH files.
    comparison_df = pd.merge(
        df_pred,
        df_real,
        on=['time', 'hex_id'],
        how='inner',
        suffixes=('_pred', '_real')
    )

    if comparison_df.empty:
        print("\nError: No matching 'time' and 'hex_id' pairs were found between the two files.")
        print("Please ensure your prediction CSV contains timesteps and zones that exist in the real data file.")
        return

    # Extract the aligned labels for comparison
    real_labels = comparison_df['label_real']
    predicted_labels = comparison_df['label_pred']

    print(f"\nFound {len(comparison_df)} matching data points to compare.")

    # Calculate the accuracy score
    accuracy = accuracy_score(real_labels, predicted_labels)

    print("\n" + "="*40)
    print(f"Accuracy on Predicted Timesteps: {accuracy * 100:.2f}%")
    print("="*40)
    print("(This percentage shows how many of the predicted labels were correct, only for the times and zones present in your output file.)")


    # Show a sample of the compared data for manual verification
    print("\n--- Side-by-Side Comparison Sample ---")
    # Display the relevant columns for clarity
    print(comparison_df[['time', 'hex_id', 'label_pred', 'label_real']].head(15))


# ==========================================================
# EXECUTION
# ==========================================================
# Ensure both files exist before running the comparison
if os.path.exists(REAL_DATA_CSV) and os.path.exists(PREDICTION_CSV):
    compare_accuracy_on_predicted_data(REAL_DATA_CSV, PREDICTION_CSV)
else:
    print(f"Error: Make sure both '{REAL_DATA_CSV}' and '{PREDICTION_CSV}' are uploaded to your Colab session.")

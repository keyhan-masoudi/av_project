import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
import time
from collections import defaultdict
from torch_geometric.nn import GCNConv

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "./final_model300"
DATA_CSV = "../testfile/final300.csv" # The script needs the original data to get a test slice
X = 15  # Input sequence length (must match training)
Y = 12  # Output prediction length (must match training)

NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The name for the single, combined output file
SINGLE_OUTPUT_CSV = "predictions_output300.csv"

# ==========================================================
# RE-DEFINE HELPER FUNCTIONS & MODEL CLASSES
# (Must match the architecture from training)
# ==========================================================

class LSTMEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n_last = h_n[-1, :, :]
        x = self.act(self.fc(h_n_last))
        return self.dropout(x)


class GNNEncoder(nn.Module):
    def __init__(self, in_dim=6, h1=128, h2=64, out_dim=64, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, h1)
        self.conv2 = GCNConv(h1, h2)
        self.fc = nn.Linear(h2, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.fc(x))
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, lstm_dim=64, gnn_dim=64, Y=12, hidden_dim=256, classes=5, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(lstm_dim + gnn_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, Y * classes)
        self.Y = Y
        self.classes = classes

    def forward(self, lstm_emb, gnn_emb):
        x = torch.cat([lstm_emb, gnn_emb], dim=1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.view(x.size(0), self.Y, self.classes)


def normalize_df(df, stats):
    df_norm = df.copy()
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    for col in feature_cols:
        if col in stats:
            min_val, max_val = stats[col]
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val + 1e-9)
        else:
            print(f"Warning: Feature '{col}' not found in normalization stats. Skipping.")
    return df_norm


def build_snapshots(df, node_list, node_to_idx):
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    snapshots = {}
    unique_times = sorted(df["time"].unique())
    for t in unique_times:
        snapshot_features = np.zeros((len(node_list), len(feature_cols)), dtype=np.float32)
        sub_df = df[df["time"] == t]
        for _, row in sub_df.iterrows():
            q, r = map(int, row.hex_id.split("_"))
            idx = node_to_idx.get((q, r))
            if idx is not None:
                snapshot_features[idx] = row[feature_cols].values
        snapshots[t] = torch.tensor(snapshot_features, dtype=torch.float32)
    return snapshots


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def main_predict(input_df, model_components):
    """
    Runs the model for a single X-step input_df and
    RETURNS the Y-step prediction as a DataFrame.
    """
    
    # Unpack model components
    lstm, gnn, dec, stats, graph_info = model_components
    node_to_idx, edge_index, node_list = graph_info['node_to_idx'], graph_info['edge_index'].to(DEVICE), graph_info['node_list']

    # --- Prepare Data ---
    df_norm = normalize_df(input_df, stats)
    sequences = []
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]

    for node in node_list:
        hex_id = f"{node[0]}_{node[1]}"
        node_df_norm = df_norm[df_norm["hex_id"] == hex_id].sort_values("time")
        seq_data = node_df_norm[feature_cols].values

        if len(seq_data) < X:
            padding = np.zeros((X - len(seq_data), len(feature_cols)))
            seq_data = np.vstack([padding, seq_data])
        elif len(seq_data) > X:
             seq_data = seq_data[-X:] # Ensure it's exactly X steps

        sequences.append(torch.tensor(seq_data, dtype=torch.float32))

    sequences = torch.stack(sequences).to(DEVICE)

    last_timestep = input_df['time'].max()
    snapshots = build_snapshots(df_norm[df_norm['time'] == last_timestep], node_list, node_to_idx)
    snapshot_tensor = snapshots[last_timestep].to(DEVICE)

    # --- Make Prediction ---
    with torch.no_grad():
        lstm_emb = lstm(sequences)
        gnn_emb = gnn(snapshot_tensor, edge_index)
        logits = dec(lstm_emb, gnn_emb)
        predicted_labels = torch.argmax(logits, dim=2).cpu().numpy()

    # --- Format Output ---
    output_rows = []
    start_time_pred = last_timestep + 1
    for i in range(len(node_list)):
        hex_id = f"{node_list[i][0]}_{node_list[i][1]}"
        for j in range(Y):
            output_rows.append({
                'time': start_time_pred + j,
                'hex_id': hex_id,
                'label': predicted_labels[i, j] + 1  # Convert 0-indexed to 1-indexed
            })
            
    return pd.DataFrame(output_rows)

# ==========================================================
# EXECUTION SCRIPT
# ==========================================================
def run_all_predictions():
    print("Loading full dataset to extract test slices...")
    try:
        full_df = pd.read_csv(DATA_CSV)
        max_data_time = full_df['time'].max()
        
        print("Creating weather features (sunny, rainy) for the input data...")
        full_df['sunny'] = (full_df['weather'] == 1).astype(int)
        full_df['rainy'] = (full_df['weather'] == 2).astype(int)

        # **** LOOP CONFIGURATION ****
        LOOP_START_TIME = 2585
        LOOP_STEP = 10
        NUM_PREDICTIONS_TO_RUN = 100
        # ****************************
        
        # --- Load Model Components ONCE ---
        print(f"Loading model components from '{MODEL_PATH}/'...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Error: Model directory '{MODEL_PATH}' not found.")

        stats = joblib.load(os.path.join(MODEL_PATH, "stats.pkl"))
        graph_info = joblib.load(os.path.join(MODEL_PATH, "graph_info.pkl"))

        print(f"Initializing models with Y={Y} and NUM_CLASSES={NUM_CLASSES}")
        lstm = LSTMEncoder()
        gnn = GNNEncoder()
        dec = Decoder(lstm_dim=64, gnn_dim=64, Y=Y, classes=NUM_CLASSES)

        lstm.load_state_dict(torch.load(os.path.join(MODEL_PATH, "lstm_model.pth")))
        gnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, "gnn_model.pth")))
        dec.load_state_dict(torch.load(os.path.join(MODEL_PATH, "decoder_model.pth")))
        lstm.to(DEVICE).eval()
        gnn.to(DEVICE).eval()
        dec.to(DEVICE).eval()
        
        # Pack components to pass to the function
        model_components = (lstm, gnn, dec, stats, graph_info)
        # --- End of model loading ---


        # This list will hold all the small prediction DataFrames
        all_predictions_list = []

        print(f"Starting sequential prediction loop... Will run {NUM_PREDICTIONS_TO_RUN} predictions.")
        
        start_loop_time = time.perf_counter()
        
        for i in range(NUM_PREDICTIONS_TO_RUN):
            current_start_time = LOOP_START_TIME + (i * LOOP_STEP)
            current_end_time = current_start_time + X - 1 # X=15, so 14 steps later
            
            prediction_start_time_for_name = current_end_time + 1
            
            print(f"  Running Prediction {i+1}/{NUM_PREDICTIONS_TO_RUN} (Input: t={current_start_time} to t={current_end_time})...")

            if current_end_time > max_data_time:
                print(f"STOPPING LOOP: Not enough data for this slice.")
                print(f"Required data up to t={current_end_time}, but data only exists up to t={max_data_time}.")
                break
            
            # --- Extract Slice ---
            input_slice_df = full_df[
                (full_df['time'] >= current_start_time) &
                (full_df['time'] <= current_end_time)
            ].copy()

            # --- Safety Check ---
            unique_times_count = len(input_slice_df['time'].unique())
            if unique_times_count != X:
                print(f"  SKIPPING PREDICTION at t={current_start_time}.")
                print(f"  The input slice must contain exactly {X} timesteps.")
                print(f"  Found only {unique_times_count} unique timesteps.")
                continue 
            
            # --- Run Prediction ---
            prediction_df = main_predict(input_slice_df, model_components)
            
            # --- Add the result to our master list ---
            all_predictions_list.append(prediction_df)
            
        end_loop_time = time.perf_counter()
        print(f"\n--- Sequential prediction loop finished in {end_loop_time - start_loop_time:.2f} seconds. ---")

        # --- Combine and Save ---
        if not all_predictions_list:
            print("No predictions were generated. No output file created.")
        else:
            print(f"Combining {len(all_predictions_list)} prediction batches into one file...")
            
            # This is the key step: combine all DataFrames in the list
            final_combined_df = pd.concat(all_predictions_list)
            
            # Sort by time and hex_id to ensure the file is orderly
            final_combined_df.sort_values(by=['time', 'hex_id'], inplace=True)
            
            # Drop duplicates if time windows overlapped (safe to do)
            final_combined_df.drop_duplicates(subset=['time', 'hex_id'], keep='last', inplace=True)

            # Save the single, combined file
            final_combined_df.to_csv(SINGLE_OUTPUT_CSV, index=False)
            
            print(f"\n✅ Success! All predictions saved to: {SINGLE_OUTPUT_CSV}")
            print(f"   Total predictions generated: {len(final_combined_df)} rows")
            print("--- Final Combined Output Head ---")
            print(final_combined_df.head())
            print("--- Final Combined Output Tail ---")
            print(final_combined_df.tail())


    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# ==========================================================
# Run the script
# ==========================================================
if __name__ == "__main__":
    run_all_predictions()
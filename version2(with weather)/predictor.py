import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
import time
import random
from collections import defaultdict
from torch_geometric.nn import GCNConv # Corrected import

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "saved_model"       # <-- UPDATED
OUTPUT_CSV = "prediction_output.csv"
DATA_CSV = "new_traffic_dataset1.csv" # The script needs the original data to get a test slice
X = 15 # <-- UPDATED: Must match the training configuration
Y = 12 # <-- UPDATED: Must match the training configuration

NUM_CLASSES = 5

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

# ==========================================================
# RE-DEFINE HELPER FUNCTIONS & MODEL CLASSES
# (Must match the architecture from training)
# ==========================================================

# --- Model Definitions (UPDATED to match new training) ---
class LSTMEncoder(nn.Module):
    # Increased dimensions and added 2 layers + dropout
    def __init__(self, in_dim=6, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # h_n is (num_layers, batch, hidden_dim), we only want the last layer's hidden state
        h_n_last = h_n[-1, :, :]
        x = self.act(self.fc(h_n_last))
        return self.dropout(x)

class GNNEncoder(nn.Module):
    # Increased dimensions and added dropout
    def __init__(self, in_dim=6, h1=128, h2=64, out_dim=64, dropout=0.1):
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
    # Increased dimensions to match encoders and added dropout
    def __init__(self, lstm_dim=64, gnn_dim=64, Y=20, hidden_dim=256, classes=5, dropout=0.1):
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


# --- Helper Functions (Updated to match new training) ---
def normalize_df(df, stats):
    df_norm = df.copy()
    # <-- Correct: Using 6 features
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    for col in feature_cols:
        if col in stats:
            min_val, max_val = stats[col]
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val + 1e-9)
        else:
            print(f"Warning: Feature '{col}' not found in normalization stats. Skipping.")
    return df_norm


def build_snapshots(df, node_list, node_to_idx):
    # <-- Correct: Using 6 features
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
# PREDICTION SCRIPT
# ==========================================================
def main_predict(input_df):
    """
    Loads the trained model and makes a Y-step (20) prediction
    based on an X-step (15) input DataFrame.
    """
    print("--- Starting Prediction ---")
    print(f"Loading model components from '{MODEL_PATH}/'...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory '{MODEL_PATH}' not found.")
        return

    stats = joblib.load(os.path.join(MODEL_PATH, "stats.pkl"))
    graph_info = joblib.load(os.path.join(MODEL_PATH, "graph_info.pkl"))
    node_to_idx, edge_index, node_list = graph_info['node_to_idx'], graph_info['edge_index'].to(DEVICE), graph_info['node_list']

    # --- UPDATED: Initialize the new, larger models ---
    print(f"Initializing models with Y={Y} and NUM_CLASSES={NUM_CLASSES}")
    lstm = LSTMEncoder()
    gnn = GNNEncoder()
    # CRITICAL: Must match the new encoder dims (64, 64) and the global Y/classes
    dec = Decoder(lstm_dim=64, gnn_dim=64, Y=Y, classes=NUM_CLASSES)
    # ---

    lstm.load_state_dict(torch.load(os.path.join(MODEL_PATH, "lstm_model.pth")))
    gnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, "gnn_model.pth")))
    dec.load_state_dict(torch.load(os.path.join(MODEL_PATH, "decoder_model.pth")))
    lstm.to(DEVICE).eval();
    gnn.to(DEVICE).eval();
    dec.to(DEVICE).eval()

    print("Preparing input data for prediction...")
    df_norm = normalize_df(input_df, stats)
    sequences = []
    # <-- Correct: Using 6 features
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

    print("Making prediction...")
    # --- Time the inference step ---
    inf_start = time.perf_counter()
    with torch.no_grad():
        lstm_emb = lstm(sequences)
        gnn_emb = gnn(snapshot_tensor, edge_index)
        logits = dec(lstm_emb, gnn_emb)
        predicted_labels = torch.argmax(logits, dim=2).cpu().numpy()
    inf_end = time.perf_counter()
    print(f"-> Inference complete in {inf_end - inf_start:.4f} seconds.")
    # ---

    print(f"Formatting prediction and saving to '{OUTPUT_CSV}'...")
    output_rows = []
    start_time_pred = last_timestep + 1
    for i in range(len(node_list)):
        hex_id = f"{node_list[i][0]}_{node_list[i][1]}"
        # <-- UPDATED: This loop now correctly goes from j=0 to Y=19
        for j in range(Y):
            output_rows.append({
                'time': start_time_pred + j,
                'hex_id': hex_id,
                'label': predicted_labels[i, j] + 1  # Convert 0-indexed label back to 1-indexed
            })
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Prediction saved to {OUTPUT_CSV}.")
    # files.download(OUTPUT_CSV) # <-- Commented out for non-Colab use
    print("\n--- Prediction Output Head ---")
    print(output_df.head())


# ==========================================================
# EXECUTION
# ==========================================================
print("Loading full dataset to extract a test slice...")
try:
    full_df = pd.read_csv(DATA_CSV)

    # <-- NEW FEATURE ENGINEERING STEP
    print("Creating weather features (sunny, rainy) for the input data...")
    full_df['sunny'] = (full_df['weather'] == 1).astype(int)
    full_df['rainy'] = (full_df['weather'] == 2).astype(int)
    # ---

    # **** CHANGE THIS VALUE TO TEST DIFFERENT TIME SLICES ****
    # This time must have X-1 (i.e., 14) future timesteps available in the data
    start_pred_time = 1600
    # **********************************************************

    end_pred_time = start_pred_time + X - 1 # This will be 1500 + 14 = 1514
    input_slice_df = full_df[
        (full_df['time'] >= start_pred_time) &
        (full_df['time'] <= end_pred_time)
    ].copy()

    unique_times_count = len(input_slice_df['time'].unique())
    if unique_times_count != X:
        print(f"Error: The input slice must contain exactly {X} timesteps of data.")
        print(f"Found only {unique_times_count} unique timesteps between {start_pred_time} and {end_pred_time}.")
        print(f"Please check your `start_pred_time` value and ensure data exists up to time {end_pred_time}.")
    else:
        print(f"Successfully extracted {X} timesteps from t={start_pred_time} to t={end_pred_time}.")
        main_predict(input_slice_df)

except FileNotFoundError:
    print(f"Error: '{DATA_CSV}' not found. Please make sure it has been uploaded.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
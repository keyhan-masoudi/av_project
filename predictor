import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
from collections import defaultdict
# The torch_geometric import might be needed if not already imported in the session
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    !pip install torch_geometric -q
    from torch_geometric.nn import GCNConv
from google.colab import files # For downloading the output file

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "saved_model"
OUTPUT_CSV = "prediction_output.csv"
DATA_CSV = "traffic_dataset.csv" # The script needs the original data to get a test slice
X = 10 # Must match the training configuration
Y = 5  # Must match the training configuration

# !!! IMPORTANT: This MUST match the value used during training !!!
# Your training script used NUM_CLASSES = 5
NUM_CLASSES = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# RE-DEFINE HELPER FUNCTIONS & MODEL CLASSES
# (This makes the cell runnable on its own)
# ==========================================================

# --- Model Definitions ---
class LSTMEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.act(self.fc(h_n.squeeze(0)))

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=4, h1=64, h2=32, out_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, h1)
        self.conv2 = GCNConv(h1, h2)
        self.fc = nn.Linear(h2, out_dim)
        self.act = nn.ReLU()
    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        return self.act(self.fc(x))

class Decoder(nn.Module):
    # Make sure classes=NUM_CLASSES to match your training
    def __init__(self, lstm_dim=32, gnn_dim=32, Y=5, hidden_dim=128, classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(lstm_dim + gnn_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, Y * classes)
        self.act = nn.ReLU()
        self.Y = Y
        self.classes = classes
    def forward(self, lstm_emb, gnn_emb):
        x = torch.cat([lstm_emb, gnn_emb], dim=1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), self.Y, self.classes)

# --- Helper Functions ---
def normalize_df(df, stats):
    df_norm = df.copy()
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    for col in feature_cols:
        min_val, max_val = stats[col]
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val + 1e-9)
    return df_norm

def build_snapshots(df, node_list, node_to_idx):
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
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
    Loads the trained model and makes a 5-step prediction based on a 10-step input DataFrame.
    """
    print("--- Starting Prediction ---")
    print(f"Loading model components from '{MODEL_PATH}/'...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory '{MODEL_PATH}' not found. Please upload it to your Colab session.")
        return

    stats = joblib.load(os.path.join(MODEL_PATH, "stats.pkl"))
    graph_info = joblib.load(os.path.join(MODEL_PATH, "graph_info.pkl"))
    node_to_idx, edge_index, node_list = graph_info['node_to_idx'], graph_info['edge_index'].to(DEVICE), graph_info['node_list']

    lstm, gnn, dec = LSTMEncoder(), GNNEncoder(), Decoder()
    lstm.load_state_dict(torch.load(os.path.join(MODEL_PATH, "lstm_model.pth")))
    gnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, "gnn_model.pth")))
    dec.load_state_dict(torch.load(os.path.join(MODEL_PATH, "decoder_model.pth")))
    lstm.to(DEVICE).eval(); gnn.to(DEVICE).eval(); dec.to(DEVICE).eval()

    print("Preparing input data for prediction...")
    df_norm = normalize_df(input_df, stats)
    sequences = []
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    for node in node_list:
        hex_id = f"{node[0]}_{node[1]}"
        node_df_norm = df_norm[df_norm["hex_id"] == hex_id].sort_values("time")
        seq_data = node_df_norm[feature_cols].values
        if len(seq_data) < X:
            padding = np.zeros((X - len(seq_data), len(feature_cols)))
            seq_data = np.vstack([padding, seq_data])
        sequences.append(torch.tensor(seq_data, dtype=torch.float32))
    sequences = torch.stack(sequences).to(DEVICE)

    last_timestep = input_df['time'].max()
    snapshots = build_snapshots(df_norm[df_norm['time'] == last_timestep], node_list, node_to_idx)
    snapshot_tensor = snapshots[last_timestep].to(DEVICE)

    print("Making prediction...")
    with torch.no_grad():
        lstm_emb = lstm(sequences)
        gnn_emb = gnn(snapshot_tensor, edge_index)
        logits = dec(lstm_emb, gnn_emb)
        predicted_labels = torch.argmax(logits, dim=2).cpu().numpy()

    print(f"Formatting prediction and saving to '{OUTPUT_CSV}'...")
    output_rows = []
    start_time_pred = last_timestep + 1
    for i in range(len(node_list)):
        hex_id = f"{node_list[i][0]}_{node_list[i][1]}"
        for j in range(Y):
            output_rows.append({
                'time': start_time_pred + j,
                'hex_id': hex_id,
                'label': predicted_labels[i, j] + 1 # Convert 0-indexed label back to 1-indexed
            })
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Prediction saved to {OUTPUT_CSV} and will be downloaded.")
    files.download(OUTPUT_CSV)
    print("\n--- Prediction Output Head ---")
    print(output_df.head())

# ==========================================================
# EXECUTION
# ==========================================================
print("Loading full dataset to extract a test slice...")
try:
    full_df = pd.read_csv(DATA_CSV)

    # **** CHANGE THIS VALUE TO TEST DIFFERENT TIME SLICES ****
    start_pred_time = 1001
    # **********************************************************

    end_pred_time = start_pred_time + X - 1
    input_slice_df = full_df[(full_df['time'] >= start_pred_time) & (full_df['time'] <= end_pred_time)].copy()

    if len(input_slice_df['time'].unique()) != X:
        print(f"Error: The input slice must contain exactly {X} timesteps of data. Please check your `start_pred_time` value.")
    else:
        main_predict(input_slice_df)
except FileNotFoundError:
    print(f"Error: '{DATA_CSV}' not found. Please make sure it has been uploaded to this Colab session.")

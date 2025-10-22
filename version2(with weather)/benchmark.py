import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
import time
from google.colab import files  # Assuming Colab for files.download

# ==========================================================
# CONSTANTS
# ==========================================================

# --- Model & Data Paths ---
MODEL_PATH = "saved_model"
DATA_CSV = "new_traffic_dataset.csv"
OUTPUT_CSV = "prediction_output.csv"

# --- Model Parameters ---
X = 10  # Number of input timesteps
Y = 5  # Number of output timesteps

# --- Environment ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ==========================================================
# MODEL & HELPER FUNCTION PLACEHOLDERS
# ==========================================================
# PLEASE PASTE your model class definitions and helper functions here.

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        # --- (Your LSTMEncoder class definition) ---
        # Example structure:
        self.lstm = nn.LSTM(input_size=6, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        # --- (Your forward pass logic) ---
        # x shape: [batch_size, seq_len, features]
        _, (h_n, _) = self.lstm(x)
        # h_n shape: [num_layers, batch_size, hidden_size]
        # Get last hidden state
        last_hidden_state = h_n[-1]  # [batch_size, hidden_size]
        out = self.fc(last_hidden_state)  # [batch_size, 32]
        return out


class GNNEncoder(nn.Module):
    def __init__(self):
        super(GNNEncoder, self).__init__()
        # --- (Your GNNEncoder class definition) ---
        # Example using torch_geometric:
        # from torch_geometric.nn import GCNConv
        # self.conv1 = GCNConv(in_channels=..., out_channels=...)
        # self.conv2 = GCNConv(in_channels=..., out_channels=32)
        pass  # Replace with your class

    def forward(self, x, edge_index):
        # --- (Your forward pass logic) ---
        # x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index)
        # return x

        # Placeholder forward pass if GNNEncoder is not defined
        if not hasattr(self, 'conv1'):
            print("Warning: GNNEncoder not fully defined. Returning dummy tensor.")
            return torch.randn(x.size(0), 32).to(DEVICE)

        return torch.randn(x.size(0), 32).to(DEVICE)  # Replace with your logic


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # --- (Your Decoder class definition) ---
        # Example structure:
        # Input will be concat of LSTM (32) and GNN (32) = 64
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, Y * 3)  # Y steps, 3 classes
        self.num_classes = 3  # Example: 3 classes (labels 1, 2, 3)

    def forward(self, lstm_emb, gnn_emb):
        # --- (Your forward pass logic) ---
        # Concat spatial and temporal embeddings
        x = torch.cat([lstm_emb, gnn_emb], dim=1)  # [batch_size, 64]
        x = self.fc1(x).relu()
        x = self.fc2(x)  # [batch_size, Y * num_classes]
        # Reshape to [batch_size, Y_steps, num_classes]
        x = x.view(-1, Y, self.num_classes)
        return x


def normalize_df(df, stats):
    """Normalizes the DataFrame using pre-computed stats."""
    # --- (Your normalize_df function definition) ---
    print("Warning: normalize_df() not defined. Returning original DataFrame.")
    # Example:
    # df_norm = df.copy()
    # for col, stat in stats.items():
    #     if col in df_norm.columns:
    #         df_norm[col] = (df_norm[col] - stat['mean']) / stat['std']
    # return df_norm
    return df


def build_snapshots(df_slice, node_list, node_to_idx):
    """Builds a graph snapshot for a single timestep."""
    # --- (Your build_snapshots function definition) ---
    print("Warning: build_snapshots() not defined. Returning dummy snapshot.")
    # Example structure:
    # import torch_geometric.data as TGeomData
    # snapshots = {}
    # features = []
    # # ... logic to build feature matrix 'x' for all nodes in node_list ...
    # # x = torch.tensor(feature_matrix, dtype=torch.float32)
    # x = torch.randn(len(node_list), 4) # Dummy features (num_vehicles, avg_speed, avg_sin, avg_cos)
    # snapshot = TGeomData.Data(x=x)
    # snapshots[df_slice['time'].min()] = snapshot
    # return snapshots

    # Dummy snapshot for placeholder
    import torch_geometric.data as TGeomData
    snapshots = {}
    dummy_x = torch.randn(len(node_list), 4)  # 4 GNN features
    snapshot = TGeomData.Data(x=dummy_x)
    snapshots[df_slice['time'].min()] = snapshot
    return snapshots


# ==========================================================
# ORIGINAL PREDICTION FUNCTION
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
    node_to_idx, edge_index, node_list = graph_info['node_to_idx'], graph_info['edge_index'].to(DEVICE), graph_info[
        'node_list']

    lstm, gnn, dec = LSTMEncoder(), GNNEncoder(), Decoder()
    lstm.load_state_dict(torch.load(os.path.join(MODEL_PATH, "lstm_model.pth")))
    gnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, "gnn_model.pth")))
    dec.load_state_dict(torch.load(os.path.join(MODEL_PATH, "decoder_model.pth")))
    lstm.to(DEVICE).eval();
    gnn.to(DEVICE).eval();
    dec.to(DEVICE).eval()

    print("Preparing input data for prediction...")
    df_norm = normalize_df(input_df, stats)
    sequences = []
    # <-- MODIFIED: Added "sunny" and "rainy"
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
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
                'label': predicted_labels[i, j] + 1  # Convert 0-indexed label back to 1-indexed
            })
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Prediction saved to {OUTPUT_CSV} and will be downloaded.")
    # files.download(OUTPUT_CSV)
    print("\n--- Prediction Output Head ---")
    print(output_df.head())


# ==========================================================
# BENCHMARKING FUNCTION
# ==========================================================

def benchmark_prediction(input_df, n_warmup=10, n_runs=100):
    """
    Loads the model and benchmarks the core prediction time.
    """
    print("--- Starting Benchmark ---")
    print(f"Loading model components from '{MODEL_PATH}/'...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory '{MODEL_PATH}' not found.")
        return

    # --- 1. Load Models and Data (Do this ONCE) ---
    stats = joblib.load(os.path.join(MODEL_PATH, "stats.pkl"))
    graph_info = joblib.load(os.path.join(MODEL_PATH, "graph_info.pkl"))
    node_to_idx, edge_index, node_list = graph_info['node_to_idx'], graph_info['edge_index'].to(DEVICE), graph_info[
        'node_list']

    lstm, gnn, dec = LSTMEncoder(), GNNEncoder(), Decoder()
    lstm.load_state_dict(torch.load(os.path.join(MODEL_PATH, "lstm_model.pth")))
    gnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, "gnn_model.pth")))
    dec.load_state_dict(torch.load(os.path.join(MODEL_PATH, "decoder_model.pth")))
    lstm.to(DEVICE).eval();
    gnn.to(DEVICE).eval();
    dec.to(DEVICE).eval()

    print("Preparing input data...")
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
        sequences.append(torch.tensor(seq_data, dtype=torch.float32))
    sequences = torch.stack(sequences).to(DEVICE)

    last_timestep = input_df['time'].max()
    snapshots = build_snapshots(df_norm[df_norm['time'] == last_timestep], node_list, node_to_idx)
    snapshot_tensor = snapshots[last_timestep].to(DEVICE)

    # --- 2. Warm-up Loop (No timing) ---
    print(f"Running {n_warmup} warm-up iterations...")
    for _ in range(n_warmup):
        with torch.no_grad():
            lstm_emb = lstm(sequences)
            gnn_emb = gnn(snapshot_tensor, edge_index)
            logits = dec(lstm_emb, gnn_emb)
            _ = torch.argmax(logits, dim=2).cpu().numpy()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    # --- 3. Measurement Loop ---
    print(f"Running {n_runs} benchmark iterations...")
    timings_ms = []
    for _ in range(n_runs):

        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # Ensure previous run is finished

        start_time = time.perf_counter()

        with torch.no_grad():
            lstm_emb = lstm(sequences)
            gnn_emb = gnn(snapshot_tensor, edge_index)
            logits = dec(lstm_emb, gnn_emb)
            # Note: .cpu() call also takes time (data transfer)
            predicted_labels = torch.argmax(logits, dim=2).cpu().numpy()

        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # Force wait for this run

        end_time = time.perf_counter()
        timings_ms.append((end_time - start_time) * 1000)

    # --- 4. Report Results ---
    print("\n--- Benchmark Results ---")
    print(f"Total iterations: {n_runs}")
    print(f"Mean time:   {np.mean(timings_ms):.4f} ms")
    print(f"Median time: {np.median(timings_ms):.4f} ms")
    print(f"Min time:    {np.min(timings_ms):.4f} ms")
    print(f"Max time:    {np.max(timings_ms):.4f} ms")
    print(f"Std Dev:     {np.std(timings_ms):.4f} ms")
    print("-------------------------\n")


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
    start_pred_time = 1800
    # **********************************************************

    end_pred_time = start_pred_time + X - 1
    input_slice_df = full_df[(full_df['time'] >= start_pred_time) & (full_df['time'] <= end_pred_time)].copy()

    if len(input_slice_df['time'].unique()) != X:
        print(
            f"Error: The input slice must contain exactly {X} timesteps of data. Please check your `start_pred_time` value.")
    else:
        # ---
        # Run the benchmark:
        benchmark_prediction(input_slice_df, n_warmup=10, n_runs=100)

        # ---
        # Or, to run just one prediction and get the file:
        # print("\nRunning a single prediction to generate output file...")
        # main_predict(input_slice_df)
        # ---

except FileNotFoundError:
    print(f"Error: '{DATA_CSV}' not found. Please make sure it has been uploaded to this Colab session.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure all placeholder functions and classes are filled in.")
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
import time
import random
from collections import defaultdict
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "new_saved_model"  # <-- Make sure this is your trained model
# MODEL_PATH = "model_finetuned"
# MODEL_PATH = "finetuned_models/run_2"
DATA_CSV = "new_traffic_dataset.csv"
X = 20  # <-- Must match the training configuration
Y = 20  # <-- Must match the training configuration
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Benchmark Configuration ---
NUM_RUNS = 100
TIME_RANGE_START = 2600
TIME_RANGE_END = 3600  # <-- Updated as requested


# ---


# ==========================================================
# RE-DEFINE HELPER FUNCTIONS & MODEL CLASSES
# (Must match the architecture from training)
# ==========================================================

# --- Model Definitions (The new, powerful models) ---
class LSTMEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.2):  # Default dropout=0.2
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
    def __init__(self, in_dim=6, h1=128, h2=64, out_dim=64, dropout=0.2):  # Default dropout=0.2
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
    def __init__(self, lstm_dim=64, gnn_dim=64, Y=20, hidden_dim=256, classes=5, dropout=0.2):  # Default dropout=0.2
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
# EFFICIENT PREDICTION & BENCHMARKING FUNCTIONS
# ==========================================================

def load_components(model_path):
    """
    Loads all necessary components (models, stats, graph) from disk once.
    """
    print(f"Loading model components from '{model_path}/'...")
    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' not found.")
        return None

    stats = joblib.load(os.path.join(model_path, "stats.pkl"))
    graph_info = joblib.load(os.path.join(model_path, "graph_info.pkl"))
    graph_info['edge_index'] = graph_info['edge_index'].to(DEVICE)

    # Initialize the correct, larger models
    print(f"Initializing models with Y={Y} and NUM_CLASSES={NUM_CLASSES}")
    lstm = LSTMEncoder()
    gnn = GNNEncoder()
    dec = Decoder(lstm_dim=64, gnn_dim=64, Y=Y, classes=NUM_CLASSES)

    lstm.load_state_dict(torch.load(os.path.join(model_path, "lstm_model.pth")))
    gnn.load_state_dict(torch.load(os.path.join(model_path, "gnn_model.pth")))
    dec.load_state_dict(torch.load(os.path.join(model_path, "decoder_model.pth")))

    lstm.to(DEVICE).eval()
    gnn.to(DEVICE).eval()
    dec.to(DEVICE).eval()

    models = {'lstm': lstm, 'gnn': gnn, 'dec': dec}
    print("All components loaded and moved to device.")
    return models, stats, graph_info


def run_inference(input_df, models, stats, graph_info):
    """
    Runs a single prediction on the input_df using pre-loaded models.
    Returns the prediction as a DataFrame.
    """
    node_to_idx = graph_info['node_to_idx']
    edge_index = graph_info['edge_index']
    node_list = graph_info['node_list']

    # --- 1. Data Preparation ---
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
            seq_data = seq_data[-X:]  # Ensure it's exactly X steps

        sequences.append(torch.tensor(seq_data, dtype=torch.float32))

    sequences = torch.stack(sequences).to(DEVICE)
    last_timestep = input_df['time'].max()
    snapshots = build_snapshots(df_norm[df_norm['time'] == last_timestep], node_list, node_to_idx)
    snapshot_tensor = snapshots[last_timestep].to(DEVICE)

    # --- 2. Run Model ---
    with torch.no_grad():
        lstm_emb = models['lstm'](sequences)
        gnn_emb = models['gnn'](snapshot_tensor, edge_index)
        logits = models['dec'](lstm_emb, gnn_emb)
        predicted_labels = torch.argmax(logits, dim=2).cpu().numpy()

    # --- 3. Format Output ---
    output_rows = []
    start_time_pred = last_timestep + 1
    for i in range(len(node_list)):
        hex_id = f"{node_list[i][0]}_{node_list[i][1]}"
        for j in range(Y):  # Loop for Y=20 steps
            output_rows.append({
                'time': start_time_pred + j,
                'hex_id': hex_id,
                'label_pred': predicted_labels[i, j] + 1  # Convert back to 1-indexed
            })

    return pd.DataFrame(output_rows)


# ==========================================================
# EXECUTION
# ==========================================================
def run_accuracy_benchmark():
    print(f"--- Starting Accuracy Benchmark ---")
    print(f"Running {NUM_RUNS} tests on random start times between {TIME_RANGE_START} and {TIME_RANGE_END}.")

    try:
        # --- 1. Load Full Dataset ---
        print(f"Loading full dataset from '{DATA_CSV}'...")
        full_df = pd.read_csv(DATA_CSV)
        # Add weather features (needed for input slices)
        full_df['sunny'] = (full_df['weather'] == 1).astype(int)
        full_df['rainy'] = (full_df['weather'] == 2).astype(int)

        # --- 2. Load Models Once ---
        components = load_components(MODEL_PATH)
        if components is None:
            return
        models, stats, graph_info = components

        # --- 3. Find Valid Start Times ---
        print(f"Searching for {NUM_RUNS} valid start times...")
        all_times_set = set(full_df['time'].unique())
        valid_start_times = []

        # A start time 't' is valid if all data exists from t to t+X+Y-1
        required_len = X + Y
        search_end = TIME_RANGE_END - required_len + 2

        for t in range(TIME_RANGE_START, search_end):
            # Check if all 35 (15+20) consecutive time steps exist in the dataset
            if all((t + i) in all_times_set for i in range(required_len)):
                valid_start_times.append(t)

        # --- FIX for UnboundLocalError ---
        num_to_run = NUM_RUNS

        if len(valid_start_times) < NUM_RUNS:
            print(f"Warning: Found only {len(valid_start_times)} valid start times, but need {NUM_RUNS}.")
            print(f"A 'valid start time' (t) must have a complete data slice from t to t+{X + Y - 1}.")

            num_to_run = len(valid_start_times)  # Use a new local variable
            print(f"Sampling {num_to_run} times instead.")

            if num_to_run == 0:
                print("Error: No valid start times found. Check your time range and data.")
                return

        start_times_sample = random.sample(valid_start_times, num_to_run)
        print(f"Found {len(valid_start_times)} valid times. Sampling {num_to_run}.")

        all_accuracy_scores = []

        # --- 4. Run Benchmark Loop ---
        for i, start_time in enumerate(start_times_sample):
            print(f"  Run {i + 1}/{num_to_run}: Testing from start_time {start_time}...")

            # --- Get Input Slice (e.g., 1500 to 1514) ---
            t_end_input = start_time + X - 1
            input_slice_df = full_df[
                (full_df['time'] >= start_time) &
                (full_df['time'] <= t_end_input)
                ].copy()

            # --- Run Prediction (gets t+1 to t+20, e.g., 1515 to 1534) ---
            pred_df = run_inference(input_slice_df, models, stats, graph_info)

            # --- Get Ground Truth Slice (e.g., 1515 to 1534) ---
            t_start_output = start_time + X
            t_end_output = start_time + X + Y - 1
            real_df = full_df[
                (full_df['time'] >= t_start_output) &
                (full_df['time'] <= t_end_output)
                ]

            # --- Compare Prediction and Ground Truth ---
            # We don't need suffixes, as 'label_pred' and 'label' are unique
            comparison_df = pd.merge(
                pred_df,
                real_df,
                on=['time', 'hex_id'],
                how='inner'
            )

            if comparison_df.empty:
                print(f"    -> Warning: No matching real data found for prediction. Skipping run.")
                continue

            # --- FIX for KeyError: 'label_real' ---
            # The real label column is 'label', not 'label_real'
            real_labels = comparison_df['label']
            predicted_labels = comparison_df['label_pred']

            # Convert both to 0-index for safety
            if real_labels.min() == 1:
                real_labels = real_labels - 1
            if predicted_labels.min() == 1:
                predicted_labels = predicted_labels - 1

            accuracy = accuracy_score(real_labels, predicted_labels)
            all_accuracy_scores.append(accuracy)
            print(f"    -> Accuracy: {accuracy * 100:.2f}%")

        # --- 5. Report Statistics ---
        if not all_accuracy_scores:
            print("No accuracy scores were calculated. Benchmark failed.")
            return

        acc_array = np.array(all_accuracy_scores)

        print("\n" + "=" * 40)
        print("--- Accuracy Benchmark Results ---")
        print(f" (Based on {len(acc_array)} successful runs) ")
        print("=" * 40)
        print(f"Mean Accuracy:   {np.mean(acc_array) * 100:.2f}%")
        print(f"Median Accuracy: {np.median(acc_array) * 100:.2f}%")
        print(f"Max Accuracy:    {np.max(acc_array) * 100:.2f}%")
        print(f"Min Accuracy:    {np.min(acc_array) * 100:.2f}%")
        print("=" * 40)

    except FileNotFoundError:
        print(f"Error: '{DATA_CSV}' not found. Please make sure it has been uploaded.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


# --- Run the Benchmark ---
run_accuracy_benchmark()
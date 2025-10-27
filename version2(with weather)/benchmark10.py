import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
import time
# No longer need 'random' for this version
from collections import defaultdict
from sklearn.metrics import accuracy_score

# The torch_geometric import might be needed if not already imported in the session
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("torch_geometric not found. Installing...")
    try:
        import subprocess

        subprocess.check_call(['pip', 'install', 'torch_geometric', '-q'])
    except Exception as e:
        print(f"Failed to install torch_geometric: {e}")
        print("Please install it manually.")
    from torch_geometric.nn import GCNConv

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "new_saved_model"  # <-- Make sure this is your trained model
DATA_CSV = "new_traffic_dataset.csv"
X = 20  # <-- Must match the training configuration
Y = 20  # <-- Must match the training configuration (Model predicts 20 steps)
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Benchmark Configuration (UPDATED) ---
# NUM_RUNS is removed, number of runs is determined by the range
SEQ_TIME_START = 2600
SEQ_TIME_END = 3560  # Inclusive end time
SEQ_TIME_STEP = 10  # Increment step
ACCURACY_HORIZON_SHORT = 15  # Compare first 10 steps
ACCURACY_HORIZON_FULL = 20  # Compare all 20 steps


# ---


# ==========================================================
# RE-DEFINE HELPER FUNCTIONS & MODEL CLASSES
# (Must match the architecture from training)
# ==========================================================

# --- Model Definitions (The new, powerful models, Dropout=0.2) ---
class LSTMEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.2):
        super().__init__();
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0);
        self.fc = nn.Linear(hidden_dim, out_dim);
        self.act = nn.ReLU();
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): _, (h_n, _) = self.lstm(x); h_n_last = h_n[-1, :, :]; x = self.act(
        self.fc(h_n_last)); return self.dropout(x)


class GNNEncoder(nn.Module):
    def __init__(self, in_dim=6, h1=128, h2=64, out_dim=64, dropout=0.2):
        super().__init__();
        self.conv1 = GCNConv(in_dim, h1);
        self.conv2 = GCNConv(h1, h2);
        self.fc = nn.Linear(h2, out_dim);
        self.act = nn.ReLU();
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index): x = self.act(self.conv1(x, edge_index)); x = self.dropout(x); x = self.act(
        self.conv2(x, edge_index)); x = self.dropout(x); x = self.act(self.fc(x)); return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, lstm_dim=64, gnn_dim=64, Y=20, hidden_dim=256, classes=5, dropout=0.2):
        super().__init__();
        self.fc1 = nn.Linear(lstm_dim + gnn_dim, hidden_dim);
        self.act = nn.ReLU();
        self.dropout = nn.Dropout(dropout);
        self.fc2 = nn.Linear(hidden_dim, Y * classes);
        self.Y = Y;
        self.classes = classes

    def forward(self, lstm_emb, gnn_emb): x = torch.cat([lstm_emb, gnn_emb], dim=1); x = self.act(
        self.fc1(x)); x = self.dropout(x); x = self.fc2(x); return x.view(x.size(0), self.Y, self.classes)


# --- Helper Functions (Robust versions) ---
def normalize_df(df, stats):
    df_norm = df.copy();
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    for col in feature_cols:
        if col in stats:
            min_val, max_val = stats[col]; denominator = max_val - min_val; df_norm[col] = 0.0 if abs(
                denominator) < 1e-9 else (df_norm[col] - min_val) / denominator
        else:
            print(f"Warn: Stats missing for '{col}'. Skip norm.")
    return df_norm


def build_snapshots(df, node_list, node_to_idx):
    snapshots = {};
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    if df.empty: return snapshots
    unique_times = sorted(df["time"].unique());
    if not unique_times: return snapshots
    num_nodes = len(node_list);
    num_features = len(feature_cols)
    node_hexid_to_idx_map = {f"{q}_{r}": node_to_idx.get((q, r)) for q, r in node_list if
                             node_to_idx.get((q, r)) is not None}
    for t in unique_times:
        snapshot_features = np.zeros((num_nodes, num_features), dtype=np.float32);
        sub_df = df.loc[df["time"] == t]
        valid_rows = sub_df[sub_df['hex_id'].isin(node_hexid_to_idx_map.keys())]
        if not valid_rows.empty:
            indices = valid_rows['hex_id'].map(node_hexid_to_idx_map).values;
            values = valid_rows[feature_cols].values
            valid_mask = ~np.isnan(indices) & (indices < num_nodes);
            valid_indices = indices[valid_mask].astype(int)
            snapshot_features[valid_indices] = values[valid_mask]
        snapshots[t] = torch.tensor(snapshot_features, dtype=torch.float32)
    return snapshots


def load_components(model_path):
    print(f"Loading model components from '{model_path}/'...")
    if not os.path.exists(model_path): print(f"Error: Model directory '{model_path}' not found."); return None
    try:
        stats = joblib.load(os.path.join(model_path, "stats.pkl"))
        graph_info = joblib.load(os.path.join(model_path, "graph_info.pkl"))
        edge_index_loaded = graph_info['edge_index']
        lstm = LSTMEncoder();
        gnn = GNNEncoder()
        dec = Decoder(lstm_dim=64, gnn_dim=64, Y=Y, classes=NUM_CLASSES)  # Still Y=20 here
        map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        lstm.load_state_dict(torch.load(os.path.join(model_path, "lstm_model.pth"), map_location=map_location))
        gnn.load_state_dict(torch.load(os.path.join(model_path, "gnn_model.pth"), map_location=map_location))
        dec.load_state_dict(torch.load(os.path.join(model_path, "decoder_model.pth"), map_location=map_location))
        lstm.to(DEVICE).eval();
        gnn.to(DEVICE).eval();
        dec.to(DEVICE).eval()
        graph_info['edge_index'] = edge_index_loaded.to(DEVICE)
        models = {'lstm': lstm, 'gnn': gnn, 'dec': dec}
        print("All components loaded and moved to device.")
        return models, stats, graph_info
    except Exception as e:
        print(f"Error loading components from {model_path}: {e}"); return None


def run_inference(input_df, models, stats, graph_info):
    node_to_idx = graph_info['node_to_idx'];
    edge_index = graph_info['edge_index'];
    node_list = graph_info['node_list']
    df_norm = normalize_df(input_df, stats);
    sequences = []
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    for node in node_list:
        hex_id = f"{node[0]}_{node[1]}";
        node_df_norm = df_norm[df_norm["hex_id"] == hex_id].sort_values("time")
        seq_data = node_df_norm[feature_cols].values
        if len(seq_data) < X:
            padding = np.zeros((X - len(seq_data), len(feature_cols))); seq_data = np.vstack([padding, seq_data])
        elif len(seq_data) > X:
            seq_data = seq_data[-X:]
        sequences.append(torch.tensor(seq_data, dtype=torch.float32))
    sequences = torch.stack(sequences).to(DEVICE);
    last_timestep = input_df['time'].max()
    snapshots = build_snapshots(df_norm[df_norm['time'] == last_timestep], node_list, node_to_idx)
    if last_timestep not in snapshots: print(f"Warn: Snap {last_timestep} miss."); return pd.DataFrame()
    snapshot_tensor = snapshots[last_timestep].to(DEVICE)
    with torch.no_grad():
        lstm_emb = models['lstm'](sequences);
        gnn_emb = models['gnn'](snapshot_tensor, edge_index)
        logits = models['dec'](lstm_emb, gnn_emb);
        predicted_labels = torch.argmax(logits, dim=2).cpu().numpy()
    output_rows = [];
    start_time_pred = last_timestep + 1
    for i in range(len(node_list)):
        hex_id = f"{node_list[i][0]}_{node_list[i][1]}"
        for j in range(Y): output_rows.append(
            {'time': start_time_pred + j, 'hex_id': hex_id, 'label_pred': predicted_labels[i, j] + 1})
    return pd.DataFrame(output_rows)


# ==========================================================
# EXECUTION (MODIFIED FOR SEQUENTIAL TIMES & DUAL ACCURACY)
# ==========================================================
def run_accuracy_benchmark_sequential():
    print(f"--- Starting Sequential Accuracy Benchmark ---")
    print(f"Testing model '{MODEL_PATH}'")
    print(f"Using start times from {SEQ_TIME_START} to {SEQ_TIME_END} (step {SEQ_TIME_STEP}).")
    print(f"Calculating accuracy for first {ACCURACY_HORIZON_SHORT} and all {ACCURACY_HORIZON_FULL} steps.")

    try:
        # --- 1. Load Full Dataset ---
        print(f"Loading full dataset from '{DATA_CSV}'...")
        full_df = pd.read_csv(DATA_CSV)
        if 'weather' in full_df.columns:
            full_df['sunny'] = (full_df['weather'] == 1).astype(int)
            full_df['rainy'] = (full_df['weather'] == 2).astype(int)
        else:
            print("Warning: 'weather' column not found.")
        if 'label' not in full_df.columns: raise ValueError("Label column missing")
        if pd.api.types.is_numeric_dtype(full_df['label']) and full_df['label'].min() == 1:
            print("Adjusting labels in full_df to 0-index.")
            full_df['label'] = full_df['label'] - 1

        # --- 2. Load Models Once ---
        components = load_components(MODEL_PATH)
        if components is None: return
        models, stats, graph_info = components

        # --- 3. Determine Sequential Start Times & Check Validity ---
        all_times_set = set(full_df['time'].unique())
        sequential_start_times = list(range(SEQ_TIME_START, SEQ_TIME_END + 1, SEQ_TIME_STEP))

        valid_start_times = []
        required_len = X + Y  # Need full Y steps for comparison data

        print(f"Checking validity for {len(sequential_start_times)} potential start times...")
        for t in sequential_start_times:
            # Check if all data exists from t to t+X+Y-1
            if all((t + i) in all_times_set for i in range(required_len)):
                valid_start_times.append(t)
            else:
                print(
                    f"  -> Skipping start time {t}: Insufficient data for full comparison (requires up to t+{required_len - 1}).")

        num_to_run = len(valid_start_times)
        if num_to_run == 0:
            print("Error: No valid start times found in the specified sequence range.")
            return

        print(f"Proceeding with {num_to_run} valid start times.")

        all_accuracy_scores_short = []  # For first 10 steps
        all_accuracy_scores_full = []  # For all 20 steps

        # --- 4. Run Benchmark Loop ---
        for i, start_time in enumerate(valid_start_times):
            print(f"  Run {i + 1}/{num_to_run}: Testing from start_time {start_time}...")

            # --- Get Input Slice (X steps) ---
            t_end_input = start_time + X - 1
            input_slice_df = full_df[(full_df['time'] >= start_time) & (full_df['time'] <= t_end_input)].copy()

            # --- Run Prediction (gets Y=20 steps) ---
            pred_df_full = run_inference(input_slice_df, models, stats, graph_info)
            if pred_df_full.empty: print(f"    -> Warn: Inference failed for {start_time}. Skip."); continue

            # --- Get Ground Truth Slice (Y=20 steps) ---
            t_start_output = start_time + X
            t_end_output_full = start_time + X + Y - 1
            real_df_full = full_df[
                (full_df['time'] >= t_start_output) &
                (full_df['time'] <= t_end_output_full)
                ]

            # --- Compare FULL Prediction (20 steps) ---
            comparison_df_full = pd.merge(pred_df_full, real_df_full, on=['time', 'hex_id'], how='inner')
            if not comparison_df_full.empty:
                real_labels_full = comparison_df_full['label']  # Should be 0-indexed now
                pred_labels_full = comparison_df_full['label_pred']
                if not pred_labels_full.empty and pred_labels_full.min() == 1: pred_labels_full -= 1

                if len(real_labels_full) == len(pred_labels_full) and len(real_labels_full) > 0:
                    accuracy_full = accuracy_score(real_labels_full, pred_labels_full)
                    all_accuracy_scores_full.append(accuracy_full)
                    print(f"    -> Accuracy (all {Y} steps): {accuracy_full * 100:.2f}%")
                else:
                    print(f"    -> Warn: Label issue for full comparison at {start_time}. Skip acc.")
            else:
                print(f"    -> Warn: No matching full real data for {start_time}. Skip acc.")

            # --- Compare SHORT Prediction (First 10 steps) ---
            t_end_output_short = start_time + X + ACCURACY_HORIZON_SHORT - 1

            # Filter the already merged full comparison DF
            comparison_df_short = comparison_df_full[comparison_df_full['time'] <= t_end_output_short]

            if not comparison_df_short.empty:
                real_labels_short = comparison_df_short['label']
                pred_labels_short = comparison_df_short['label_pred']
                # Labels should already be 0-indexed from the full comparison part

                if len(real_labels_short) == len(pred_labels_short) and len(real_labels_short) > 0:
                    accuracy_short = accuracy_score(real_labels_short, pred_labels_short)
                    all_accuracy_scores_short.append(accuracy_short)
                    print(f"    -> Accuracy (first {ACCURACY_HORIZON_SHORT} steps): {accuracy_short * 100:.2f}%")
                else:
                    print(f"    -> Warn: Label issue for short comparison at {start_time}. Skip acc.")
            # If full comparison was empty, short will be too, no need for separate warning

        # --- 5. Report Statistics ---
        print("\n" + "=" * 50)
        print("--- Accuracy Benchmark Results ---")
        print(f" (Based on {num_to_run} start times in sequence)")
        print("=" * 50)

        if all_accuracy_scores_short:
            acc_array_short = np.array(all_accuracy_scores_short)
            print(f"\n--- Statistics for First {ACCURACY_HORIZON_SHORT} Steps ({len(acc_array_short)} runs) ---")
            print(f"Mean Accuracy:   {np.mean(acc_array_short) * 100:.2f}%")
            print(f"Median Accuracy: {np.median(acc_array_short) * 100:.2f}%")
            print(f"Max Accuracy:    {np.max(acc_array_short) * 100:.2f}%")
            print(f"Min Accuracy:    {np.min(acc_array_short) * 100:.2f}%")
        else:
            print(f"\nNo accuracy scores calculated for the first {ACCURACY_HORIZON_SHORT} steps.")

        if all_accuracy_scores_full:
            acc_array_full = np.array(all_accuracy_scores_full)
            print(f"\n--- Statistics for All {ACCURACY_HORIZON_FULL} Steps ({len(acc_array_full)} runs) ---")
            print(f"Mean Accuracy:   {np.mean(acc_array_full) * 100:.2f}%")
            print(f"Median Accuracy: {np.median(acc_array_full) * 100:.2f}%")
            print(f"Max Accuracy:    {np.max(acc_array_full) * 100:.2f}%")
            print(f"Min Accuracy:    {np.min(acc_array_full) * 100:.2f}%")
        else:
            print(f"\nNo accuracy scores calculated for all {ACCURACY_HORIZON_FULL} steps.")

        print("=" * 50)

    except FileNotFoundError:
        print(f"Error: '{DATA_CSV}' not found."); return
    except ValueError as ve:
        print(f"Data validation error: {ve}"); return
    except Exception as e:
        print(f"\nAn error occurred: {e}"); import traceback; traceback.print_exc()


# --- Run the Benchmark ---
if __name__ == "__main__":
    # Set seeds
    # random.seed(42); np.random.seed(42); torch.manual_seed(42) # No randomness in start times now
    # if torch.cuda.is_available(): torch.cuda.manual_seed_all(42) # Still good for model init consistency if needed elsewhere
    run_accuracy_benchmark_sequential()
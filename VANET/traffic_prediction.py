import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
from torch_geometric.nn import GCNConv



# ==========================================================
# MODEL DEFINITIONS (Must match the saved model structure)
# ==========================================================
# (Using the powerful models, Dropout=0.2, matching your training X=20, Y=20)
class LSTMEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
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
    # Initialize with the Y the model was *trained* with (e.g., 20)
    def __init__(self, lstm_dim=64, gnn_dim=64, Y=20, hidden_dim=256, classes=5, dropout=0.2):
        super().__init__();
        self.fc1 = nn.Linear(lstm_dim + gnn_dim, hidden_dim);
        self.act = nn.ReLU();
        self.dropout = nn.Dropout(dropout);
        self.fc2 = nn.Linear(hidden_dim, Y * classes);
        self.Y = Y;
        self.classes = classes

    def forward(self, lstm_emb, gnn_emb): x = torch.cat([lstm_emb, gnn_emb], dim=1); x = self.act(
        self.fc1(x)); x = self.dropout(x); x = self.fc2(x); return x.view(-1, self.Y, self.classes)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def normalize_df(df, stats):
    df_norm = df.copy();
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    for col in feature_cols:
        if col in stats: min_val, max_val = stats[col]; denominator = max_val - min_val; df_norm[col] = 0.0 if abs(
            denominator) < 1e-9 else (df_norm[col] - min_val) / denominator
        # else: print(f"Warn: Stats missing for '{col}'. Skip norm.")
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


# --- Function moved from main script ---
def get_current_traffic_features_as_df(current_time, partitions, traffic_data):
    """
    Helper to extract features for the current time into a DataFrame format
    suitable for historical storage and predictor input assembly.
    Includes weather information.
    """
    rows = []
    for partition in partitions:
        hex_id = getattr(partition, 'unique_id', None)  # Safely get hex_id
        if hex_id is None: continue  # Skip if partition has no id

        data = traffic_data.get(hex_id, {})
        is_raining = getattr(partition, 'is_raining', False)  # Safely get weather
        is_sunny = not is_raining
        label = data.get('label', -1)  # Get current label if available

        rows.append({
            'time': current_time, 'hex_id': hex_id,
            'num_vehicles': data.get('num_vehicles', 0),
            'avg_speed': data.get('avg_speed', 0),
            'avg_sin': data.get('avg_sin', 0),
            'avg_cos': data.get('avg_cos', 0),
            'sunny': int(is_sunny), 'rainy': int(is_raining),
            'label': label
        })
    return pd.DataFrame(rows)


# --- End moved function ---


# ==========================================================
# CORE PREDICTOR FUNCTIONS (to be imported)
# ==========================================================
def load_predictor_components(model_path, model_y_trained, num_classes, device):
    """
    Loads predictor components (models, stats, graph). Initializes Decoder
    with the Y the model was trained with.
    Args:
        model_path (str): Path to the saved model directory.
        model_y_trained (int): The prediction horizon (Y) the model was TRAINED with (e.g., 20).
        num_classes (int): Number of output classes.
        device (str): 'cuda' or 'cpu'.
    Returns:
        tuple: (models, stats, graph_info) or None if loading fails.
    """
    print(f"Loading predictor components from '{model_path}/'...")
    if not os.path.exists(model_path): print(f"Error: Predictor model directory '{model_path}' not found."); return None
    try:
        stats = joblib.load(os.path.join(model_path, "stats.pkl"))
        graph_info = joblib.load(os.path.join(model_path, "graph_info.pkl"))
        edge_index_loaded = graph_info['edge_index']

        lstm = LSTMEncoder()  # Assumes saved model matches this structure
        gnn = GNNEncoder()  # Assumes saved model matches this structure
        # --- Initialize Decoder correctly with the TRAINED Y ---
        dec = Decoder(lstm_dim=64, gnn_dim=64, Y=model_y_trained, classes=num_classes)
        # ---

        map_location = torch.device(device)  # Use the determined device (handles CUDA/CPU)

        lstm.load_state_dict(torch.load(os.path.join(model_path, "lstm_model.pth"), map_location=map_location))
        gnn.load_state_dict(torch.load(os.path.join(model_path, "gnn_model.pth"), map_location=map_location))
        dec.load_state_dict(torch.load(os.path.join(model_path, "decoder_model.pth"), map_location=map_location))

        lstm.to(device).eval();
        gnn.to(device).eval();
        dec.to(device).eval()
        graph_info['edge_index'] = edge_index_loaded.to(device)

        models = {'lstm': lstm, 'gnn': gnn, 'dec': dec}
        print("Predictor components loaded successfully.")
        return models, stats, graph_info
    except Exception as e:
        print(f"Error loading predictor components from {model_path}: {e}"); return None


def run_traffic_prediction(input_df, models, stats, graph_info, predictor_x, predictor_y_needed, device):
    """
    Runs traffic prediction and returns only the first predictor_y_needed steps.
    Args:
        input_df (pd.DataFrame): DataFrame containing the last X time steps of features.
        models (dict): Dictionary containing the loaded models ('lstm', 'gnn', 'dec').
        stats (dict): Normalization statistics.
        graph_info (dict): Graph information ('node_to_idx', 'edge_index', 'node_list').
        predictor_x (int): Input sequence length (X).
        predictor_y_needed (int): How many prediction steps the simulation needs (e.g., 12).
        device (str): 'cuda' or 'cpu'.
    Returns:
        pd.DataFrame: DataFrame with the first predictor_y_needed predictions, empty if error.
    """
    if not models or not stats or not graph_info: print("Error: Predictor components missing."); return pd.DataFrame()

    node_to_idx = graph_info['node_to_idx'];
    edge_index = graph_info['edge_index'];
    node_list = graph_info['node_list']

    # --- 1. Data Preparation ---
    df_norm = normalize_df(input_df, stats);
    sequences = []
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    for node in node_list:
        hex_id = f"{node[0]}_{node[1]}";
        node_df_norm = df_norm[df_norm["hex_id"] == hex_id].sort_values("time")
        seq_data = node_df_norm[feature_cols].values
        if len(seq_data) < predictor_x:
            padding = np.zeros((predictor_x - len(seq_data), len(feature_cols))); seq_data = np.vstack(
                [padding, seq_data])
        elif len(seq_data) > predictor_x:
            seq_data = seq_data[-predictor_x:]
        sequences.append(torch.tensor(seq_data, dtype=torch.float32))
    if not sequences: print("Warn: No input sequences built."); return pd.DataFrame()
    sequences = torch.stack(sequences).to(device)
    if input_df.empty or 'time' not in input_df.columns: print(
        "Warn: Input DF empty/missing time."); return pd.DataFrame()
    last_timestep = input_df['time'].max()
    last_step_df = df_norm[df_norm['time'] == last_timestep]
    if last_step_df.empty: print(f"Warn: No norm data for last step {last_timestep}."); return pd.DataFrame()
    snapshots = build_snapshots(last_step_df, node_list, node_to_idx)
    if last_timestep not in snapshots: print(f"Warn: Snap {last_timestep} miss."); return pd.DataFrame()
    snapshot_tensor = snapshots[last_timestep].to(device)

    # --- 2. Run Model (Predicts FULL Y=20 steps) ---
    try:
        with torch.no_grad():
            lstm_emb = models['lstm'](sequences);
            gnn_emb = models['gnn'](snapshot_tensor, edge_index)
            if lstm_emb.shape[0] != gnn_emb.shape[0]: print(f"Warn: LSTM/GNN shape mismatch."); return pd.DataFrame()
            logits = models['dec'](lstm_emb, gnn_emb);  # This outputs Y=20 steps
            predicted_labels_full = torch.argmax(logits, dim=2).cuda().numpy()  # Shape: (num_nodes, 20)
    except Exception as e:
        print(f"Error during inference: {e}"); return pd.DataFrame()

    # --- 3. Format Output (Filter to first predictor_y_needed steps) ---
    output_rows = [];
    start_time_pred = last_timestep + 1
    for i in range(len(node_list)):
        hex_id = f"{node_list[i][0]}_{node_list[i][1]}"
        # --- Only loop for the steps needed ---
        for j in range(predictor_y_needed):  # e.g., range(12)
            output_rows.append(
                {'time': start_time_pred + j, 'hex_id': hex_id, 'label_pred': predicted_labels_full[i, j] + 1})

    return pd.DataFrame(output_rows)
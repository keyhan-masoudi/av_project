import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
import joblib  # Used for saving Python objects
import copy  # Used for Early Stopping
import time  # Added for timing epochs
import math
from collections import defaultdict
from torch_geometric.nn import GCNConv


# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_CSV = "../testfile/final300.csv"
# DATA_CSV = "../testfile/final250.csv"
# DATA_CSV = "../testfile/final200.csv"
# DATA_CSV = "../testfile/final150.csv"
# DATA_CSV = "../testfile/final100.csv"
X = 15
Y = 12
EPOCHS = 60  # Max epochs
BATCH_SIZE = 128
NUM_CLASSES = 5
# MODEL_SAVE_PATH = "final_model100"  # Save to a new folder
# MODEL_SAVE_PATH = "final_model150"  # Save to a new folder
# MODEL_SAVE_PATH = "final_model200"  # Save to a new folder
# MODEL_SAVE_PATH = "final_model250"  # Save to a new folder
MODEL_SAVE_PATH = "final_model300"  # Save to a new folder

MIN_X = 29.24
MAX_X = 2128.64
MIN_Y = 30.38
MAX_Y = 1586.68
HEX_SIZE = 200  # This is your 'radius'

# --- Training Data Time Range ---
TRAIN_START_TIME = 300
TRAIN_END_TIME = 2600

# Must be AFTER the training range and contain sufficient data
VALIDATION_START_TIME = 2650
VALIDATION_END_TIME = 3150  

# --- Early Stopping Config (Based on Validation Loss) ---
PATIENCE = 10  

# --- Optimizer Config ---
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5  

# --- Gradient Clipping ---
CLIP_GRAD_NORM = 1.0  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
HEX_DIRECTIONS = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]


# ==========================================================
# MODEL DEFINITIONS (Unchanged GCN Architecture)
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


# ==========================================================
# HELPER FUNCTIONS (Your Tested Logic)
# ==========================================================
def _xy_to_axial(x, y, size):
    q_f = (np.sqrt(3) / 3 * x - 1 / 3 * y) / size
    r_f = (2 / 3 * y) / size
    q = round(q_f)
    r = round(r_f)
    s = round(-q_f - r_f)

    q_diff = abs(q - q_f)
    r_diff = abs(r - r_f)
    s_diff = abs(s - (-q_f - r_f))

    if q_diff > r_diff and q_diff > s_diff:
        q = -r - s
    elif r_diff > s_diff:
        r = -q - s
    return (int(q), int(r))


def compute_norm_stats(df, start_time, end_time):
    print(f"Computing normalization stats for time range {start_time}-{end_time}...")
    train_df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
    stats = {}
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    if train_df.empty:
        return {col: (0.0, 1.0) for col in feature_cols}
    for col in feature_cols:
        if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
            min_val, max_val = train_df[col].min(), train_df[col].max()
            stats[col] = (min_val, max_val)
        else:
            stats[col] = (0.0, 1.0)
    return stats


def normalize_df(df, stats):
    df_norm = df.copy()
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    for col in feature_cols:
        if col in stats:
            min_val, max_val = stats[col]
            denominator = max_val - min_val
            if abs(denominator) < 1e-9:
                df_norm[col] = 0.0
            else:
                df_norm[col] = (df_norm[col] - min_val) / denominator
    return df_norm


def build_samples(df, node_to_idx, X, Y, start_time, end_time):
    samples = []
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    df_copy = df.copy()
    if df_copy['label'].min() == 1: 
        df_copy['label'] = df_copy['label'] - 1

    grouped = df_copy.groupby("hex_id")
    total_groups = len(grouped)
    print(f"Building samples ({start_time}-{end_time}) for {total_groups} hex_ids...")
    count = 0
    for hex_id, group in grouped:
        count += 1
        if count % 200 == 0: print(f"  Processed {count}/{total_groups}...")
        group = group.sort_values("time").reset_index(drop=True)
        node_idx = node_to_idx.get(tuple(map(int, hex_id.split("_"))))
        if node_idx is None or len(group) < X + Y: continue

        features = group[feature_cols].values.astype(np.float32)
        labels = group["label"].values
        times = group["time"].values

        for i in range(len(features) - X - Y + 1):
            last_input_time = int(times[i + X - 1])
            if last_input_time < start_time or last_input_time > end_time: continue
            input_seq = features[i:i + X]
            target_seq = labels[i + X:i + X + Y]
            if len(target_seq) != Y: continue
            samples.append({
                "seq": torch.tensor(input_seq, dtype=torch.float32),
                "node_idx": node_idx,
                "t_last": last_input_time,
                "targets": torch.tensor(target_seq, dtype=torch.long)
            })
    print(f"Finished building samples ({start_time}-{end_time}). Total: {len(samples)}")
    return samples


def build_snapshots(df, node_list, node_to_idx):
    snapshots = {}
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos", "sunny", "rainy"]
    if df.empty: return snapshots
    unique_times = sorted(df["time"].unique())
    if not unique_times: return snapshots

    num_nodes = len(node_list)
    num_features = len(feature_cols)
    node_hexid_to_idx_map = {f"{q}_{r}": node_to_idx.get((q, r)) for q, r in node_list if
                             node_to_idx.get((q, r)) is not None}

    print(f"Building snapshots for {len(unique_times)} time steps...")
    count = 0
    for t in unique_times:
        count += 1
        if count % 200 == 0: print(f"  Processed {count}/{len(unique_times)}...")
        snapshot_features = np.zeros((num_nodes, num_features), dtype=np.float32)
        sub_df = df.loc[df["time"] == t]
        valid_rows = sub_df[sub_df['hex_id'].isin(node_hexid_to_idx_map.keys())]

        if not valid_rows.empty:
            indices = valid_rows['hex_id'].map(node_hexid_to_idx_map).values
            values = valid_rows[feature_cols].values
            valid_mask = ~np.isnan(indices) & (indices < num_nodes)
            snapshot_features[indices[valid_mask].astype(int)] = values[valid_mask]
        snapshots[t] = torch.tensor(snapshot_features, dtype=torch.float32)
    print("Finished building snapshots.")
    return snapshots


def build_hex_graph(df=None):
    print("Building static hexagonal graph based on min/max map dimensions...")
    unique_hex_coords_set = set()
    hex_width = math.sqrt(3) * HEX_SIZE
    row_spacing = 1.5 * HEX_SIZE

    min_row = int(math.floor(MIN_Y / row_spacing)) - 1
    max_row = int(math.ceil(MAX_Y / row_spacing)) + 1
    min_col = int(math.floor(MIN_X / hex_width)) - 1
    max_col = int(math.ceil(MAX_X / hex_width)) + 1

    for row in range(min_row, max_row + 1):
        row_is_even = (row % 2 == 0)
        start_x = 0.0 if row_is_even else (hex_width / 2.0)
        center_y = row * row_spacing

        for col in range(min_col, max_col + 1):
            center_x = start_x + col * hex_width
            if (MIN_X - hex_width/2 <= center_x <= MAX_X + hex_width/2) and \
               (MIN_Y - row_spacing/2 <= center_y <= MAX_Y + row_spacing/2):
                q, r = _xy_to_axial(center_x, center_y, HEX_SIZE)
                unique_hex_coords_set.add((q, r))

    unique_hex_coords = sorted(list(unique_hex_coords_set))
    if not unique_hex_coords:
        return {}, torch.empty((2, 0), dtype=torch.long), []

    node_to_idx = {h: i for i, h in enumerate(unique_hex_coords)}
    idx_to_node = {i: h for h, i in node_to_idx.items()}
    num_nodes = len(unique_hex_coords)
    edges = []

    for i in range(num_nodes):
        q, r = idx_to_node[i]
        for dq, dr in HEX_DIRECTIONS:
            neighbor_hex = (q + dq, r + dr)
            if neighbor_hex in node_to_idx:
                edges.append((i, node_to_idx[neighbor_hex]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Graph built: {num_nodes} nodes, {edge_index.shape[1]} edges.")
    return node_to_idx, edge_index, unique_hex_coords


# ==========================================================
# TRAINING FUNCTION (OPTIMIZED LOOPS & CACHING)
# ==========================================================
def train_model(train_samples, val_samples, 
                snapshots, edge_index, lstm, gnn, dec,
                epochs, device, batch_size, patience,  
                learning_rate, weight_decay, clip_grad_norm):
    lstm.to(device); gnn.to(device); dec.to(device)
    edge_index = edge_index.to(device)

    # Clean explicit parameter combination
    optimizer = optim.AdamW(list(lstm.parameters()) + list(gnn.parameters()) + list(dec.parameters()),
                            lr=learning_rate, weight_decay=weight_decay)
    
    # OPTIMIZATION 1: Adaptive Scheduler based on Val Loss validation metrics
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    train_ids = list(range(len(train_samples)))

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_lstm_state, best_gnn_state, best_dec_state = None, None, None

    print(f"Starting training loop with validation...")
    for ep in range(1, epochs + 1):
        epoch_start_time = time.time()

        # --- Training Phase ---
        lstm.train(); gnn.train(); dec.train()
        random.shuffle(train_ids)
        total_train_loss = 0
        samples_processed_train = 0

        # OPTIMIZATION 2: Global Epoch Cache initialization (Massive Time Saving)
        train_gnn_cache = {} 

        for b_start in range(0, len(train_ids), batch_size):
            b_end = min(b_start + batch_size, len(train_ids))
            batch_indices = train_ids[b_start:b_end]
            batch = [train_samples[i] for i in batch_indices]
            valid_batch = [s for s in batch if s["t_last"] in snapshots]
            if not valid_batch: continue

            seqs = torch.stack([s["seq"] for s in valid_batch]).to(device)
            gnn_nodes, targets = [], []
            valid_indices_for_lstm = []

            for i, s in enumerate(valid_batch):
                t = s["t_last"]
                node_idx_val = s["node_idx"]
                
                # Reads from global epoch cache instead of wiping data every batch step
                # Updated Fix Block
                if t not in train_gnn_cache:
                    x_snap = snapshots[t].to(device)
                    try:
                        # We compute it normally for gradients on its first pass
                        train_gnn_cache[t] = gnn(x_snap, edge_index)
                    except Exception:
                        continue
                if node_idx_val < train_gnn_cache[t].shape[0]:
                    # .detach() prevents PyTorch from tracking its history across different batches!
                    gnn_nodes.append(train_gnn_cache[t][node_idx_val].detach())
                    targets.append(s["targets"])
                    valid_indices_for_lstm.append(i)

            if not gnn_nodes: continue
            gnn_emb = torch.stack(gnn_nodes).to(device)
            targets = torch.stack(targets).to(device)
            seqs_valid = seqs[valid_indices_for_lstm]

            try:
                lstm_emb = lstm(seqs_valid)
                if lstm_emb.shape[0] != gnn_emb.shape[0]: continue
                logits = dec(lstm_emb, gnn_emb)
            except RuntimeError:
                continue

            # OPTIMIZATION 3: Smoothed target decay to protect late-stage horizon evaluation metrics
            weights = torch.tensor([0.95 ** k for k in range(Y)], device=device)
            weights /= weights.sum()
            try:
                loss_components = [loss_fn(logits[:, k, :], targets[:, k]) for k in range(Y)]
                loss = sum(weights[k] * loss_components[k] for k in range(Y))
            except IndexError:
                continue

            optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm > 0: 
                torch.nn.utils.clip_grad_norm_(list(lstm.parameters()) + list(gnn.parameters()) + list(dec.parameters()), clip_grad_norm)
            optimizer.step()
            total_train_loss += loss.item() * len(gnn_nodes)
            samples_processed_train += len(gnn_nodes)

        avg_train_loss = total_train_loss / samples_processed_train if samples_processed_train > 0 else 0.0

        # --- Validation Phase ---
        lstm.eval(); gnn.eval(); dec.eval()
        total_val_loss = 0
        samples_processed_val = 0
        
        # Global Epoch Cache for validation pass
        val_gnn_cache = {} 

        if val_samples:  
            with torch.no_grad():
                for b_start in range(0, len(val_samples), batch_size):
                    b_end = min(b_start + batch_size, len(val_samples))
                    batch = val_samples[b_start:b_end]  
                    valid_batch = [s for s in batch if s["t_last"] in snapshots]
                    if not valid_batch: continue

                    seqs = torch.stack([s["seq"] for s in valid_batch]).to(device)
                    gnn_nodes, targets = [], []
                    valid_indices_for_lstm = []

                    for i, s in enumerate(valid_batch):
                        t = s["t_last"]
                        node_idx_val = s["node_idx"]
                        if t not in val_gnn_cache:
                            x_snap = snapshots[t].to(device)
                            try:
                                val_gnn_cache[t] = gnn(x_snap, edge_index)
                            except Exception:
                                continue
                        if node_idx_val < val_gnn_cache[t].shape[0]:
                            gnn_nodes.append(val_gnn_cache[t][node_idx_val])
                            targets.append(s["targets"])
                            valid_indices_for_lstm.append(i)

                    if not gnn_nodes: continue
                    gnn_emb = torch.stack(gnn_nodes).to(device)
                    targets = torch.stack(targets).to(device)
                    seqs_valid = seqs[valid_indices_for_lstm]

                    try:
                        lstm_emb = lstm(seqs_valid)
                        if lstm_emb.shape[0] != gnn_emb.shape[0]: continue
                        logits = dec(lstm_emb, gnn_emb)
                    except RuntimeError:
                        continue

                    weights = torch.tensor([0.95 ** k for k in range(Y)], device=device)
                    weights /= weights.sum()
                    try:
                        loss_components = [loss_fn(logits[:, k, :], targets[:, k]) for k in range(Y)]
                        val_loss = sum(weights[k] * loss_components[k] for k in range(Y))
                    except IndexError:
                        continue

                    total_val_loss += val_loss.item() * len(gnn_nodes)
                    samples_processed_val += len(gnn_nodes)

            avg_val_loss = total_val_loss / samples_processed_val if samples_processed_val > 0 else 0.0
        else:
            avg_val_loss = 0.0  

        # Step the plateau scheduler with validation performance trends
        scheduler.step(avg_val_loss)  
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {ep}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Duration: {epoch_duration:.2f}s")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_lstm_state = copy.deepcopy(lstm.state_dict())
            best_gnn_state = copy.deepcopy(gnn.state_dict())
            best_dec_state = copy.deepcopy(dec.state_dict())
            print(f"  -> New best model saved with Val Loss: {best_val_loss:.5f}")
        elif avg_val_loss > 0:  
            epochs_no_improve += 1
            print(f"  -> Val Loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"--- Early stopping triggered after {ep} epochs ---")
            break

    print("Training loop finished.")

    if best_lstm_state:
        print(f"Loading best model weights (Val Loss: {best_val_loss:.5f})")
        lstm.load_state_dict(best_lstm_state)
        gnn.load_state_dict(best_gnn_state)
        dec.load_state_dict(best_dec_state)
    else:
        print("Warning: Returning last state.")

    return lstm, gnn, dec


# ==========================================================
# MAIN TRAIN EXECUTION BLOCK (Your Exact Pipeline Layout)
# ==========================================================
def main_train():
    start_time_main = time.time()
    print(f"Starting main training process...")

    print(f"Loading data from {DATA_CSV}...")
    try:
        df = pd.read_csv(DATA_CSV)
        if 'label' not in df.columns: raise ValueError("'label' column not found")
        if not pd.api.types.is_numeric_dtype(df['label']):
            df['label'] = pd.to_numeric(df['label'], errors='coerce').dropna().astype(int)
            print("Converted 'label' to numeric, dropped invalid.")
        if df['label'].min() == 1:
            print("Adjusting labels to 0-index.")
            df['label'] = df['label'] - 1
        if 'weather' not in df.columns: raise ValueError("'weather' column not found")
    except Exception as e:
        print(f"Error loading CSV: {e}"); return

    print("Creating weather features...")
    df['sunny'] = (df['weather'] == 1).astype(int)
    df['rainy'] = (df['weather'] == 2).astype(int)

    node_to_idx, edge_index, node_list = build_hex_graph(df)
    if not node_to_idx: print("Error: Graph building failed."); return

    print(f"Normalizing features based on training data ({TRAIN_START_TIME}-{TRAIN_END_TIME})...")
    stats = compute_norm_stats(df, TRAIN_START_TIME, TRAIN_END_TIME)

    required_start = min(TRAIN_START_TIME, VALIDATION_START_TIME) - X + 1
    required_end = max(TRAIN_END_TIME, VALIDATION_END_TIME) + Y
    df_slice_for_processing = df[(df["time"] >= required_start) & (df["time"] <= required_end)].copy()
    if df_slice_for_processing.empty: print("Error: No data in specified time range for processing."); return
    df_norm = normalize_df(df_slice_for_processing, stats)

    print(f"Building training samples ({TRAIN_START_TIME}-{TRAIN_END_TIME})...")
    train_samples = build_samples(df_norm, node_to_idx, X, Y, TRAIN_START_TIME, TRAIN_END_TIME)
    if not train_samples: print("Error: No training samples generated."); return

    print(f"Building validation samples ({VALIDATION_START_TIME}-{VALIDATION_END_TIME})...")
    val_samples = build_samples(df_norm, node_to_idx, X, Y, VALIDATION_START_TIME, VALIDATION_END_TIME)

    print("Building snapshots for required time steps...")
    required_train_times = {s['t_last'] for s in train_samples}
    required_val_times = {s['t_last'] for s in val_samples}
    all_required_times = required_train_times.union(required_val_times)

    snapshot_df = df_norm[df_norm['time'].isin(all_required_times)]
    snapshots = {}
    if not snapshot_df.empty:
        snapshots = build_snapshots(snapshot_df, node_list, node_to_idx)

    missing_snapshots = all_required_times - snapshots.keys()
    if missing_snapshots:
        train_samples = [s for s in train_samples if s['t_last'] in snapshots]
        val_samples = [s for s in val_samples if s['t_last'] in snapshots]
        if not train_samples: print("Error: No training samples left after snapshot filtering."); return

    print("Initializing models...")
    lstm = LSTMEncoder()
    gnn = GNNEncoder()
    dec = Decoder(lstm_dim=64, gnn_dim=64, Y=Y, classes=NUM_CLASSES)

    print(f"Starting training up to {EPOCHS} epochs on {DEVICE}...")

    # Removed legacy positional step inputs to conform to Plateau requirements
    lstm, gnn, dec = train_model(
        train_samples, val_samples,  
        snapshots, edge_index,
        lstm, gnn, dec,
        epochs=EPOCHS, device=DEVICE, batch_size=BATCH_SIZE,
        patience=PATIENCE,  
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        clip_grad_norm=CLIP_GRAD_NORM
    )

    print("Training complete.")
    print(f"\nSaving BEST model components to '{MODEL_SAVE_PATH}/'...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    lstm.to('cpu'); gnn.to('cpu'); dec.to('cpu')
    torch.save(lstm.state_dict(), os.path.join(MODEL_SAVE_PATH, "lstm_model.pth"))
    torch.save(gnn.state_dict(), os.path.join(MODEL_SAVE_PATH, "gnn_model.pth"))
    torch.save(dec.state_dict(), os.path.join(MODEL_SAVE_PATH, "decoder_model.pth"))

    joblib.dump(stats, os.path.join(MODEL_SAVE_PATH, "stats.pkl"))
    graph_info = {'node_to_idx': node_to_idx, 'edge_index': edge_index.cpu(), 'node_list': node_list}
    joblib.dump(graph_info, os.path.join(MODEL_SAVE_PATH, "graph_info.pkl"))

    print("✅ All components saved successfully.")
    end_time_main = time.time()
    print(f"Total script duration: {(end_time_main - start_time_main) / 60:.2f} minutes")


# ==========================================================
# SEEDING & EXECUTION
# ==========================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

if __name__ == "__main__":
    main_train()
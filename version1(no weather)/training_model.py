import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
import joblib # Used for saving Python objects
from collections import defaultdict
from torch_geometric.nn import GCNConv

# -----------------------------
# Configuration
# -----------------------------
DATA_CSV = "new_traffic_dataset.csv"
X = 10
Y = 5
EPOCHS = 20
BATCH_SIZE = 128
NUM_CLASSES = 5
MODEL_SAVE_PATH = "saved_model" # Folder to save model files

TRAIN_START_TIME = 400
TRAIN_END_TIME = 1300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEX_DIRECTIONS = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]

# ==========================================================
# RE-DEFINE NECESSARY FUNCTIONS & MODEL CLASSES
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
    def __init__(self, lstm_dim=32, gnn_dim=32, Y=5, hidden_dim=128, classes=5):
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
def compute_norm_stats(df, start_time, end_time):
    train_df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
    stats = {}
    # This list correctly ignores the 'weather' column
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    for col in feature_cols:
        stats[col] = (train_df[col].min(), train_df[col].max())
    return stats


def normalize_df(df, stats):
    df_norm = df.copy()
    # This list correctly ignores the 'weather' column
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    for col in feature_cols:
        min_val, max_val = stats[col]
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val + 1e-9)
    return df_norm

def build_samples(df, node_to_idx, X, Y, start_time, end_time):
    samples = []
    # This list correctly ignores the 'weather' column
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    for hex_id, group in df.groupby("hex_id"):
        group = group.sort_values("time").reset_index(drop=True)
        features = group[feature_cols].values.astype(np.float32)
        labels = group["label"].values
        times = group["time"].values
        node_idx = node_to_idx.get(tuple(map(int, hex_id.split("_"))))
        if node_idx is None: continue

        for i in range(len(features) - X - Y + 1):
            last_input_time = int(times[i + X - 1])
            if last_input_time < start_time or last_input_time > end_time:
                continue
            input_seq = features[i:i + X]
            target_seq = labels[i + X:i + X + Y]
            samples.append({
                "seq": torch.tensor(input_seq, dtype=torch.float32),
                "node_idx": node_idx,
                "t_last": last_input_time,
                "targets": torch.tensor(target_seq, dtype=torch.long)
            })
    return samples

def build_snapshots(df, node_list, node_to_idx):
    snapshots = {}
    # This list correctly ignores the 'weather' column
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    unique_times = sorted(df["time"].unique())
    for t in unique_times:
        snapshot_features = np.zeros((len(node_list), len(feature_cols)), dtype=np.float32)
        sub_df = df[df["time"] == t]
        for _, row in sub_df.iterrows():
            q, r = map(int, row.hex_id.split("_"))
            idx = node_to_idx.get((q, r))
            if idx is not None:
                # This will only select the columns in feature_cols
                snapshot_features[idx] = row[feature_cols].values
        snapshots[t] = torch.tensor(snapshot_features, dtype=torch.float32)
    return snapshots

def build_hex_graph(df):
    unique_hex_coords = sorted({tuple(map(int, h.split("_"))) for h in df["hex_id"].unique()})
    node_to_idx = {h: i for i, h in enumerate(unique_hex_coords)}
    edges = []
    for (q, r) in unique_hex_coords:
        i = node_to_idx[(q, r)]
        for dq, dr in HEX_DIRECTIONS:
            neighbor_hex = (q + dq, r + dr)
            if neighbor_hex in node_to_idx:
                edges.append((i, node_to_idx[neighbor_hex]))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return node_to_idx, edge_index, unique_hex_coords

# -----------------------------
# 6. Training Function
# -----------------------------
def train_model(samples, snapshots, edge_index, lstm, gnn, dec, epochs, device, batch_size):
    lstm.to(device); gnn.to(device); dec.to(device)
    edge_index = edge_index.to(device)
    optimizer = optim.Adam(list(lstm.parameters()) + list(gnn.parameters()) + list(dec.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    ids = list(range(len(samples)))
    for ep in range(1, epochs + 1):
        random.shuffle(ids)
        total_loss = 0
        for b_start in range(0, len(ids), batch_size):
            b_end = b_start + batch_size
            batch_indices = ids[b_start:b_end]
            batch = [samples[i] for i in batch_indices]
            seqs = torch.stack([s["seq"] for s in batch]).to(device)
            lstm_emb = lstm(seqs)
            gnn_cache = {}
            for s in batch:
                t = s["t_last"]
                if t not in gnn_cache:
                    x_snap = snapshots[t].to(device)
                    gnn_cache[t] = gnn(x_snap, edge_index)
            gnn_nodes = []
            targets = []
            for s in batch:
                gnn_nodes.append(gnn_cache[s["t_last"]][s["node_idx"]])
                targets.append(s["targets"])
            gnn_emb = torch.stack(gnn_nodes).to(device)
            targets = torch.stack(targets).to(device)
            logits = dec(lstm_emb, gnn_emb)
            loss = sum(loss_fn(logits[:, k, :], targets[:, k]) for k in range(dec.Y)) / dec.Y
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / len(samples)
        print(f"Epoch {ep}/{epochs}  |  Average Loss: {avg_loss:.5f}")
    return lstm, gnn, dec
# ==========================================================
# Main Training Execution Block
# ==========================================================
def main_train():
    """Trains the model and saves all necessary components to disk."""
    print(f"Loading data from {DATA_CSV}...")
    try:
        df = pd.read_csv(DATA_CSV)
        if df['label'].min() == 1:
            df['label'] = df['label'] - 1
    except FileNotFoundError:
        print(f"Error: '{DATA_CSV}' not found. Please upload it to your Colab session.")
        return

    print("Building hexagonal graph...")
    node_to_idx, edge_index, node_list = build_hex_graph(df)

    print(f"Normalizing features based on data from time {TRAIN_START_TIME} to {TRAIN_END_TIME}...")
    stats = compute_norm_stats(df, TRAIN_START_TIME, TRAIN_END_TIME)
    df_norm = normalize_df(df, stats)

    print(f"Building training samples from time {TRAIN_START_TIME} to {TRAIN_END_TIME}...")
    train_samples = build_samples(df_norm, node_to_idx, X, Y, TRAIN_START_TIME, TRAIN_END_TIME)
    if not train_samples:
        print("No training samples were generated.")
        return

    print("Building temporal snapshots for GNN...")
    # We build snapshots for the whole dataset for later use in fine-tuning/prediction
    # but only the relevant ones (from 500-1300) will be used in training
    snapshots = build_snapshots(df_norm, node_list, node_to_idx)

    print("Initializing models...")
    lstm = LSTMEncoder()
    gnn = GNNEncoder()
    dec = Decoder(Y=Y, classes=NUM_CLASSES)

    print(f"Starting training for {EPOCHS} epochs on device: {DEVICE}...")
    lstm, gnn, dec = train_model(
        train_samples, snapshots, edge_index, lstm, gnn, dec,
        epochs=EPOCHS, device=DEVICE, batch_size=BATCH_SIZE
    )
    print("Training complete.")

    # --- Save the trained model and helper objects ---
    print(f"\nSaving model components to '{MODEL_SAVE_PATH}/'...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    torch.save(lstm.state_dict(), os.path.join(MODEL_SAVE_PATH, "lstm_model.pth"))
    torch.save(gnn.state_dict(), os.path.join(MODEL_SAVE_PATH, "gnn_model.pth"))
    torch.save(dec.state_dict(), os.path.join(MODEL_SAVE_PATH, "decoder_model.pth"))

    # Save the normalization stats and graph info
    joblib.dump(stats, os.path.join(MODEL_SAVE_PATH, "stats.pkl"))
    graph_info = {'node_to_idx': node_to_idx, 'edge_index': edge_index, 'node_list': node_list}
    joblib.dump(graph_info, os.path.join(MODEL_SAVE_PATH, "graph_info.pkl"))

    print("All components saved successfully.")

# Set random seeds before running
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Run the training process
main_train()
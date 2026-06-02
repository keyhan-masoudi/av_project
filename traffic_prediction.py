import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import joblib

# -----------------------------
# Configuration
# -----------------------------
DATA_CSV = "traffic_dataset.csv"
X = 10
Y = 5
EPOCHS = 30
BATCH_SIZE = 128
NUM_CLASSES = 5

TRAIN_START_TIME = 300
TRAIN_END_TIME = 1000
TEST_START_TIME = 1001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEX_DIRECTIONS = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]

MODEL_SAVE_PATH = "saved_model"

# -----------------------------
# 1. Build Hexagonal Graph
# -----------------------------
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
# 2. Normalize Features
# -----------------------------
def compute_norm_stats(df, start_time, end_time):
    train_df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
    stats = {}
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    for col in feature_cols:
        stats[col] = (train_df[col].min(), train_df[col].max())
    return stats

def normalize_df(df, stats):
    df_norm = df.copy()
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    for col in feature_cols:
        min_val, max_val = stats[col]
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val + 1e-9)
    return df_norm

# -----------------------------
# 3. Build Training/Testing Samples
# -----------------------------
def build_samples(df, node_to_idx, X, Y, start_time, end_time):
    samples = []
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

# -----------------------------
# 4. Build Temporal Snapshots for GNN
# -----------------------------
def build_snapshots(df, node_list, node_to_idx):
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    snapshots = {}
    for t in sorted(df["time"].unique()):
        snapshot_features = np.zeros((len(node_list), len(feature_cols)), dtype=np.float32)
        sub_df = df[df["time"] == t]
        for _, row in sub_df.iterrows():
            q, r = map(int, row.hex_id.split("_"))
            idx = node_to_idx.get((q, r))
            if idx is not None:
                snapshot_features[idx] = row[feature_cols].values
        snapshots[t] = torch.tensor(snapshot_features, dtype=torch.float32)
    return snapshots

# -----------------------------
# 5. Model Definitions
# -----------------------------
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
    def __init__(self, lstm_dim=32, gnn_dim=32, Y=5, hidden_dim=128, classes=3):
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

# -----------------------------
# 7. Evaluation Function
# -----------------------------
def evaluate_model(lstm, gnn, dec, test_samples, snapshots, edge_index, device, Y, num_classes):
    lstm.to(device).eval(); gnn.to(device).eval(); dec.to(device).eval()
    edge_index = edge_index.to(device)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for s in test_samples:
            seq = s["seq"].unsqueeze(0).to(device)
            t_last = s["t_last"]
            node_idx = s["node_idx"]
            lstm_emb = lstm(seq)
            x_snap = snapshots[t_last].to(device)
            gnn_emb_all = gnn(x_snap, edge_index)
            gnn_emb = gnn_emb_all[node_idx].unsqueeze(0)
            logits = dec(lstm_emb, gnn_emb)
            preds = torch.argmax(logits, dim=2).cpu().numpy().flatten()
            all_preds.append(preds)
            all_targets.append(s["targets"].numpy())
    if not all_targets:
        print("No test samples found to evaluate.")
        return
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    print("\n--- Model Evaluation Results ---")
    for i in range(Y):
        print(f"\n>>> Metrics for Prediction Timestep t+{i+1}")
        accuracy = accuracy_score(all_targets[:, i], all_preds[:, i])
        print(f"Accuracy: {accuracy:.4f}")
        report = classification_report(
            all_targets[:, i], all_preds[:, i], labels=range(num_classes),
            target_names=[f"Class {j}" for j in range(num_classes)], zero_division=0
        )
        print("Classification Report:")
        print(report)

# -----------------------------
# 8. NEW: Prediction Comparison Function
# -----------------------------
def predict_and_compare(start_time, df, df_norm, snapshots, node_list, node_to_idx, edge_index,
                        lstm, gnn, dec, X, Y, device):
    """Makes a prediction for a specific start time and compares it to the real labels."""
    print(f"\n--- Comparing Prediction for Timestep {start_time} ---")
    lstm.to(device).eval(); gnn.to(device).eval(); dec.to(device).eval()
    edge_index = edge_index.to(device)

    # 1. Prepare the input data for the model
    input_start_time = start_time - X
    input_end_time = start_time - 1

    sequences = []
    feature_cols = ["num_vehicles", "avg_speed", "avg_sin", "avg_cos"]
    for node in node_list:
        hex_id = f"{node[0]}_{node[1]}"
        # Get the normalized data for the input window
        node_df_norm = df_norm[(df_norm["hex_id"] == hex_id) &
                               (df_norm["time"] >= input_start_time) &
                               (df_norm["time"] <= input_end_time)]
        node_df_norm = node_df_norm.sort_values("time")
        
        # Handle cases where a node might not have data for all X timesteps
        seq_data = node_df_norm[feature_cols].values
        if len(seq_data) < X:
            # Pad with zeros if data is missing
            padding = np.zeros((X - len(seq_data), len(feature_cols)))
            seq_data = np.vstack([padding, seq_data])
        sequences.append(torch.tensor(seq_data, dtype=torch.float32))

    sequences = torch.stack(sequences).to(device)
    
    # 2. Make the prediction
    with torch.no_grad():
        lstm_emb = lstm(sequences)
        snapshot_time = input_end_time
        x_snap = snapshots[snapshot_time].to(device)
        gnn_emb = gnn(x_snap, edge_index)
        logits = dec(lstm_emb, gnn_emb)
        predicted_labels = torch.argmax(logits, dim=2).cpu().numpy()

    # 3. Get the real labels from the original dataframe
    real_labels = []
    for node in node_list:
        hex_id = f"{node[0]}_{node[1]}"
        real_df = df[(df["hex_id"] == hex_id) &
                     (df["time"] >= start_time) &
                     (df["time"] < start_time + Y)]
        real_df = real_df.sort_values("time")
        
        node_real_labels = real_df["label"].values
        # Pad with a placeholder (e.g., -1) if real data is missing
        if len(node_real_labels) < Y:
            padding = np.full(Y - len(node_real_labels), -1)
            node_real_labels = np.concatenate([node_real_labels, padding])
        real_labels.append(node_real_labels)
    
    real_labels = np.array(real_labels)

    # 4. Display the comparison
    comparison_dfs = []
    for i in range(Y):
        df_t = pd.DataFrame({
            'hex_id': [f"{q}_{r}" for q, r in node_list],
            f'Predicted_t+{i+1}': predicted_labels[:, i],
            f'Real_t+{i+1}': real_labels[:, i]
        })
        comparison_dfs.append(df_t.set_index('hex_id'))
        
    final_comparison_df = pd.concat(comparison_dfs, axis=1)
    print("Side-by-side comparison (first 15 zones):")
    print(final_comparison_df.head(15))


# ==========================================================
# Main Execution Block
# # ==========================================================
# def main():
#     print(f"Loading data from {DATA_CSV}...")
#     try:
#         df = pd.read_csv(DATA_CSV)
#         if df['label'].min() == 1:
#             df['label'] = df['label'] - 1
#     except FileNotFoundError:
#         print(f"Error: '{DATA_CSV}' not found. Please update the DATA_CSV variable.")
#         return

#     print("Building hexagonal graph...")
#     node_to_idx, edge_index, node_list = build_hex_graph(df)
#     print(f"Graph built with {len(node_list)} nodes and {edge_index.shape[1]} edges.")

#     print(f"Normalizing features based on data from time {TRAIN_START_TIME} to {TRAIN_END_TIME}...")
#     stats = compute_norm_stats(df, TRAIN_START_TIME, TRAIN_END_TIME)
#     df_norm = normalize_df(df, stats)

#     print(f"Building training samples from time {TRAIN_START_TIME} to {TRAIN_END_TIME}...")
#     train_samples = build_samples(df_norm, node_to_idx, X, Y, TRAIN_START_TIME, TRAIN_END_TIME)
#     print(f"Created {len(train_samples)} training samples.")
#     if not train_samples:
#         print("No training samples were generated.")
#         return

#     print("Building temporal snapshots for GNN...")
#     snapshots = build_snapshots(df_norm, node_list, node_to_idx)

#     print("Initializing models...")
#     lstm = LSTMEncoder()
#     gnn = GNNEncoder()
#     dec = Decoder(Y=Y, classes=NUM_CLASSES)

#     print(f"Starting training for {EPOCHS} epochs on device: {DEVICE}...")
#     lstm, ggn, dec = train_model(
#         train_samples, snapshots, edge_index, lstm, gnn, dec,
#         epochs=EPOCHS, device=DEVICE, batch_size=BATCH_SIZE
#     )
#     print("Training complete.")

#     print(f"\nBuilding test samples from time {TEST_START_TIME} onwards...")
#     max_time = df['time'].max()
#     test_samples = build_samples(df_norm, node_to_idx, X, Y, TEST_START_TIME, max_time)
#     print(f"Created {len(test_samples)} test samples.")

#     evaluate_model(lstm, gnn, dec, test_samples, snapshots, edge_index, DEVICE, Y, NUM_CLASSES)

#     # Use the new function to check a specific prediction
#     # You can change the timestep here (e.g., 1001, 1010, etc.)
#     predict_and_compare(start_time=TEST_START_TIME, df=df, df_norm=df_norm, snapshots=snapshots,
#                         node_list=node_list, node_to_idx=node_to_idx, edge_index=edge_index,
#                         lstm=lstm, gnn=gnn, dec=dec, X=X, Y=Y, device=DEVICE)

# if __name__ == "__main__":
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main()


def main_train():
    """Trains the model and saves all necessary components to disk."""
    print(f"Loading data from {DATA_CSV}...")
    try:
        df = pd.read_csv(DATA_CSV)
        if df['label'].min() == 1:
            df['label'] = df['label'] - 1
    except FileNotFoundError:
        print(f"Error: '{DATA_CSV}' not found. Please update the DATA_CSV variable.")
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

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Run the training process
    main_train()


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
DATA_CSV = "traffic_dataset.csv"
X = 10
Y = 5
EPOCHS = 30
BATCH_SIZE = 128
NUM_CLASSES = 5
MODEL_SAVE_PATH = "saved_model" # Folder to save model files

TRAIN_START_TIME = 300
TRAIN_END_TIME = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEX_DIRECTIONS = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]

# (All helper functions and model class definitions from the previous script remain here)
# ... [build_hex_graph, compute_norm_stats, normalize_df, build_samples, build_snapshots] ...
# ... [LSTMEncoder, GNNEncoder, Decoder classes] ...
# ... [train_model function] ...

# NOTE: The full script with all necessary functions is provided at the end of this response.

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

# ==========================================================
# LSTM + GNN for multi-step zone prediction (modular)
# ==========================================================
import torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np, random
from collections import defaultdict
from torch_geometric.nn import GCNConv

# -----------------------------
# Config
# -----------------------------
data_csv = "traffic_dataset.csv"
X = 10      # number of past timesteps as input
Y = 5       # number of future timesteps to predict
epochs = 30
batch_size = 128
num_classes = 5
train_start_time = 400
train_end_time = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
HEX_DIRECTIONS = [(+1,0),(+1,-1),(0,-1),(-1,0),(-1,+1),(0,+1)]

# -----------------------------
# Build hex graph
# -----------------------------
def build_hex_graph(df):
    unique = sorted({tuple(map(int,h.split("_"))) for h in df["hex_id"].unique()})
    node_to_idx = {h:i for i,h in enumerate(unique)}
    edges=[]
    for (q,r) in unique:
        i = node_to_idx[(q,r)]
        for dq,dr in HEX_DIRECTIONS:
            nh = (q+dq,r+dr)
            if nh in node_to_idx:
                edges.append((i, node_to_idx[nh]))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return node_to_idx, edge_index, unique

# -----------------------------
# Normalize features
# -----------------------------
def compute_norm_stats(df, train_end):
    train = df[df["time"]<=train_end]
    stats = {}
    for col in ["num_vehicles","avg_speed","avg_sin","avg_cos"]:
        stats[col] = (train[col].min(), train[col].max())
    return stats

def normalize_df(df, stats):
    df = df.copy()
    for col in ["num_vehicles","avg_speed","avg_sin","avg_cos"]:
        mn,mx = stats[col]
        df[col] = (df[col]-mn)/(mx-mn+1e-9)
    return df

# -----------------------------
# Build training samples
# -----------------------------
def build_samples(df, node_to_idx, X, Y, min_time, max_time):
    samples = []
    for hex_id, grp in df.groupby("hex_id"):
        grp = grp.sort_values("time").reset_index(drop=True)
        feats = grp[["num_vehicles","avg_speed","avg_sin","avg_cos"]].values.astype(np.float32)
        labels = grp["label"].values
        times = grp["time"].values
        node_idx = node_to_idx.get(tuple(map(int,hex_id.split("_"))))
        if node_idx is None: continue
        for i in range(len(feats)-X-Y+1):
            t_last = int(times[i+X-1])
            # Filter samples based on min_time and max_time
            if t_last < min_time or t_last > max_time: continue

            seq = feats[i:i+X]
            target = labels[i+X:i+X+Y]
            if len(target)<Y: continue
            samples.append({
                "seq": torch.tensor(seq),
                "node_idx": node_idx,
                "t_last": t_last,
                "targets": torch.tensor(target-1)  # 0..num_classes-1
            })
    return samples

# -----------------------------
# Build snapshot tensors
# -----------------------------
def build_snapshots(df, node_list, node_to_idx):
    feat_cols = ["num_vehicles","avg_speed","avg_sin","avg_cos"]
    snaps = {}
    for t in sorted(df["time"].unique()):
        X_snap = np.zeros((len(node_list), len(feat_cols)), dtype=np.float32)
        sub = df[df["time"]==t]
        for _, r in sub.iterrows():
            q,r2 = map(int, r.hex_id.split("_"))
            idx = node_to_idx.get((q,r2))
            if idx is not None:
                X_snap[idx] = r[feat_cols].values
        snaps[t] = torch.tensor(X_snap, dtype=torch.float32)
    return snaps

# -----------------------------
# Model definitions
# -----------------------------
class LSTMEnc(nn.Module):
    def __init__(self,in_dim=4,hid=64,out=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim,hid,batch_first=True)
        self.fc = nn.Linear(hid,out)
        self.act = nn.ReLU()
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.act(self.fc(out[:,-1,:]))

class GNNEnc(nn.Module):
    def __init__(self,in_dim=4,h1=64,h2=32,out=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim,h1)
        self.conv2 = GCNConv(h1,h2)
        self.fc = nn.Linear(h2,out)
        self.act = nn.ReLU()
    def forward(self,x,edge_index):
        x = self.act(self.conv1(x,edge_index))
        x = self.act(self.conv2(x,edge_index))
        return self.act(self.fc(x))

class Decoder(nn.Module):
    def __init__(self,lstm_dim=32,gnn_dim=32,Y=5,hid=128,classes=5):
        super().__init__()
        self.fc1 = nn.Linear(lstm_dim+gnn_dim,hid)
        self.fc2 = nn.Linear(hid,Y*classes)
        self.act = nn.ReLU()
        self.Y = Y
        self.classes = classes
    def forward(self,l,g):
        x = torch.cat([l,g],1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), self.Y, self.classes)

# ==========================================================
# Training function
# ==========================================================
def train_model(samples, snaps, node_list, edge_index, lstm, gnn, dec,
                epochs, device, batch_size=128):
    lstm.to(device); gnn.to(device); dec.to(device)
    edge_index = edge_index.to(device)
    optimizer = optim.Adam(list(lstm.parameters())+list(gnn.parameters())+list(dec.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    ids = list(range(len(samples)))
    for ep in range(1,epochs+1):
        random.shuffle(ids)
        tot = 0; cnt = 0
        for b in range(0,len(ids), batch_size):
            idxs = ids[b:b+batch_size]
            batch = [samples[i] for i in idxs]
            seqs = torch.stack([s["seq"] for s in batch]).to(device)
            lstm_emb = lstm(seqs)

            # GNN embeddings per t_last
            tlasts = defaultdict(list)
            for s in batch: tlasts[s["t_last"]].append(s)
            g_cache = {}
            for t in tlasts:
                x = snaps[t].to(device)
                g_cache[t] = gnn(x, edge_index)

            g_nodes = []
            tgts = []
            for s in batch:
                g_nodes.append(g_cache[s["t_last"]][s["node_idx"]])
                tgts.append(s["targets"])
            g_nodes = torch.stack(g_nodes).to(device)
            tgts = torch.stack(tgts).to(device)

            logits = dec(lstm_emb, g_nodes)
            loss = sum(nn.CrossEntropyLoss()(logits[:,k,:], tgts[:,k]) for k in range(dec.Y))/dec.Y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()*seqs.size(0)
            cnt += seqs.size(0)
        print(f"Epoch {ep}/{epochs}  loss={tot/cnt:.4f}")
    return lstm, gnn, dec

# ==========================================================
# Prediction function
# ==========================================================
def predict_future(df, snaps, node_list, node_to_idx, edge_index,
                   lstm, gnn, dec, X, Y, device, stats):
    lstm.to(device); gnn.to(device); dec.to(device)
    edge_index = edge_index.to(device)
    lstm.eval(); gnn.eval(); dec.eval()

    seqs = []
    for node in node_list:
        hex_id = f"{node[0]}_{node[1]}"
        node_df = df[df["hex_id"]==hex_id].sort_values("time")
        last_seq = node_df.iloc[-X:][["num_vehicles","avg_speed","avg_sin","avg_cos"]].values.astype(np.float32)
        # Manually normalize the sequence for prediction
        for i,col in enumerate(["num_vehicles","avg_speed","avg_sin","avg_cos"]):
            mn,mx = stats[col]
            last_seq[:,i] = (last_seq[:,i]-mn)/(mx-mn+1e-9)
        seqs.append(torch.tensor(last_seq,dtype=torch.float32))

    seqs = torch.stack(seqs).to(device)
    with torch.no_grad():
        lstm_emb = lstm(seqs)
        Tmax = max(snaps.keys())
        xT = snaps[Tmax].to(device)
        g_out = gnn(xT, edge_index)
        logits = dec(lstm_emb, g_out)
        preds = torch.argmax(logits, 2).cpu().numpy() + 1  # 1..num_classes
    return Tmax, preds

# ==========================================================
# Main execution
# ==========================================================
def main():
    """Main function to run the data processing, training, and prediction pipeline."""
    # 1. Load data
    print(f"Loading data from {data_csv}...")
    try:
        df = pd.read_csv(data_csv)
    except FileNotFoundError:
        print(f"Error: {data_csv} not found. Please make sure the dataset is in the correct path.")
        return

    # 2. Build graph structure from hex coordinates
    print("Building hexagonal graph...")
    node_to_idx, edge_index, node_list = build_hex_graph(df)
    print(f"Graph built with {len(node_list)} nodes and {edge_index.shape[1]} edges.")

    # 3. Normalize features
    # Compute stats only on the training time window to avoid data leakage
    print(f"Normalizing features based on data up to time={train_end_time}...")
    stats = compute_norm_stats(df, train_end_time)
    df_norm = normalize_df(df, stats)

    # 4. Build training samples
    # This is where train_start_time and train_end_time are used to filter the samples
    print(f"Building training samples from time {train_start_time} to {train_end_time}...")
    train_samples = build_samples(df_norm, node_to_idx, X, Y, train_start_time, train_end_time)
    print(f"Created {len(train_samples)} training samples.")
    if not train_samples:
        print("No training samples were generated. Check your time ranges and data.")
        return

    # 5. Build snapshots for GNN
    print("Building temporal snapshots for GNN...")
    snaps = build_snapshots(df_norm, node_list, node_to_idx)

    # 6. Initialize models
    print("Initializing models (LSTM, GNN, Decoder)...")
    lstm = LSTMEnc()
    gnn = GNNEnc()
    dec = Decoder(Y=Y, classes=num_classes)

    # 7. Train the model
    print(f"Starting training for {epochs} epochs on device: {device}...")
    lstm, gnn, dec = train_model(
        train_samples, snaps, node_list, edge_index,
        lstm, gnn, dec,
        epochs=epochs, device=device, batch_size=batch_size
    )
    print("Training complete.")

    # 8. Predict future
    print("\nPredicting future traffic levels...")
    Tmax, preds = predict_future(
        df, snaps, node_list, node_to_idx, edge_index,
        lstm, gnn, dec, X, Y, device, stats
    )
    print(f"Prediction made based on final timestamp T={Tmax}.")
    print("Shape of predictions (Nodes, Future Timesteps):", preds.shape)
    
    # Create a DataFrame for better readability of predictions
    pred_df = pd.DataFrame(preds, 
                           index=[f"{q}_{r}" for q,r in node_list],
                           columns=[f"t+{t}" for t in range(1, Y + 1)])
    print("Example predictions for the first 5 nodes:")
    print(pred_df.head())


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    main()


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ================================================================
# 1. Helper functions
# ================================================================

def compute_neighbor_features(df, neighbor_map):
    """
    Adds average neighbor metrics (vehicles, speed) for each zone & timestep.
    neighbor_map: dict(hex_id -> list of neighbor hex_ids)
    """
    neighbor_vehicles, neighbor_speed = [], []
    for _, row in df.iterrows():
        neighbors = neighbor_map.get(row["hex_id"], [])
        t = row["time"]
        neigh_data = df[(df["time"] == t) & (df["hex_id"].isin(neighbors))]
        if not neigh_data.empty:
            neighbor_vehicles.append(neigh_data["num_vehicles"].mean())
            neighbor_speed.append(neigh_data["avg_speed"].mean())
        else:
            neighbor_vehicles.append(0)
            neighbor_speed.append(0)
    df["neighbor_avg_vehicles"] = neighbor_vehicles
    df["neighbor_avg_speed"] = neighbor_speed
    return df


def add_temporal_features(df, lag=1):
    """
    Adds lagged features for previous timesteps (t-1, t-2, ...).
    """
    df = df.sort_values(["hex_id", "time"])
    for l in range(1, lag + 1):
        df[f"num_vehicles_t-{l}"] = df.groupby("hex_id")["num_vehicles"].shift(l)
        df[f"avg_speed_t-{l}"] = df.groupby("hex_id")["avg_speed"].shift(l)
    return df


def preprocess_data(df, neighbor_map, lag=2):
    """
    Prepares the training DataFrame with all derived, temporal, and spatial features.
    """
    # Derived features
    df["std_speed"] = df.groupby("hex_id")["avg_speed"].transform("std").fillna(0)
    df["delta_vehicles"] = df.groupby("hex_id")["num_vehicles"].diff().fillna(0)
    df["truck_ratio"] = (df.get("num_LKW", 0) / df["num_vehicles"].replace(0, np.nan)).fillna(0)

    # Add spatial and temporal context
    df = compute_neighbor_features(df, neighbor_map)
    df = add_temporal_features(df, lag=lag)

    # Drop rows with NaN (first few timesteps after shift)
    df = df.dropna().reset_index(drop=True)
    return df


# ================================================================
# 2. Example usage
# ================================================================

# zone_stats = pd.read_csv("hex_aggregated_features.csv")  # Example input
# zone_stats should have: time, hex_id, num_vehicles, avg_speed, traffic_color, (optional truck counts)

# Example neighbor map (for 3 zones)
neighbor_map = {
    "hex_001": ["hex_002", "hex_003"],
    "hex_002": ["hex_001", "hex_003"],
    "hex_003": ["hex_001", "hex_002"],
}

# Prepare the dataset
lag = 2
df = preprocess_data(zone_stats, neighbor_map, lag=lag)

# Encode traffic color as target label
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["traffic_color"])

# Define feature columns (excluding ID/time/color)
feature_cols = [c for c in df.columns if c not in ["time", "hex_id", "traffic_color", "label"]]

# Split train/test
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

X_train, y_train = train_df[feature_cols], train_df["label"]
X_test, y_test = test_df[feature_cols], test_df["label"]

# ================================================================
# 3. Train Random Forest
# ================================================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# ================================================================
# 4. Evaluate
# ================================================================

y_pred = rf.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 Important Features:")
print(importances.head(10))

# Save feature importances
importances.to_csv("rf_feature_importances.csv", index=False)

# Save model
import joblib

joblib.dump(rf, "traffic_random_forest_model.pkl")
print("✅ Model saved to traffic_random_forest_model.pkl")

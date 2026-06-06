import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def plot_all_convergences():
    target_dir = "convergence"

    if not os.path.exists(target_dir):
        print("Convergence directory not found. Please run the simulation first.")
        return

    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))

    if not csv_files:
        print("No CSV files found in the convergence directory.")
        return

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if 'TimeStep' not in df.columns:
            print(f"Required columns are missing in {file_path}")
            continue

        # Create a figure with 3 subplots
        plt.figure(figsize=(10, 12))

        # 1. Packet Loss Plot
        plt.subplot(3, 1, 1)
        plt.plot(df['TimeStep'], df['PacketLossRatio'], label='Packet Loss Ratio (%)', color='red', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Ratio (%)')
        plt.title('Packet Loss Ratio over Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 2. Deadline Miss Plot
        plt.subplot(3, 1, 2)
        plt.plot(df['TimeStep'], df['DeadlineMissRatio'], label='Deadline Miss Ratio (%)', color='blue', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Ratio (%)')
        plt.title('Deadline Miss Ratio over Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 3. Reward Convergence Plot
        plt.subplot(3, 1, 3)

        # Calculate moving average for a smoother step reward curve
        smoothed_step_reward = df['StepAvgReward'].rolling(window=20, min_periods=1).mean()

        plt.plot(df['TimeStep'], smoothed_step_reward, label='Smoothed Step Reward (Window=20)', color='orange',
                 alpha=0.7, linestyle='--')
        plt.plot(df['TimeStep'], df['CumulativeAvgReward'], label='Cumulative Avg Reward', color='green', linewidth=2)

        plt.xlabel('Time Step')
        plt.ylabel('Reward Value')
        plt.title('Agent Reward Convergence')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()

        # Save the plot
        base_name = os.path.basename(file_path).replace('.csv', '.png')
        plot_path = os.path.join(target_dir, base_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Convergence plot saved successfully to {plot_path}")


if __name__ == "__main__":
    plot_all_convergences()
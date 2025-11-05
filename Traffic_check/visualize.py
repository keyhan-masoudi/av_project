import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import os

# --- Settings ---
INPUT_EXCEL_FILE = 'traffic_analysis_report.xlsx'
OUTPUT_IMAGE_FILE = 'traffic_density_map_final.pdf'
HEX_RADIUS = 200

# --- Font settings for the paper ---
# The invalid 'colorbar.labelsize' has been removed from this section.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})


def calculate_average_density(row):
    """
    Calculates a weighted average traffic density score for a partition.
    """
    counts = {
        'green': row['Green_Count'],
        'orange': row['Orange_Count'],
        'red': row['Red_Count'],
        'black': row['Black_Count']
    }
    total_timesteps = sum(counts.values())
    if total_timesteps == 0:
        return 1
    weighted_sum = (
            counts['green'] * 1 +
            counts['orange'] * 2 +
            counts['red'] * 3 +
            counts['black'] * 4
    )
    return weighted_sum / total_timesteps


def draw_map(data_df):
    """
    Draws and saves the hexagonal traffic map.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')

    patches = []
    colors = []

    data_df['avg_density'] = data_df.apply(calculate_average_density, axis=1)

    print("Generating hexagon patches for each partition...")
    for index, row in data_df.iterrows():
        center_x = row['center_x']
        center_y = row['center_y']
        density_score = row['avg_density']

        vertices = []
        for i in range(6):
            angle_deg = 60 * i + 30
            angle_rad = np.pi / 180 * angle_deg
            x = center_x + HEX_RADIUS * np.cos(angle_rad)
            y = center_y + HEX_RADIUS * np.sin(angle_rad)
            vertices.append((x, y))

        polygon = Polygon(vertices)
        patches.append(polygon)
        colors.append(density_score)

    p = PatchCollection(patches, alpha=1.0, edgecolor='black', linewidth=0.3)

    cmap = plt.get_cmap('YlOrRd')
    p.set_cmap(cmap)
    p.set_array(np.array(colors))

    ax.add_collection(p)

    cbar = fig.colorbar(p, ax=ax, aspect=30, shrink=0.7)

    # --- تغییرات درخواستی شما ---
    # 1. تنظیم اندازه فونت اعداد روی کالربار به 50
    # cbar.ax.tick_params(labelsize=40)
    cbar.set_ticks([])


    # تنظیم عنوان کالربار (با فونت متناسب)
    cbar.set_label('Average Traffic Density', fontsize=40, labelpad=20)

    ax.set_title('')
    ax.set_xlim(data_df['center_x'].min() - HEX_RADIUS, data_df['center_x'].max() + HEX_RADIUS)
    ax.set_ylim(data_df['center_y'].min() - HEX_RADIUS, data_df['center_y'].max() + HEX_RADIUS)

    # 2. مخفی کردن کامل محورهای x و y برای داشتن یک نقشه تمیز
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
    print(f"🎉 Map successfully saved to '{OUTPUT_IMAGE_FILE}'")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(INPUT_EXCEL_FILE):
        print(f"Error: The file '{INPUT_EXCEL_FILE}' was not found.")
    else:
        try:
            print(f"Reading data from '{INPUT_EXCEL_FILE}'...")
            traffic_data_df = pd.read_excel(INPUT_EXCEL_FILE)
            draw_map(traffic_data_df)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
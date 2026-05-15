import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# محدوده مستطیل
MIN_X, MAX_X = 290.61, 4210.7
MIN_Y, MAX_Y = 313.31, 3299.87

# خواندن فایل XML
xml_file = "hamburg.zon.xml"  # نام فایل ورودی XML


def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    zones = []

    for zone in root.findall("zone"):
        x = float(zone.get("x"))
        y = float(zone.get("y"))
        radius = float(zone.get("radius"))
        zones.append((x, y, radius))

    return zones


# رسم نقاط
def plot_zones(zones):
    fig, ax = plt.subplots(figsize=(8, 6))

    # رسم مستطیل اصلی
    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)


    # اضافه کردن نقاط به نمودار
    for x, y, radius in zones:
        circle = patches.Circle((x, y), radius, edgecolor='r', facecolor='none', linestyle='dashed', alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(x, y, color='blue', marker='o', label=f"({x}, {y})")

    plt.grid(False)
    plt.show()


# پردازش فایل XML و رسم نمودار
zones = parse_xml(xml_file)
plot_zones(zones)

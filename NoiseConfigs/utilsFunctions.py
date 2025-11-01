import math
import importlib
import pandas as pd
from collections import defaultdict

from NoiseConfigs.noiseConfigGeneralAttribute import NoiseConfigGeneralAttribute


def read_vehicle_positions(csv_file):
    df = pd.read_csv(csv_file)

    positions = list(df[['x', 'y']].itertuples(index=False, name=None))

    return positions

def yellow_bg(text):
    return f"\033[43m{text}\033[0m"

class UtilsFunc:
    FREQUENCY_GH = 5

    @staticmethod
    # Step 1: Add 'partitions' as an argument to the function
    def recognize_traffic_status(file_name, partitions):
        vehicles_positions = read_vehicle_positions(file_name)

        # Step 2: REMOVE the line below. Do not load partitions inside this function anymore.
        # partitions = UtilsFunc.load_partitions("generated_hex_partitions")

        if not partitions:
            print("No partitions provided to recognize_traffic_status.")
            return {}

        partition_traffic = defaultdict(int)

        for x, y in vehicles_positions:
            partition = UtilsFunc.find_partition(partitions, x, y)
            if partition:
                partition_traffic[partition] += 1

        traffic_statuses = {}
        for partition in partitions:
            count = partition_traffic.get(partition, 0)
            if count < 15:
                traffic_statuses[partition] = NoiseConfigGeneralAttribute.Traffic_options[3]
            elif count < 30:
                traffic_statuses[partition] = NoiseConfigGeneralAttribute.Traffic_options[2]
            elif count < 45:
                traffic_statuses[partition] = NoiseConfigGeneralAttribute.Traffic_options[1]
            else:
                traffic_statuses[partition] = NoiseConfigGeneralAttribute.Traffic_options[0]

        return traffic_statuses

    @staticmethod
    def get_hex_vertices_flat(cx, cy, r):
        return [
            (cx - r / 2, cy - (math.sqrt(3) * r) / 2),
            (cx + r / 2, cy - (math.sqrt(3) * r) / 2),
            (cx + r, cy),
            (cx + r / 2, cy + (math.sqrt(3) * r) / 2),
            (cx - r / 2, cy + (math.sqrt(3) * r) / 2),
            (cx - r, cy)
        ]

    @staticmethod
    def is_point_in_polygon(px, py, vertices):
        inside = False
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            intersect = ((yi > py) != (yj > py)) and \
                        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            if intersect:
                inside = not inside
        return inside

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def find_partition(partitions, x, y):
        for p in partitions:
            vertices = UtilsFunc.get_hex_vertices_flat(p.centerX, p.centerY, p.radius)
            if UtilsFunc.is_point_in_polygon(x, y, vertices):
                return p

        closest_partition = None
        closest_distance = float("inf")
        for p in partitions:
            dist = UtilsFunc.distance(x, y, p.centerX, p.centerY)
            if dist <= p.radius and dist < closest_distance:
                closest_partition = p
                closest_distance = dist

        return closest_partition

    @staticmethod
    def load_partitions(module_name):
        try:
            module = importlib.import_module(module_name)
            partitions = []
            for attr_name in dir(module):
                if attr_name.startswith("Partition") and attr_name[9:].isdigit():
                    partition_class = getattr(module, attr_name)
                    partitions.append(partition_class())
            return partitions
        except ImportError as e:
            print(f"Error importing module '{module_name}': {e}")
            return []

    @staticmethod
    def on_segment(px, py, qx, qy, rx, ry):
        """
        Checks if point (px, py) lies on segment (qx, qy) to (rx, ry).
        """
        return min(qx, rx) <= px <= max(qx, rx) and min(qy, ry) <= py <= max(qy, ry)

    @staticmethod
    def do_lines_intersect(p1, p2, q1, q2):
        """
        Checks if line segments p1-p2 and q1-q2 intersect.
        """
        # Find the orientation of the ordered triplets (p1, p2, q1) and (p1, p2, q2)
        o1 = UtilsFunc.orientation(p1, p2, q1)
        o2 = UtilsFunc.orientation(p1, p2, q2)
        o3 = UtilsFunc.orientation(q1, q2, p1)
        o4 = UtilsFunc.orientation(q1, q2, p2)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases
        if o1 == 0 and UtilsFunc.on_segment(q1[0], q1[1], p1[0], p1[1], p2[0], p2[1]):
            return True
        if o2 == 0 and UtilsFunc.on_segment(q2[0], q2[1], p1[0], p1[1], p2[0], p2[1]):
            return True
        if o3 == 0 and UtilsFunc.on_segment(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1]):
            return True
        if o4 == 0 and UtilsFunc.on_segment(p2[0], p2[1], q1[0], q1[1], q2[0], q2[1]):
            return True

        return False

    @staticmethod
    def orientation(p, q, r):
        """
        Find the orientation of the ordered triplet (p, q, r).
        0 -> p, q and r are collinear
        1 -> Clockwise
        2 -> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise

    @staticmethod
    def find_line_intersections(p1, p2, partitions):
        """
        Given two points p1 and p2, find all the partitions their connecting line intersects.
        and if p1 != p2, and they are in the same partition, then return their partition
        """
        # print(f"p1:{p1}, p2:{p2}")

        intersecting_partitions = []

        for partition in partitions:
            vertices = UtilsFunc.get_hex_vertices_flat(partition.centerX, partition.centerY, partition.radius)
            # Check if the line between p1 and p2 intersects any side of the hexagon
            for i in range(6):
                if UtilsFunc.do_lines_intersect(p1, p2, vertices[i], vertices[(i + 1) % 6]):
                    intersecting_partitions.append(partition)
                    break  # Once a partition is found, no need to check further sides

        if len(intersecting_partitions) == 0 and ((p1[0] != p2[0]) or (p1[1] != p2[1])):
            intersecting_partitions.append(UtilsFunc().find_partition(partitions, p1[0], p1[1]))
        return intersecting_partitions

    @staticmethod
    def get_max_urban_status(intersecting_partitions):
        if not intersecting_partitions:
            return None

        max_urban_status = float('-inf')

        # note: removed! it was added to add HRL, but this feature has been canceled
        # for partition in intersecting_partitions:
        #     if hasattr(partition, 'urbanStatus'):
        #         if partition.is_factory:
        #             factory_n_factor = 5.0
        #         else:
        #             factory_n_factor = 0.0
        #         max_urban_status = max(max_urban_status, partition.urbanStatus.random_noise_coff(), factory_n_factor)

        for partition in intersecting_partitions:
            if hasattr(partition, 'urbanStatus'):
                max_urban_status = max(max_urban_status, partition.urbanStatus.random_noise_coff())

        return max_urban_status

    @staticmethod
    def get_max_rain_attenuation(intersecting_partitions):
        if not intersecting_partitions:
            return None

        max_rain_attenuation = float('-inf')

        for partition in intersecting_partitions:
            if hasattr(partition, 'rainStatus'):
                max_rain_attenuation = max(max_rain_attenuation, partition.rainStatus.random_noise())

        return max_rain_attenuation

    @staticmethod
    def path_loss_km_ghz(d_km, f_ghz, n=2):
        """
        Calculate Path Loss in decibels (dB) using distance in km, frequency in GHz, and path loss exponent

        :param d_km: Distance between transmitter and receiver (kilometers)
        :param f_ghz: Frequency of transmitted wave (GHz)
        :param n: Path loss exponent (default: 2 for free space)
        :return: Path Loss value in dB
        """
        if d_km < 0 or f_ghz <= 0:
            raise ValueError(f"d_km and f_ghz must be positive values : {d_km}, {f_ghz}")
        elif d_km == 0:
            return 0
        c = 3e8
        pi = math.pi

        PL = 10 * n * math.log10(d_km) + 10 * n * math.log10(f_ghz) + 10 * n * math.log10(4 * pi / c) + 10 * n * (3+9)
        return PL

    @staticmethod
    def find_neighbors(all_partitions):
        """
        Finds all neighboring partitions for each partition in a hexagonal grid.
        Returns a dictionary where keys are partitions and values are lists of their neighbors.
        """
        neighbors_map = {p: [] for p in all_partitions}
        if not all_partitions:
            return {}

        # In a hex grid, distance between centers of neighbors is roughly 2 * radius.
        # We add a small tolerance (5%) to be safe.
        radius = all_partitions[0].radius
        neighbor_distance_threshold = 2 * radius * 1.05

        for i, p1 in enumerate(all_partitions):
            for j, p2 in enumerate(all_partitions):
                if i == j:
                    continue

                dist = UtilsFunc.distance(p1.centerX, p1.centerY, p2.centerX, p2.centerY)
                if dist < neighbor_distance_threshold:
                    neighbors_map[p1].append(p2)

        return neighbors_map

# Example usage
if __name__ == "__main__":
    x = UtilsFunc.distance(4214.90, 1932.26, 6000, 1500)
    print(x)
    print(UtilsFunc().path_loss_km_ghz(x/1000, 5, 0.2))
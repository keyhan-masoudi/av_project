import xml.etree.ElementTree as ET
import random


# Load SUMO vehicle XML
tree = ET.parse("test.xml")
root = tree.getroot()

# Prepare a dict to store angles per vehicle
vehicle_angles = {}

# Collect angles for each vehicle at each timestep
for timestep in root.findall("timestep"):
    time = int(timestep.attrib["time"])
    for vehicle in timestep.findall("vehicle"):
        vid = vehicle.attrib["id"]
        angle = float(vehicle.attrib["angle"])
        if vid not in vehicle_angles:
            vehicle_angles[vid] = []
        vehicle_angles[vid].append((time, angle))

# Create root for tasks XML
tasks_root = ET.Element("fcd-export", version="1.0")

# Generate rotation tasks
for vid, angles in vehicle_angles.items():
    i = 1  # start from second timestep to compare with previous
    while i < len(angles):
        prev_time, prev_angle = angles[i-1]
        curr_time, curr_angle = angles[i]
        if curr_angle != prev_angle:
            # Rotation task detected
            start_time = prev_time
            start_angle = prev_angle
            # Find end of rotation interval
            j = i
            while j < len(angles) - 1 and angles[j][1] != angles[j+1][1]:
                j += 1
            end_time, end_angle = angles[j]
            
            # Create timestep element in tasks XML
            timestep_elem = tasks_root.find(f".//timestep[@time='{start_time}']")
            if timestep_elem is None:
                timestep_elem = ET.SubElement(tasks_root, "timestep", time=str(start_time))
            
            # Create task element
            task_elem = ET.SubElement(timestep_elem, "task",
                                      id=f"{vid}_{start_time}",
                                      deadline=str(end_time),
                                      creator=vid,
                                      priority="crucial",
                                      start_angle=str(start_angle),
                                      end_angle=str(end_angle),
                                      dataSize="",  # function to fill later
                                      cycles_needed="")  # function to fill later
            
            # Move i to timestep after end of rotation
            i = j + 1
        else:
            i += 1

# Prepare a dict to store speeds per vehicle
vehicle_speeds = {}

# Collect speeds for each vehicle at each timestep
for timestep in root.findall("timestep"):
    time = int(timestep.attrib["time"])
    for vehicle in timestep.findall("vehicle"):
        vid = vehicle.attrib["id"]
        speed = float(vehicle.attrib["speed"])
        if vid not in vehicle_speeds:
            vehicle_speeds[vid] = []
        vehicle_speeds[vid].append((time, speed))

# Generate accelerate/brake tasks
for vid, speeds in vehicle_speeds.items():
    i = 1
    while i < len(speeds):
        prev_time, prev_speed = speeds[i-1]
        curr_time, curr_speed = speeds[i]
        if curr_speed != prev_speed:
            # Speed change detected
            start_time = curr_time
            start_speed = prev_speed
            # Find end of speed change interval
            j = i
            while j < len(speeds) - 1 and speeds[j][1] != speeds[j+1][1]:
                j += 1
            end_time, end_speed = speeds[j]

            # Determine random deadline offset
            if end_speed > start_speed:  # accelerate
                offset = random.uniform(3, 7)
            else:  # brake
                offset = random.uniform(2, 3)
            deadline = end_time + offset

            # Create timestep element in tasks XML
            timestep_elem = tasks_root.find(f".//timestep[@time='{start_time}']")
            if timestep_elem is None:
                timestep_elem = ET.SubElement(tasks_root, "timestep", time=str(start_time))

            # Create task element
            task_elem = ET.SubElement(timestep_elem, "task",
                                      id=f"{vid}_speed_{start_time}",
                                      deadline=str(round(deadline, 2)),
                                      creator=vid,
                                      priority="crucial",
                                      start_speed=str(start_speed),
                                      end_speed=str(end_speed),
                                      dataSize="",  # to fill later
                                      cycles_needed="")  # to fill later

            i = j + 1
        else:
            i += 1

# Save tasks XML
tasks_tree = ET.ElementTree(tasks_root)
tasks_tree.write("rotation_tasks.xml", encoding="utf-8", xml_declaration=True)
print("Rotation tasks XML generated!")

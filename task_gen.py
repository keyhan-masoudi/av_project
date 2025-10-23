import xml.etree.ElementTree as ET
import random

# Load SUMO vehicle XML
tree = ET.parse("chunk_0.xml")
root = tree.getroot()

# Prepare dicts to store angles and speeds per vehicle
vehicle_angles = {}
vehicle_speeds = {}

# Collect angles and speeds for each vehicle at each timestep
for timestep in root.findall("timestep"):
    time = int(timestep.attrib["time"])
    for vehicle in timestep.findall("vehicle"):
        vid = vehicle.attrib["id"]
        angle = float(vehicle.attrib["angle"])
        speed = float(vehicle.attrib["speed"])
        
        if vid not in vehicle_angles:
            vehicle_angles[vid] = []
        vehicle_angles[vid].append((time, angle))
        
        if vid not in vehicle_speeds:
            vehicle_speeds[vid] = []
        vehicle_speeds[vid].append((time, speed))

# Create root for tasks XML
tasks_root = ET.Element("fcd-export", version="1.0")

# --- Generate Rotation Tasks ---
for vid, angles in vehicle_angles.items():
    i = 1
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
            
            # Create timestep element
            timestep_elem = tasks_root.find(f".//timestep[@time='{start_time}']")
            if timestep_elem is None:
                timestep_elem = ET.SubElement(tasks_root, "timestep", time=str(start_time))
            
            # Create task element
            task_elem = ET.SubElement(timestep_elem, "task",
                                      id=f"{vid}_rotation_{start_time}",
                                      creator=vid,
                                      priority="crucial",
                                      task_type="rotation",
                                      start_time=str(start_time),
                                      deadline=str(end_time),
                                      start_angle=str(start_angle),
                                      end_angle=str(end_angle),
                                      dataSize="",  # to fill later
                                      cycles_needed="")  # to fill later
            
            i = j + 1
        else:
            i += 1

# --- Generate Accelerate/Brake Tasks ---
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
                task_name = "accelerate"
            else:  # brake
                offset = random.uniform(2, 3)
                task_name = "brake"
            deadline = end_time + offset

            # Create timestep element
            timestep_elem = tasks_root.find(f".//timestep[@time='{start_time}']")
            if timestep_elem is None:
                timestep_elem = ET.SubElement(tasks_root, "timestep", time=str(start_time))

            # Create task element
            task_elem = ET.SubElement(timestep_elem, "task",
                                      id=f"{vid}_{task_name}_{start_time}",
                                      creator=vid,
                                      priority="crucial",
                                      task_type=task_name,
                                      start_time=str(start_time),
                                      deadline=str(round(deadline, 2)),
                                      start_speed=str(start_speed),
                                      end_speed=str(end_speed),
                                      dataSize="",  # to fill later
                                      cycles_needed="")  # to fill later

            i = j + 1
        else:
            i += 1

# Save tasks XML
tasks_tree = ET.ElementTree(tasks_root)
tasks_tree.write("out.xml", encoding="utf-8", xml_declaration=True)
print("Tasks XML generated with rotation and accelerate/brake tasks!")

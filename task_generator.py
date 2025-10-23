import xml.etree.ElementTree as ET

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

# Save tasks XML
tasks_tree = ET.ElementTree(tasks_root)
tasks_tree.write("rotation_tasks.xml", encoding="utf-8", xml_declaration=True)
print("Rotation tasks XML generated!")

import xml.etree.ElementTree as ET

file_path = "../tasks/chunk_0.xml"

tree = ET.parse(file_path)
root = tree.getroot()

max_task_power = 0

for timestep in root.findall("timestep"):
    for vehicle in timestep.findall("task"):
        power = float(vehicle.attrib["power"])
        if power > max_task_power:
            max_task_power = power

print(f"max task power: {max_task_power}")

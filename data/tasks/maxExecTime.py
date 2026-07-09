import xml.etree.ElementTree as ET

file_path = "../tasks/chunk_0.xml"

tree = ET.parse(file_path)
root = tree.getroot()

max_exec_time = 0
min_exec_time = 110

for timestep in root.findall("timestep"):
    for vehicle in timestep.findall("task"):
        exec_time = float(vehicle.attrib["exec_time"])
        if exec_time > max_exec_time:
            max_exec_time = exec_time
        if exec_time < min_exec_time:
            min_exec_time = exec_time

print(f"max exec time: {max_exec_time} s")
print(f"min exec time: {min_exec_time} s")
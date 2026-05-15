import xml.etree.ElementTree as ET

file_path = "chunk_0.xml"

tree = ET.parse(file_path)
root = tree.getroot()

max_speed = 0

for timestep in root.findall("timestep"):
    for vehicle in timestep.findall("vehicle"):
        speed = float(vehicle.attrib["power"])
        if speed > max_speed:
            max_speed = speed

print(f"max Speed: {max_speed} m/s")

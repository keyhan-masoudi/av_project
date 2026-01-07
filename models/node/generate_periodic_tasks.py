import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

PERIODIC_TASKS = [
    (7.0,  (800.0, 1200.0), (1000.0, 1200.0)),
    (5.0,  (1000.0, 5000.0), (1000.0, 1200.0)),
    (6.0,  (500.0, 1000.0),  (500.0, 1000.0)),
]

SIM_TIME = 1200.0
FREQUENCY = 2e9
VEHICLE_ID = "LKW0"

root = ET.Element("fcd-export", version="1.0")

task_counter = 0
current_time = 0.0
next_release = {p: 0.0 for p, _, _ in PERIODIC_TASKS}

while current_time <= SIM_TIME:
    timestep = ET.SubElement(root, "timestep", time=f"{current_time:.2f}")

    for period, data_range, cycles_range in PERIODIC_TASKS:
        if current_time >= next_release[period] - 1e-9:
            data_kb = random.uniform(*data_range)
            cycles_per_bit = random.uniform(*cycles_range)
            bits = data_kb * 1024
            exec_time = (bits * cycles_per_bit) / FREQUENCY
            deadline = current_time + period

            ET.SubElement(
                timestep,
                "task",
                id=f"{VEHICLE_ID}_{task_counter}",
                deadline=f"{deadline:.2f}",
                exec_time=f"{exec_time:.2f}",
                power="0", # we do not have power
                creator=VEHICLE_ID, # just as an example because every car has these crucial tasks
                cycles_per_bit=f"{cycles_per_bit:.2f}",
                dataSize=f"{data_kb:.2f}"
            )

            task_counter += 1
            next_release[period] += period

    current_time += 1.0

xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")

with open("periodic_tasks.xml", "w") as f:
    f.write(xml_str)

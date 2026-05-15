import xml.etree.ElementTree as ET

def decrease_power_by_one(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for vehicle in root.iter('vehicle'):
        power = float(vehicle.attrib['power'])
        vehicle.set('power', str(round(power - 1, 2)))

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


def increase_power_by_one(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for vehicle in root.iter('vehicle'):
        power = float(vehicle.attrib['power'])
        vehicle.set('power', str(round(power + 1, 2)))

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    decrease_power_by_one('chunk_0.xml')
    # increase_power_by_one('chunk_0.xml')

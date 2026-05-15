import xml.etree.ElementTree as ET


def add_data_size_to_tasks(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for timestep in root.findall('timestep'):
        for task in timestep.findall('task'):
            exec_time = float(task.get('exec_time'))

            data_size = round(320 + (exec_time / 25) * 320, 2)
            # data_size = 32 + (exec_time / 25) * 32

            task.set('dataSize', str(data_size))

    tree.write(xml_file)

def remove_data_size_from_tasks(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for timestep in root.findall('timestep'):
        for task in timestep.findall('task'):
            if 'dataSize' in task.attrib:
                del task.attrib['dataSize']

    tree.write(xml_file)


if __name__ == '__main__':
    add_data_size_to_tasks('chunk_0.xml')
    # remove_data_size_from_tasks('chunk_0.xml')
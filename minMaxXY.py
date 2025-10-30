import xml.etree.ElementTree as ET


def find_min_max_coordinates(file_name):
    # Parse the XML file
    tree = ET.parse(file_name)
    root = tree.getroot()

    # Initialize min and max values with extreme values
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    # Iterate over each timestep and vehicle in the XML
    for timestep in root.findall('timestep'):
        for vehicle in timestep.findall('vehicle'):
            # Get the x and y coordinates of the vehicle
            x = float(vehicle.get('x'))
            y = float(vehicle.get('y'))

            # Update min and max values for x and y
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    # Print the results
    print(f"Minimum x: {min_x}")
    print(f"Maximum x: {max_x}")
    print(f"Minimum y: {min_y}")
    print(f"Maximum y: {max_y}")


# Example usage
if __name__ == "__main__":
    find_min_max_coordinates("simulation.out.xml")

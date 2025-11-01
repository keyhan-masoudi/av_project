import numpy as np


def calculate_P_LoS(xi, yi, xm, ym, a, b, h):
    """
    Calculate P_LoS based on the given formula.

    Parameters:
    xi, yi: Coordinates of point i
    xm, ym: Coordinates of point m
    a, b: Parameters for the formula
    h: Height value

    Returns:
    P_LoS: The calculated value for P_LoS
    """
    # Compute the Euclidean distance between points i and m
    distance = np.sqrt((xm - xi) ** 2 + (ym - yi) ** 2)

    # Compute the arctangent of h / distance
    arctan_value = np.arctan(h / distance)

    # Compute the final value of P_LoS using the given formula
    P_LoS = 1 / (1 + a * np.exp(-b * arctan_value - a))

    return P_LoS


# Example of using the function
# xi, yi = 0, 0  # Coordinates of point i
# xm, ym = 60, 40  # Coordinates of point m
# a = 3.8  # Example value for a
# b = 2.0  # Example value for b
# h = 0  # Example value for h
#
# P_LoS = calculate_P_LoS(xi, yi, xm, ym, a, b, h)
# print("P_LoS:", P_LoS)


import sys
print(sys.executable)
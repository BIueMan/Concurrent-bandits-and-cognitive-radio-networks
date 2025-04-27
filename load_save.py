import numpy as np
import os
from datetime import datetime

def save_matrix_with_mu(matrix, mu_array, name, dest="output"):
    """
    Saves a uint8 matrix to a compact CSV format with one row per user.
    
    Each row contains:
        mu_value, hex_encoded_data

    Args:
        matrix (np.ndarray): 2D matrix of shape (X, Y) with values 0–15.
        mu_array (np.ndarray): 1D array of length Y (user mus).
        name (str): Base filename.
        dest (str): Destination folder.
    """
    if not os.path.exists(dest):
        os.makedirs(dest)

    timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    filename = f"{name}_{timestamp}.csv"
    filepath = os.path.join(dest, filename)

    # Sanity checks
    X, Y = matrix.shape
    if len(mu_array) != Y:
        raise ValueError("Length of mu_array must match number of columns in matrix (users).")

    with open(filepath, "w") as f:
        for user_idx in range(Y):
            user_data = matrix[:, user_idx]
            hex_string = ''.join(format(val, 'X') for val in user_data)
            f.write(f"{mu_array[user_idx]},{hex_string}\n")

    print(f"Saved to {filepath}")
    
def load_matrix_with_mu(filepath):
    """
    Loads a compact CSV file saved by `save_matrix_with_mu`.

    Returns:
        matrix (np.ndarray): uint8 matrix of shape (X, Y).
        mu_array (np.ndarray): 1D array of float mu values.
    """
    mu_list = []
    data_list = []

    with open(filepath, "r") as f:
        for line in f:
            mu_str, hex_str = line.strip().split(",")
            mu_list.append(float(mu_str))
            row_data = [int(c, 16) for c in hex_str]
            data_list.append(row_data)

    matrix = np.array(data_list, dtype=np.uint8).T  # Transpose to (X, Y)
    mu_array = np.array(mu_list, dtype=np.float32)
    return matrix, mu_array
import json
import pickle
import numpy as np
import os

# Global variables
__locations = None
__data_columns = None
__model = None

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))  # Create a feature array

    # Assign correct feature values
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # One-hot encode location
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)  # Predict price


def get_location_names():
    global __locations
    if __locations is None:  # Ensure artifacts are loaded
        load_saved_artifacts()
    return __locations


def load_saved_artifacts():
    print("Loading saved artifacts... start")
    global __data_columns
    global __locations
    global __model

    # Use absolute paths
    columns_path = os.path.join(BASE_DIR, "artifacts", "columns.json")
    model_path = os.path.join(BASE_DIR, "artifacts", "bangalore_home_prices_model.pickle")

    # Load column names
    with open(columns_path, 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    # Load trained model
    with open(model_path, 'rb') as f:
        __model = pickle.load(f)

    print("Loading saved artifacts... done")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())

    # Sample price estimations
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalli', 1000, 2, 2))
    print(get_estimated_price('Tripura', 1000, 2, 2))

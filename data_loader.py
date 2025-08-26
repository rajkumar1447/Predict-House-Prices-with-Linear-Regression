import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data():
    """Load and return California housing dataset as DataFrame."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df

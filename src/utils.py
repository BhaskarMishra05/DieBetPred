import os
import sys
import pandas as pd
import numpy as np
import pickle
import joblib

def save_object(path: str, object):
    os.makedirs(os.path.dirname(path), exist_ok= True)
    with open (path, 'wb') as f:
        return pickle.dump(object, f)
    
def load_object(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    with open (path, 'rb') as f:
        return pickle.load(f)
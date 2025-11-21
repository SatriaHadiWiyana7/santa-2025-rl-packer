import json
import os
import numpy as np

def save_to_processed(data, filename):
    """Simpan data (dictionary/list) ke folder data/processed sebagai JSON"""
    os.makedirs("data/processed", exist_ok=True)
    path = os.path.join("data/processed", filename)
    
    # Convert numpy types to python types for JSON serialization
    def convert(o):
        if isinstance(o, np.float32) or isinstance(o, np.float64): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError
        
    with open(path, 'w') as f:
        json.dump(data, f, default=convert, indent=4)
    print(f"Data tersimpan di {path}")

def load_from_processed(filename):
    path = os.path.join("data/processed", filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}
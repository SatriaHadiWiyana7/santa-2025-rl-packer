import os

# --- PATHS ---
# Lokasi folder root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# --- HYPERPARAMETERS (Pengaturan AI) ---
RL_ALGORITHM = "PPO"
TOTAL_TIMESTEPS = 1000000  # Naikkan
LEARNING_RATE = 0.0003
N_STEPS = 2048           # Buffer size sebelum update
BATCH_SIZE = 64
GAMMA = 0.99             # Diskon reward masa depan

# --- GEOMETRY CONSTANTS ---
# Batas dunia simulasi
WORLD_LIMIT = 25.0
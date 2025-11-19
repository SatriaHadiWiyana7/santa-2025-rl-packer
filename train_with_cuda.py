import sys
import os
import time
import numpy as np
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env # Utilitas untuk multiple env
from stable_baselines3.common.vec_env import SubprocVecEnv  # Menggunakan multiple CPU cores

from tqdm import tqdm 
# Import modul local
from src.env import ChristmasTreeEnv
from src.optimizer import squeeze_solution
from src.utils import save_to_processed, load_from_processed 
from src.config import (
    MODELS_DIR, 
    TOTAL_TIMESTEPS, 
    RL_ALGORITHM, 
    LEARNING_RATE
)
from src.agent import SaveOnBestTrainingRewardCallback


# Set jumlah core CPU yang akan digunakan untuk training paralel
N_CPUS = 8 


def train_and_solve(n_trees_target):
    """
    Fungsi utama untuk menyelesaikan 1 puzzle.
    """
    
    print(f"\n{'='*60}")
    print(f"MEMULAI MISI UNTUK JUMLAH POHON: {n_trees_target}")
    print(f"CPU Parallel: {N_CPUS} cores")
    print(f"{'='*60}")

    # --- SETUP ENVIRONMENT & MODEL ---
    
    # Menggunakan Vectorized Environment
    vec_env = make_vec_env(
        ChristmasTreeEnv, 
        n_envs=N_CPUS,
        env_kwargs={'n_trees': n_trees_target},
        vec_env_cls=SubprocVecEnv
    ) 
    
    # Nama file model
    model_name = f"{RL_ALGORITHM}_tree_{n_trees_target:03d}"
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Cek apakah kita punya model yang sudah dilatih sebelumnya?
    if os.path.exists(model_path + ".zip"):
        print(f"Model ditemukan: {model_name}.zip")
        print("   Memuat model untuk melanjutkan/memprediksi...")
        model = PPO.load(model_path, env=vec_env)
    else:
        print(f"Model tidak ditemukan. Membuat model baru...")
        
        # Menggunakan GPU/CUDA secara otomatis jika tersedia
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=0, 
            learning_rate=LEARNING_RATE,
            device='auto' # Ini akan mencari 'cuda' (GPU) jika tidak ada, fallback ke 'cpu'
        )
        
        # --- TRAINING LOOP ---
        print(f"Mulai Training selama {TOTAL_TIMESTEPS} langkah...")
        
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=MODELS_DIR)
        
        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        end_time = time.time()
        
        print(f"Training selesai dalam {(end_time - start_time):.2f} detik.")
        model.save(model_path) 

    # --- PREDIKSI KASAR (RL INFERENCE) ---
    print("AI sedang mencoba menyusun posisi awal...")
    
    # Saat prediksi, kita perlu env standar (unvectorized) untuk mendapatkan state akhir
    single_env = ChristmasTreeEnv(n_trees=n_trees_target)
    obs, _ = single_env.reset()
    
    # Biarkan AI memperbaiki posisi selama beberapa step
    for _ in range(50): 
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = single_env.step(action)
    
    ai_state = obs
    initial_score = info.get('score', 0) 
    print(f"Skor Awal AI (Area): {initial_score:.4f}")

    # --- OPTIMASI HALUS (THE SQUEEZER) ---
    print("Menjalankan 'The Squeezer' (SciPy Optimize)...")
    
    final_side, final_coords = squeeze_solution(ai_state, n_trees_target)
    
    final_score = final_side ** 2
    if initial_score > 0:
        improvement = ((initial_score - final_score) / initial_score) * 100
    else:
        improvement = 0.0
    
    print(f"OPTIMASI SELESAI!")
    print(f"      Skor Akhir (Area): {final_score:.4f}")
    print(f"      Sisi Kotak (Side): {final_side:.4f}")
    print(f"      Peningkatan: {improvement:.2f}% lebih padat.")

    # --- FORMAT HASIL UNTUK DISIMPAN ---
    solution_list = []
    for i in range(n_trees_target):
        idx = i * 3
        solution_list.append({
            "id": f"{n_trees_target:03d}_{i}", 
            "x": final_coords[idx],
            "y": final_coords[idx+1],
            "deg": final_coords[idx+2]
        })
        
    return solution_list

if __name__ == "__main__":
    # --- KONFIGURASI TARGET PUZZLE ---
    TARGET_PUZZLES = range(1, 51) # 1 - 200
    
    filename_json = "final_solutions_checkpoint.json"
    
    print("PROGRAM STARTED")
    
    # --- MUAT DATA LAMA AGAR TIDAK TERTIPA ---
    filename_json = "final_solutions_checkpoint.json"
    try:
        loaded_data = load_from_processed(filename_json)
        if isinstance(loaded_data, list):
            all_solutions = loaded_data
            print(f"Berhasil memuat {len(all_solutions)} solusi pohon yang sudah ada sebelumnya.")
        else:
            all_solutions = []
    except Exception:
        all_solutions = []
    
    # Loop untuk setiap puzzle
    for n in tqdm(TARGET_PUZZLES, desc="Total Progress"):
        
        # Cek apakah puzzle ini sudah ada di daftar solusi
        already_done = any(item['id'].startswith(f"{n:03d}_") for item in all_solutions)
        if already_done:
            print(f"Skip N={n} (Sudah ada di database)")
            continue

        try:
            sol = train_and_solve(n)
            all_solutions.extend(sol)
            
            # Simpan progress setiap kali satu puzzle selesai
            save_to_processed(all_solutions, filename_json)
            
        except Exception as e:
            print(f"Error fatal pada N={n}: {e}")
            import traceback
            traceback.print_exc()
            break
            
    print("\nSEMUA TARGET SELESAI! Jalankan 'python submit.py' untuk mengumpulkan hasil.")
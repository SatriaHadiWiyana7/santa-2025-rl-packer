import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
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

def train_and_solve(n_trees_target):
    """
    Fungsi utama untuk menyelesaikan 1 puzzle.
    Proses: RL Training -> Inference Kasar -> Optimasi Halus (Squeezer).
    """
    
    print(f"\n{'='*60}")
    print(f"MEMULAI MISI UNTUK JUMLAH POHON: {n_trees_target}")
    print(f"{'='*60}")

    # --- SETUP ENVIRONMENT & MODEL ---
    env = ChristmasTreeEnv(n_trees=n_trees_target)
    
    # Bungkus Env dengan Monitor agar data reward terekam untuk callback
    env = Monitor(env) 
    
    # Nama file model
    model_name = f"{RL_ALGORITHM}_tree_{n_trees_target:03d}"
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Cek apakah kita punya model yang sudah dilatih sebelumnya?
    if os.path.exists(model_path + ".zip"):
        print(f"Model ditemukan: {model_name}.zip")
        print("   Memuat model untuk melanjutkan/memprediksi...")
        model = PPO.load(model_path, env=env)
    else:
        print(f"Model tidak ditemukan. Membuat model baru...")
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=LEARNING_RATE)
        
        # --- TRAINING LOOP ---
        print(f"Mulai Training selama {TOTAL_TIMESTEPS} langkah...")
        
        # Setup Callback (akan menyimpan model terbaik saat training)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=MODELS_DIR)
        
        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        end_time = time.time()
        
        print(f"Training selesai dalam {(end_time - start_time):.2f} detik.")
        model.save(model_path) # Simpan versi terakhir

    # --- 3. PREDIKSI KASAR (RL INFERENCE) ---
    print("AI sedang mencoba menyusun posisi awal...")
    obs, _ = env.reset()
    
    # Biarkan AI memperbaiki posisi selama beberapa step
    for _ in range(50): 
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = env.step(action)
    
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
        
        # Cek apakah puzzle ini sudah ada di daftar solusi?
        already_done = any(item['id'].startswith(f"{n:03d}_") for item in all_solutions)
        if already_done:
            # Jika sudah ada, lewati dan lanjut ke puzzle berikutnya
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
            break # Berhenti jika ada error agar tidak boros waktu
            
    print("\nSEMUA TARGET SELESAI! Jalankan 'python submit.py' untuk mengumpulkan hasil.")
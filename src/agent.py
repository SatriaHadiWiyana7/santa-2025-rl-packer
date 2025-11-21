from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback untuk menyimpan model setiap kali mencapai reward rata-rata terbaik baru.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Buat folder jika belum ada
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            
            # Cek apakah ada data episode yang tersimpan
            if len(self.model.ep_info_buffer) > 0:
                # Ambil rata-rata reward dari 100 episode terakhir
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

                # Jika reward lebih baik dari rekor sebelumnya, simpan!
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print(f"Simpan model terbaik baru di {self.save_path}")
                        print(f"      Reward: {mean_reward:.2f} (Sebelumnya: {self.best_mean_reward:.2f})")
                    
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)

        return True
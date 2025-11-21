import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .geometry import create_polys_from_state

class ChristmasTreeEnv(gym.Env):
    def __init__(self, n_trees=5):
        super(ChristmasTreeEnv, self).__init__()
        self.n_trees = n_trees
        
        # Batas koordinat area kerja
        self.limit = 20.0
        
        # ACTION SPACE: Perubahan posisi [dx, dy, d_theta] untuk setiap pohon
        # Nilai continuous antara -1 sampai 1
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(n_trees * 3,), 
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: Posisi absolut [x, y, theta]
        self.observation_space = spaces.Box(
            low=-self.limit, high=self.limit,
            shape=(n_trees * 3,), 
            dtype=np.float32
        )
        
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Inisialisasi posisi acak tapi rapat di tengah
        self.state = np.random.uniform(-2, 2, size=(self.n_trees * 3)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # Skala gerakan
        move_scale = 0.2   # Maks geser 0.2 unit
        rot_scale = 5.0    # Maks putar 5 derajat
        
        # Terapkan aksi
        scale_vec = np.tile([move_scale, move_scale, rot_scale], self.n_trees)
        self.state += action * scale_vec
        
        # Clip agar tidak keluar batas dunia
        self.state = np.clip(self.state, -self.limit, self.limit)
        
        # --- HITUNG REWARD ---
        polys = create_polys_from_state(self.state, self.n_trees)
        
        # Bounding Box (Area)
        min_x = min(p.bounds[0] for p in polys)
        min_y = min(p.bounds[1] for p in polys)
        max_x = max(p.bounds[2] for p in polys)
        max_y = max(p.bounds[3] for p in polys)
        
        width = max_x - min_x
        height = max_y - min_y
        side = max(width, height)
        area_score = side ** 2
        
        # Overlap (Hukuman Tabrakan)
        overlap_area = 0.0
        # Cek setiap pasangan pohon
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].intersects(polys[j]):
                    inter = polys[i].intersection(polys[j]).area
                    overlap_area += inter

        # Rumus Reward:
        penalty_weight = 10000.0  # Bobot penalti overlap 
        
        reward = -area_score - (overlap_area * penalty_weight)
        
        terminated = False
        truncated = False
        
        info = {
            "side": side,
            "score": area_score,
            "overlap": overlap_area
        }
        
        return self.state, reward, terminated, truncated, info
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
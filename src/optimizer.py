import numpy as np
from scipy.optimize import minimize
from shapely import affinity
import math
from .geometry import get_tree_polygon

# Perlu fungsi helper untuk mengubah state ke polygon di optimizer
def create_polys_from_state_local(flat_state, n_trees, base_poly):
    """Mengubah state array 1D menjadi list of Polygons untuk Shapely."""
    polys = []
    # flat_state format: [x1, y1, deg1, x2, y2, deg2, ...]
    for i in range(n_trees):
        idx = i * 3
        x = flat_state[idx]
        y = flat_state[idx+1]
        deg = flat_state[idx+2]
        
        p = affinity.rotate(base_poly, deg, origin=(0,0))
        p = affinity.translate(p, x, y)
        polys.append(p)
    return polys

def objective_function(flat_params, n_trees, base_poly):
    """
    Fungsi Tujuan Baru: MINIMIZE (Calculated Area + OVERLAP PENALTY)
    
    flat_params sekarang hanya berisi koordinat dan rotasi [x1, y1, d1, x2, y2, d2, ...]
    """
    
    # --- Hitung Polygons dan Overlap ---
    polys = create_polys_from_state_local(flat_params, n_trees, base_poly)
    
    overlap_area = 0.0
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            if polys[i].intersects(polys[j]):
                overlap_area += polys[i].intersection(polys[j]).area
    
    # --- Hitung Bounding Box  ---
    min_x = min(p.bounds[0] for p in polys)
    min_y = min(p.bounds[1] for p in polys)
    max_x = max(p.bounds[2] for p in polys)
    max_y = max(p.bounds[3] for p in polys)
    
    side = max(max_x - min_x, max_y - min_y)
    calculated_area = side ** 2
    
    # --- Hitung Total (Cost) ---
    # Penalti harus SANGAT BESAR agar optimizer takut overlap
    PENALTY_WEIGHT = 1_000_000.0 
    
    # Tujuan kita adalah meminimalkan area yang Dihitung, ditambah penalti
    cost = calculated_area + (overlap_area * PENALTY_WEIGHT)
    
    return cost

def squeeze_solution(initial_state, n_trees):
    """
    Menggunakan algoritma matematik untuk memadatkan posisi.
    """
    base_poly = get_tree_polygon()
    
    # initial_state sekarang langsung digunakan sebagai tebakan awal
    initial_guess = initial_state
    
    print(f"   Running Squeezer Optimizer for N={n_trees}...")
    
    # Jalankan Optimizer
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(n_trees, base_poly),
        method='Nelder-Mead', # meminimalisasi geometri
        options={'maxiter': 5000, 'disp': False} # Menaikkan maxiter agar lebih teliti
    )
    
    # --- HITUNG METRIK HASIL AKHIR ---
    final_coords = result.x
    polys = create_polys_from_state_local(final_coords, n_trees, base_poly)
    
    min_x = min(p.bounds[0] for p in polys)
    max_x = max(p.bounds[2] for p in polys)
    min_y = min(p.bounds[1] for p in polys)
    max_y = max(p.bounds[3] for p in polys)
    
    final_side = max(max_x - min_x, max_y - min_y)
    
    return final_side, final_coords
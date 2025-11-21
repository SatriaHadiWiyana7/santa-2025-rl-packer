from shapely.geometry import Polygon
from shapely import affinity
import numpy as np

def get_tree_polygon():
    """
    Mendefinisikan bentuk poligon pohon standar sesuai spesifikasi kompetisi.
    """
    # Dimensi pohon
    trunk_w, trunk_h = 0.15, 0.2
    base_w, mid_w, top_w = 0.7, 0.4, 0.25
    tip_y, tier_1_y, tier_2_y, base_y = 0.8, 0.5, 0.25, 0.0
    trunk_bottom_y = -trunk_h

    # Koordinat titik-titik poligon
    coords = [
        (0.0, tip_y), 
        (top_w/2, tier_1_y), (top_w/4, tier_1_y),
        (mid_w/2, tier_2_y), (mid_w/4, tier_2_y), 
        (base_w/2, base_y),
        (trunk_w/2, base_y), (trunk_w/2, trunk_bottom_y),
        (-trunk_w/2, trunk_bottom_y), (-trunk_w/2, base_y),
        (-base_w/2, base_y), 
        (-mid_w/4, tier_2_y), (-mid_w/2, tier_2_y),
        (-top_w/4, tier_1_y), (-top_w/2, tier_1_y)
    ]
    return Polygon(coords)

def create_polys_from_state(state, n_trees):
    """
    Mengubah array 1D (flat) menjadi list objek Polygon Shapely.
    State format: [x1, y1, deg1, x2, y2, deg2, ...]
    """
    base_poly = get_tree_polygon()
    polys = []
    for i in range(n_trees):
        idx = i * 3
        x = state[idx]
        y = state[idx+1]
        deg = state[idx+2]
        
        # Rotasi dulu, baru translasi (geser)
        p = affinity.rotate(base_poly, deg, origin=(0,0))
        p = affinity.translate(p, x, y)
        polys.append(p)
    return polys
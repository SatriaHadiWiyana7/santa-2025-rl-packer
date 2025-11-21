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
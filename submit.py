import pandas as pd
import numpy as np
from src.utils import load_from_processed
from src.geometry import get_tree_polygon
from shapely import affinity

def validate_overlaps(df):
    """Validasi akhir sebelum submit."""
    print("Memvalidasi Overlap...")
    base_tree = get_tree_polygon()
    
    # Group by puzzle ID (001, 002...)
    df['puzzle_id'] = df['id'].apply(lambda x: x.split('_')[0])
    valid = True
    
    for pid in df['puzzle_id'].unique():
        subset = df[df['puzzle_id'] == pid]
        polys = []
        for _, row in subset.iterrows():
            # Remove 's'
            x = float(row['x'].replace('s',''))
            y = float(row['y'].replace('s',''))
            d = float(row['deg'].replace('s',''))
            p = affinity.rotate(base_tree, d, origin=(0,0))
            p = affinity.translate(p, x, y)
            polys.append(p)
            
        for i in range(len(polys)):
            for j in range(i+1, len(polys)):
                if polys[i].intersects(polys[j]):
                    area = polys[i].intersection(polys[j]).area
                    if area > 1e-5: # Toleransi kecil
                        print(f"Overlap di Puzzle {pid} (Area: {area:.6f})")
                        valid = False
    return valid

def main():
    # 1. Load Solusi Kita
    my_solutions = load_from_processed("final_solutions_checkpoint.json")
    if not my_solutions:
        print(" Tidak ada data solusi ditemukan di data/processed!")
        return

    print(f"Memuat {len(my_solutions)} posisi pohon dari hasil training...")
    my_df = pd.DataFrame(my_solutions)
    
    # Load Template Kaggle
    # Pastikan Anda sudah menaruh sample_submission.csv di data/raw/
    try:
        sample_df = pd.read_csv("data/raw/sample_submission.csv")
    except FileNotFoundError:
        print("File data/raw/sample_submission.csv tidak ditemukan!")
        return

    # Gabungkan (Update)
    # Set index ke ID untuk memudahkan update
    sample_df.set_index('id', inplace=True)
    my_df.set_index('id', inplace=True)
    
    # Update hanya baris yang kita punya solusinya
    sample_df.update(my_df)
    sample_df.reset_index(inplace=True)
    
    # Formatting 's' (Wajib Kaggle)
    print("Memformat angka dengan prefix 's'...")
    final_csv = sample_df.copy()
    cols = ['x', 'y', 'deg']
    
    for col in cols:
        # Bersihkan 's' lama jika ada, convert float, format string, tambah 's' baru
        final_csv[col] = final_csv[col].astype(str).str.replace('s', '').astype(float)
        final_csv[col] = final_csv[col].apply(lambda x: f"s{x:.6f}")
    
    # Simpan
    output_file = "submission.csv"
    final_csv.to_csv(output_file, index=False)
    print(f"File siap: {output_file}")

if __name__ == "__main__":
    main()
import os
import pandas as pd
from relbench.datasets import get_dataset

# Cấu hình đường dẫn khớp với script import bash của bạn
OUTPUT_DIR = "./data/database"
DATASET_NAME = "rel-f1"  # Ví dụ: rel-stack, rel-amazon, rel-arxiv

def export_to_csv(dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading and processing {dataset_name}...")
    
    # Tải dataset (sẽ cache tự động)
    dataset = get_dataset(name=dataset_name, download=True)
    db = dataset.make_db()
    
    print(f"Exporting tables to {output_dir}...")
    for table_name, table in db.table_dict.items():
        df = table.df
        
        # Lưu ý: RelBench có thể chứa các cột object phức tạp, cần convert về string nếu cần
        # Postgres COPY yêu cầu format CSV chuẩn
        file_path = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(file_path, index=False, header=True)
        print(f"-> Saved {table_name}.csv ({len(df)} rows)")

if __name__ == "__main__":
    export_to_csv(DATASET_NAME, OUTPUT_DIR)
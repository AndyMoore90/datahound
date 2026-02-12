import argparse
import os
import glob
import pandas as pd

def convert_parquet_to_csv(file_path):
    # Check if the file is a Parquet file
    if not file_path.lower().endswith('.parquet'):
        print(f"Skipping {file_path}: Not a Parquet file.")
        return
    
    # Generate output CSV filename
    csv_path = os.path.splitext(file_path)[0] + '.csv'
    
    try:
        # Read Parquet file
        df = pd.read_parquet(file_path)
        
        # Write to CSV with UTF-8 encoding
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Converted {file_path} to {csv_path}")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert Parquet file(s) to CSV.")
    parser.add_argument('file', nargs='?', help="Specific Parquet file to convert. If not provided, convert all Parquet files in the current directory.")
    
    args = parser.parse_args()
    
    if args.file:
        # Convert specific file
        convert_parquet_to_csv(args.file)
    else:
        # Convert all Parquet files in current directory
        parquet_files = glob.glob('*.parquet')
        if not parquet_files:
            print("No Parquet files found in the current directory.")
            return
        for file in parquet_files:
            convert_parquet_to_csv(file)

if __name__ == "__main__":
    main()
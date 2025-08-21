import pyarrow.parquet as pq

from datasets import load_dataset
import pyarrow as pa
import glob, os

files = glob.glob('parquet/*.parquet')
cnt = 0
for f in files:
    cnt += 1
    try:
        dataset = load_dataset("parquet", data_files=f, split="train")
    except Exception as e:
        os.remove(f)
        print('❌', f, e)
        cnt -= 1
print(f'Romove {cnt} files')
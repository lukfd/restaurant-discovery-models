import os
import argparse
from pathlib import Path

import pyarrow.dataset as ds

def main(args):
    dataset = ds.dataset(args.parquet, format="parquet")
    ids = os.listdir(args.directory.resolve())
    ids = [ i.split(".")[0] for i in ids ]

    if args.debug:
        print(f"Found {len(ids)} number of files")
        print()

        df = dataset.to_table().to_pandas()
        print(df)
        print()

        num_rows = sum(batch.num_rows for batch in dataset.to_batches())
        num_columns = len(dataset.schema)
        print(f"Shape: ({num_rows}, {num_columns})")
        print()

    missing = set()

    for batch in dataset.scanner(columns=[args.parquet_id]).to_batches():
        arr = batch.column(0)

        for v in arr.to_pylist():
            if args.debug:
                print(f"ID: {v}")

            if v not in ids:
                if args.debug:
                    print(f"Missing ID found: {v}")
                missing.add(v)

    print("Any missing parquet ids on disk?", bool(missing))
    if missing:
        print("Sample missing:", len(missing))
        print(missing)


if __name__ == "__main__":
    description = "Check if the parquet file ids are in the directory file"
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--parquet", type=Path, help="Path to parquet file")
    parser.add_argument("-i", "--parquet-id", type=str, help="ID ")
    parser.add_argument("-d", "--directory", type=Path, help="Directory path with id files")
    parser.add_argument("--debug", action="store_true", help="Print additional debug messages")
    
    args = parser.parse_args()
    print(args)
    main(args)


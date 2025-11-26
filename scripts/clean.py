import argparse
import os
import random
import re
import shutil
from PIL import Image
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


# %%
def to_snake_case(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    s = re.sub(r"\W+", "_", s)
    return s.lower()


def clean_and_split_df(table: pa.Table):
    photo_columns = ["photo_id", "business_id", "caption", "label"]
    business_columns = set(table.column_names) - set(photo_columns)
    business_columns = ["business_id", *business_columns]

    photo_details = table.select(photo_columns)
    business_details = table.select(business_columns)

    # Deduplicate business_details by business_id
    business_details_df = business_details.to_pandas()
    business_details_df = business_details_df.drop_duplicates(subset=["business_id"])
    business_details_df.reset_index(drop=True, inplace=True)

    business_details = pa.Table.from_pandas(business_details_df)
    business_details = business_details.rename_columns(
        [
            to_snake_case(col.removeprefix("attributes."))
            for col in business_details.column_names
        ]
    )

    return photo_details, business_details


def reduce(
    sampled_photo_ids: list[str], photo_details: pa.Table, business_details: pa.Table
):
    photo_details_r = photo_details.filter(
        pc.is_in(pc.field("photo_id"), pa.array(sampled_photo_ids))
    )

    business_details_r = business_details.filter(
        pc.is_in(
            pc.field("business_id"),
            photo_details_r.column("business_id"),
        )
    )

    return photo_details_r, business_details_r


# %%
def main(sample_size: int, data_dir: str):
    # %%
    photo_dir = f"{data_dir}/original/yelp_photos/photos"
    photo_ids = [i.removesuffix(".jpg") for i in os.listdir(photo_dir)]
    yelp_data = pq.read_table(f"{data_dir}/clean/yelp.gz")

    photo_details, business_details = clean_and_split_df(yelp_data)
    pq.write_table(photo_details, f"{data_dir}/clean/photo_details.parquet")
    pq.write_table(business_details, f"{data_dir}/clean/business_details.parquet")

    if sample_size >= 0:
        sampled_photo_ids = random.sample(photo_ids, sample_size)
        photo_details, business_details = reduce(
            sampled_photo_ids, photo_details, business_details
        )
        pq.write_table(photo_details, f"{data_dir}/clean/reduced_photo_details.parquet")
        pq.write_table(
            business_details, f"{data_dir}/clean/reduced_business_details.parquet"
        )
    else:
        sampled_photo_ids = photo_ids

    # %%
    target_photo_dir = f"{data_dir}/clean/reduced_photos"

    for photo_id in photo_details.column("photo_id"):
        source_path = os.path.join(photo_dir, f"{photo_id}.jpg")
        target_path = os.path.join(target_photo_dir, f"{photo_id}.jpg")

        if os.path.exists(target_path):
            continue

        try:
            _ = Image.open(source_path)
            shutil.move(source_path, target_path)
        except IOError as e:
            print(e)
            print(f"Photo could not be opened: {source_path}")

    photo_ids_r = [i.removesuffix(".jpg") for i in os.listdir(target_photo_dir)]
    photos_scores = (
        photo_details.filter(pc.is_in(pc.field("photo_id"), pa.array(photo_ids_r)))
        .join(business_details, "business_id")
        .combine_chunks()
    )
    print(len(photo_ids_r))
    print(photo_details.shape)
    print(photos_scores.shape)
    pq.write_table(photos_scores, f"{data_dir}/clean/model_data.parquet")


# %%
if __name__ == "__main__":
    description = "Check if the parquet file ids are in the directory file"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-n", type=int, help="Number of images to use in subset")
    parser.add_argument("-d", type=Path, help="folder containing clean + original data")

    args = parser.parse_args()
    print(args.n, args.d.resolve())
    main(args.n, args.d.resolve())

"""
convert_to_hf_dataset.py

Usage:
  python3 convert_to_hf_dataset.py --dataset-dir synthetic_dataset

Description:
  This script takes the JSON files from your synthetic data generator 
  and converts them into a Hugging Face dataset (with text, bbox, and image path).
  It does not modify or remove any lines in your existing generation code.
"""

import os
import json
import argparse
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", 
        type=str, 
        default="synthetic_dataset",
        help="Directory containing the generated synthetic images and .json annotations."
    )
    args = parser.parse_args()

    records = []

    # Loop over all per-image .json files
    for file_name in sorted(os.listdir(args.dataset_dir)):
        if file_name.endswith(".json"):
            json_path = os.path.join(args.dataset_dir, file_name)
            with open(json_path, "r") as f:
                annotation = json.load(f)

            # Collect relevant fields
            # 'annotation["image"]' should be the .png path
            # 'annotation["text"]' the GT text
            # 'annotation["bbox"]' or 'annotation["bboxes"]' the bounding boxes
            record = {
                "image_path": annotation["image"],
                "text": annotation["text"],
                "bbox": annotation.get("bbox", [])  
            }
            records.append(record)

    # Build a Hugging Face Dataset
    hf_dataset = Dataset.from_list(records)

    # Save it to disk in the same directory or a subfolder
    save_path = os.path.join(args.dataset_dir, "hf_dataset")
    hf_dataset.save_to_disk(save_path)
    print(f"HF dataset saved to: {save_path}")
    print(f"Number of samples: {len(hf_dataset)}")

if __name__ == "__main__":
    main()
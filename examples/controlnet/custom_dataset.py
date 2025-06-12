import os
from PIL import Image
from datasets import Dataset, DatasetDict

def load_custom_image_dataset(data_dir: str) -> Dataset:
    input_dir = os.path.join(data_dir, "input_small")
    target_dir = os.path.join(data_dir, "targets_small")

    # Ensure folders exist
    if not os.path.isdir(input_dir) or not os.path.isdir(target_dir):
        raise FileNotFoundError("Both input_small and targets_small directories must exist in train_data_dir.")

    input_files = sorted(os.listdir(input_dir))
    target_files = sorted(os.listdir(target_dir))

    # Match by filenames
    common_files = sorted(set(input_files).intersection(set(target_files)))

    def generator():
        for fname in common_files:
            input_path = os.path.join(input_dir, fname)
            target_path = os.path.join(target_dir, fname)

            try:
                input_img = Image.open(input_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")

                yield {
                    "input_small": input_img,
                    "targets_small": target_img
                }
            except Exception as e:
                print(f"Failed to load image pair {fname}: {e}")
                continue

    dataset = Dataset.from_generator(generator)
    return DatasetDict({"train": dataset})
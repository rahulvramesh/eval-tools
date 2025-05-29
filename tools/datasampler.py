import os
import json
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("sampling.log"),
        logging.StreamHandler()
    ]
)

INPUT_BASE = Path("./data")
OUTPUT_BASE = Path("./sampled_data")
SAMPLE_SIZE = 100

file_sampled_count = 0

def ensure_dir(path: Path):
    """Ensure the directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def get_output_path(input_path: Path) -> Path:
    """Return the output path in sampled_data with same structure."""
    relative_path = input_path.relative_to(INPUT_BASE)
    output_dir = OUTPUT_BASE / relative_path.parent
    ensure_dir(output_dir)  # FIXED: ensure the actual directory
    return output_dir / f"sampled_{input_path.name}"

def sample_jsonl_file(input_path: Path, sample_size: int):
    global file_sampled_count
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            logging.warning(f"[SKIPPED] Empty file: {input_path}")
            return

        sampled = random.sample(lines, min(sample_size, len(lines)))
        output_path = get_output_path(input_path)
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.writelines(sampled)

        logging.info(f"[JSONL] Sampled {len(sampled)} from {input_path} → {output_path}")
        file_sampled_count += 1
    except Exception as e:
        logging.error(f"Failed to sample JSONL file {input_path}: {e}")

def sample_json_file(input_path: Path, sample_size: int):
    global file_sampled_count
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            logging.warning(f"[SKIPPED] Invalid or empty JSON array: {input_path}")
            return

        sampled = random.sample(data, min(sample_size, len(data)))
        output_path = get_output_path(input_path)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(sampled, out_f, indent=2)

        logging.info(f"[JSON] Sampled {len(sampled)} from {input_path} → {output_path}")
        file_sampled_count += 1
    except Exception as e:
        logging.error(f"Failed to sample JSON file {input_path}: {e}")

def walk_and_sample(input_base: Path, sample_size: int):
    total_files = 0
    for root, _, files in os.walk(input_base):
        for file in files:
            total_files += 1
            print("Found:", os.path.join(root, file))
            path = Path(root) / file
            if path.suffix == ".jsonl":
                sample_jsonl_file(path, sample_size)
            elif path.suffix == ".json":
                sample_json_file(path, sample_size)
            else:
                logging.debug(f"[SKIPPED] Unsupported file: {path}")
    
    logging.info(f"Sampling completed. Processed {file_sampled_count} out of {total_files} files.")

# Execute
if __name__ == "__main__":
    logging.info("Starting dataset sampling...")
    walk_and_sample(INPUT_BASE, SAMPLE_SIZE)
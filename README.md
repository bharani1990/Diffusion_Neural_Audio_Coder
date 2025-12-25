# Diffusion-Based Neural Audio Coder

## Steps

- **Step 1:** Convert the .flac to Mel-Specs
  - `uv run python main.py --stage extract --root_in data/raw/train-clean-5 --root_out data/processed/train-clean-5`
  - `uv run python main.py --stage extract --root_in data/raw/dev-clean-2   --root_out data/processed/dev-clean-2`
    
- **Step 2:** Normalise the Train data and create the manifest files
  - `uv run python main.py \
  --stage preprocess \
  --root_in data/processed/train-clean-5 \
  --root_out data/processed_norm/train-clean-5 \
  --stats_path data/processed/stats.pt \
  --manifest_path data/processed/train_manifest.jsonl`
  - `uv run python main.py \
  --stage preprocess \
  --root_in data/processed/dev-clean-2 \
  --root_out data/processed_norm/dev-clean-2 \
  --stats_path data/processed/train-clean-5/stats.pt \
  --manifest_path data/processed/dev_manifest.jsonl`

- **Step 3:** Create Model and test it via the test_model file inside the tests folder
  - `uv run python tests/test_model.py`

- **Step 4:** Create the dataset and test it via the test_dataset file inside the tests folder
  - `uv run python tests/test_dataset.py`

- **Step 5:** Create a lightning module and use it as a wrapper for training
  - `uv run python main.py --stage train \
  --train_manifest data/processed/train_manifest.jsonl \
  --dev_manifest   data/processed/dev_manifest.jsonl \
  --batch_size 8 \
  --frames 120 \
  --max_epochs 10 \
  --lr 1e-4`

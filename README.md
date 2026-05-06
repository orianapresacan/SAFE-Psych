# Safe-Psych Benchmark

This repository contains the code for evaluating LLMs on Safe-Psych, a sequential psychiatric diagnostic benchmark built from clinical notes divided into staged information sections. The evaluation measures diagnostic accuracy, clarification-seeking, premature diagnosis, and whether models appropriately abstain when the available evidence is insufficient.


## Data Preprocessing

1. Download the benchmark CSV from [Hugging Face](https://huggingface.co/datasets/safepsych/Safe-Psych)

2. Place the downloaded file in the `data/` folder:

   ```text
   data/data.csv
   ```
3. Run the preprocessing script from the repository root:
    ``` bash
    python data/data_preprocessing.py
    ```

    This creates two JSON files:
    - data/samples.json contains the case IDs and ordered clinical sections used for LLM inference.
    - evaluations/data_gt.json contains the ground-truth annotations used to compute evaluation metrics.


## Installation

For double-anonymous review, download the anonymized repository from:

https://anonymous.4open.science/r/SAFE-Psych-32C1/

Use the **ZIP** button, then create a folder and add the downloaded archive there:

```bash
mkdir Safe-Psych
cd Safe-Psych
unzip SAFE-Psych-32C1.zip
```


## Environment

We provide installation options with `uv`, `pip`, and `conda`.

### Option 1: uv

```bash
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### Option 2: conda
```bash
conda env create -f environment.yml
conda activate safe_psych
```

### Option 3: pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The core environment is sufficient for data preprocessing, metric computation, plotting, and API-based model calls.

### API keys
For closed-source models, export the relevant API keys before running inference:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```


### GPU inference for open-weight models

GPU inference requires a CUDA-compatible PyTorch build and `vllm`:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_gpu.txt
```

Adjust the PyTorch install command for your CUDA/platform as needed: https://pytorch.org/get-started/locally/


## Running inference

Run inference with a target model, judge model, and prompting strategy:

```bash
python main.py \
  --model gemma-3-4b \
  --judge gpt-5.4 \
  --strategy full_info_no_abstention \
  --data_path data/samples_1.json \
  --seed 123
```


Available strategies are:
- full_info_no_abstention
- full_info_abstention_aware
- sequential_info_neutral_prompting
- sequential_info_abstention_aware


Available target models:
```text
llama3.1-8b
mistral-small-3.1-24B
gemma-3-4b
gemma-3-12b
gemma-3-27b
medgemma-27b
med42-8b
qwen_3-32b
gpt-5.4
claude-opus-4.6
gemini-2.5-flash-no-thinking
gemini-2.5-flash-thinking
```

### Outputs
Outputs are written to the `runs/` directory. Each run is saved in a timestamped subdirectory named by model, strategy, judge, and seed, for example:

```text
runs/gemma-3-4b_full_info_no_abstention_judge-gpt-4o-mini_seed-123_20260506_143012/
```

Each run directory contains:

```text
_tmp/
trajectories.jsonl
timing.json
```

The `_tmp/` directory stores intermediate model and judge inputs/outputs at each step. `trajectories.jsonl` contains one data sample per row, including the complete conversation states and judge actions. `timing.json` records the runtime for the run.

## Evaluation

Evaluation scripts are provided in the `evaluations/` folder. Detailed instructions for reproducing all the metrics, tables, and figures reported in the paper are included in that folder.


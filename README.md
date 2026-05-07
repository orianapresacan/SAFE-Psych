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

Choose one of the following setups:

- **Base environment:** for data preprocessing, metric computation, plotting, and API-based model calls.
- **GPU environment:** for open-weight model local inference. This includes the base environment plus GPU dependencies such as `vllm`.

### Base environment

#### Option 1: uv

```bash
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

#### Option 2: conda

```bash
conda env create -f environment.yml
conda activate safe_psych
```

#### Option 3: pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### GPU environment

Use this setup if you want to run open-weight models locally. 

Adjust the PyTorch install command for your CUDA/platform using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

#### Option 1: uv

```bash
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# First install PyTorch with the command recommended for your CUDA/platform:
uv pip install torch torchvision torchaudio

uv pip install -r requirements_gpu.txt
```

#### Option 2: conda

```bash
conda create -n safe_psych python=3.11 pip
conda activate safe_psych

# First install PyTorch with the command recommended for your CUDA/platform:
pip install torch torchvision torchaudio

pip install -r requirements_gpu.txt
```

#### Option 3: pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# First install PyTorch with the command recommended for your CUDA/platform:
pip install torch torchvision torchaudio

pip install -r requirements_gpu.txt
```

`requirements_gpu_lock.txt` was generated with `uv` and pins the exact Python dependency versions for the GPU environment, including the base requirements and `vllm`, to ensure dependency compatibility and reproducible installation.

## API keys

For closed-source models, export the relevant API keys before running inference:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## Running inference

Run inference with a target model, judge model, and prompting strategy:

```bash
python main.py \
  --model gemma-3-4b \
  --judge gpt-5.4 \
  --strategy full_info_no_abstention \
  --data_path data/samples.json \
  --seed 123
```


Available **strategies** are:
```text
full_info_no_abstention
full_info_abstention_aware
sequential_info_neutral_prompting
sequential_info_abstention_aware
```

Available **target models**:
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

Available **judge model**:
```text
gpt-5.4
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


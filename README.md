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

Clone the repository:

```bash
git clone https://anonymous.4open.science/r/SAFE-Psych-32C1/
cd SAFE-Psych-32C1
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

### GPU inference for open-weight models

Open-weight model inference requires a CUDA-compatible PyTorch installation and vllm:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```
The correct PyTorch CUDA wheel may differ depending on your machine. Closed-source models are evaluated through provider APIs and do not require local GPU dependencies.



## Evaluation

Evaluation scripts are provided in the `evaluations/` folder. Detailed instructions for reproducing all the metrics, tables, and figures reported in the paper are included in that folder.


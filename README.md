# Safe-Psych Benchmark | [Paper]()


This repository contains the code for evaluating large language models on a clinical diagnostic reasoning benchmark. The benchmark is structured around patient cases divided into sequential information sections. Models are evaluated on whether they should request more ..

```text
.
├── data/
│   ├── data.csv
│   ├── data_preprocessing.py
│   └── samples.json
├── evaluations/
│   └── data_gt.json
├── README.md
└── ...
```

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


## Evaluation

Evaluation scripts are provided in the `evaluations/` folder. Detailed instructions for reproducing all the metrics, tables, and figures reported in the paper are included in that folder.
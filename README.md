# MMCA Repository  
## Replication Package (All Models)

This repository is provided **for replication purposes only**.  
It contains implementations of multiple models evaluated under the same experimental protocol.

---

## Table of Contents

1. [Clone or Download the Repository](#1-clone-or-download-the-repository)  
2. [Download Datasets](#2-download-datasets)  
3. [Download Pretrained Embeddings](#3-download-pretrained-embeddings)  
4. [Running Experiments](#4-running-experiments)  
5. [Output Structure](#5-output-structure)  
6. [Notes](#6-notes)  

---

## 1. Clone or Download the Repository

You may obtain the code in one of the following ways:

### Option A: Clone via Git (Recommended)

```bash
git clone https://github.com/AnonymousCodeBase-Research/MMCA.git
cd MMCA
```

### Option B: Download ZIP

1. Visit: https://github.com/AnonymousCodeBase-Research/MMCA  
2. Click **Code → Download ZIP**  
3. Extract the archive locally  

---

## 2. Download Datasets

Download all datasets from:

https://drive.google.com/drive/folders/1RHvr4ysbrEzUXyFoc8KXFIy8qbrnWpc5

There are four dataset categories:

- **Structured**
- **Structured_Textual**
- **Textual**
- **Dirty**

For each dataset:

- 10 folds are provided  
- Each fold includes:
  - `train`
  - `valid`
  - `test`

### Special Case (MMCA)

For all datasets except **Structured**, an additional:

```
train_temp
```

dataset is provided for each fold.

These are used exclusively for MMCA’s merging strategy to enrich the original training data.

### Dataset Placement

Place the `datasets` folder inside each model folder.

Example:

```
MMCA/
│
├── datasets/
│
├── script_MMCA.py
│
└── wiki.en.bin
```

Similarly:

```
AAM/datasets/
DeepER/datasets/
DeepMatcher/datasets/
MCA/datasets/
ML/datasets/
```

Each model expects its datasets locally under its own directory.

---

## 3. Download Pretrained Embeddings

Download **English: bin+text** from:

https://fasttext.cc/docs/en/pretrained-vectors.html

After extracting the downloaded archive, locate:

```
wiki.en.bin
```

Place it inside each model folder:

```
MMCA/wiki.en.bin
AAM/wiki.en.bin
DeepER/wiki.en.bin
DeepMatcher/wiki.en.bin
MCA/wiki.en.bin
```

Each model loads embeddings from its local directory.

---

## 4. Running Experiments

All scripts should be executed from within each model’s folder.

---

### AAM, DeepER, DeepMatcher, MCA, MMCA

Run:

```bash
python script_MODEL.py
```

Replace `MODEL` with the corresponding model name  
(e.g., `script_MMCA.py`, `script_AAM.py`, etc.).

Each script will automatically:

- Iterate through all datasets  
- Run all 10 folds  
- Execute training, validation, and testing  
- Log hyperparameters and performance  

---

### ML Baselines

Navigate to the `ML` folder and run the appropriate script.

Files ending with:

```
Structured
```

are used for Structured datasets.

Other files are used for:

- Textual  
- Dirty  

Each script automatically runs all applicable datasets.

---

## 5. Output Structure

After execution, each model generates a:

```
runs/
```

directory inside its folder.

Example:

```
MMCA/runs/
```

Inside `runs/`:

### results.csv

This file records:

- Dataset path  
- Split (train / valid / test)  
- Fold index  
- Timestamp  
- Hyperparameter configuration  
- Performance metrics:
  - Precision
  - Recall
  - F1
  - Accuracy
  - PR-AUC
  - Loss  
- Checkpoint path  

### checkpoints/

All saved model checkpoints are stored in this folder.

---

## 6. Notes

- This repository is intended strictly for replication of reported results.  
- All scripts automatically iterate through datasets and folds.  
- No manual configuration per dataset is required.  
- Ensure correct placement of:
  - `datasets/`
  - `wiki.en.bin`

before running experiments.

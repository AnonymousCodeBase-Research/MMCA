# Reproducing Experiments (All Models)

This repository provides scripts to run multiple models on the same set of datasets.  
Follow the steps below to reproduce results for each method.

---

## 1) Download datasets

Download the datasets from Google Drive and place them into the repository:

- Dataset link: https://drive.google.com/drive/folders/1iwVDasLq2U95BDuVLS9Zn3Cx6M2HsX9C

> Note: If your scripts expect a different path (e.g., `./data/`), adjust the folder name accordingly or update the path in the scripts.

---

## 2) Download embeddings (wiki.en.bin)

Download `wiki.en.bin` and place them into the repository:

Suggested structure:

repo_root/
  datasets/
  wiki.en.bin

## 3) Run a model script

Each model can be executed via its corresponding `script_MODEL_.py`.

### General command (from repo root)
python script_MODEL.py

Replace `MODEL` with the target model name (e.g., `script_MMCA.py`, `script_AAM.py`, etc., depending on what exists in this repo).

---

## DeepER and ML notes

For **DeepER** and **ML** baselines, the script file is located inside the corresponding model folder.


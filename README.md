# xCARE: Topology-Aware Continual LoRA for Concept Drift Mitigation (Journal Extension)

This repository contains the official implementation, datasets, and experimental results for the journal paper: 
**"Topology-Aware Continual Low-Rank Adaptation for Concept Drift Mitigation in Knowledge Tracing"** (Accepted at JEDM).

## 📌 What's New in this Version?
Compared to the CSEDU 2026 conference paper, this repository introduces the **Memory Sanctuary** mechanism, which significantly reduces catastrophic forgetting during adaptation:
- **Functional Distillation**: KL-divergence on a learner-disjoint retention split (`H_retain`).
- **Stratified Replay**: Replay buffer sourced directly from the pre-drift distribution.
- **Asymmetric Early Stopping**: Quadratic penalty to balance stability and plasticity.
- **Result**: Reduces Backward Transfer (BWT) from `-22.4%` (standard LoRA) to `-8.85%` over 15 seeds, and to just `-0.98%` with early stopping.

## 📂 Repository Structure
- `code/`: The Jupyter Notebook containing the full pipeline (Data loading, Memory Sanctuary training, Pareto Sweep, RAG evaluation).
- `data/`: Pretrained weights (`reliable_xlstm_model.pth`), feature metadata, and RAG annotations.
- `results/`: Per-seed metrics (`results_long.csv`), raw predictions (`predictions.pkl`), and statistical tests (`paired_tests.csv`).
- `manuscript/`: LaTeX source and generated figures.

## 🚀 Quick Start
1. Clone this repository.
2. Ensure you have the dependencies installed: `pip install -r requirements.txt`.
3. Open `code/xCARE_Official_Implementation_v3.ipynb` in Google Colab or Jupyter Lab.
   *Note: If you run locally, make sure to change `CFG["DRIVE_DIR"]` at the top of the notebook to point to the `data/` folder of this repository.*
4. The notebook is fully reproducible and includes a checkpointing mechanism (`RESILIENT RUN`) to resume training if interrupted.

## 📊 Main Results (Held-out Test, 15 seeds)
- **Sudden Drift**: Accuracy `0.717 ± 0.028`, AUC-ROC `0.820 ± 0.015`.
- **Memory Retention (BWT)**: `-8.85 ± 1.72` (vs `-11.69` for standard FFT).
- **Parameter Efficiency**: Uses 9,504 trainable parameters (56% fewer than FFT).

## 📥 Large Data Files (Tensors & Train Block)
Due to GitHub's file size limits, the large pre-processed tensors (`.pt`) and the `train_block.csv` are not hosted directly in this repository. You can download them from the following Google Drive folder:

👉 **[Download Tensors and Train Block from Google Drive](https://drive.google.com/file/d/1FHgAZrWVuf0zgrzWn5Ilh5434lwXu5eU/view?usp=sharing, https://drive.google.com/file/d/1ZDLq3DxTpY9XZTLWFjcIltkH5_Czim8E/view?usp=sharing, https://drive.google.com/file/d/1_ZV7VG17pRjhiPsEt_SDo9K4Gy9Jjf0p/view?usp=sharing, https://drive.google.com/file/d/1azZFLjZi81GycjnBhV8UyA1Vl9-V8lKM/view?usp=sharing, https://drive.google.com/file/d/1cEqsTQ5oTZzvbUMgSjoeLJ9T2uIPDL88/view?usp=sharing, https://drive.google.com/file/d/1dIfS3Lvmx9CeYmgOND98tZ6-6OetynMs/view?usp=sharing, https://drive.google.com/file/d/1goyXEXMtH_qlHffIXQ5S_lg6ia3ASI9s/view?usp=sharing)**

*Once downloaded, place these files inside the `data/` folder of this repository before running the notebook.*

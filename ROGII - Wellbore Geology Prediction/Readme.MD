# 🛢️ ROGII - Wellbore Geology Prediction

This repository contains my solutions, experiments, and notebooks for the [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) competition on Kaggle.

## Objective
Drilling a horizontal well requires immense technical precision. The goal of this project is to develop machine learning models that predict the geology (True Vertical Thickness - `TVT`) encountered along a horizontal wellbore. By identifying favorable rock layers from drilling data (Gamma Ray logs and spatial telemetry), these models aim to automate geosteering, reduce resource waste, and improve operational safety.

## Tech Stack & Methodology
* **Data Processing Engine:** `Polars` for ultra-fast, multi-threaded feature engineering, replacing Pandas to handle millions of rolling window calculations efficiently.
* **Modeling:** GPU-Accelerated `XGBoost` and `LightGBM` (Hist tree methods).
* **Validation Strategy:** 5-Fold GroupKFold (grouped by `well_id`) to prevent sequential data leakage.
* **Domain Integration:** Physics-Informed Machine Learning. Integrating deterministic algorithms like Z-Velocity Particle Filters and Beam Search Trajectory estimation to act as spatial constraints for the gradient boosters.
* **Memory Management:** Aggressive downcasting (Float32) and automated garbage collection designed to run flawlessly on constrained 15GB T4x2 / P100 GPU environments.

## Configuration & Paths
The code is structured to be highly portable between Kaggle, local environments, and customized portals (like Shipd). Global variables are explicitly exposed at the top of all scripts to easily adjust input/output streams:
```python
INPUT_PATH  = "/kaggle/input/rogii-wellbore-geology-prediction"
OUTPUT_PATH = "/kaggle/working"

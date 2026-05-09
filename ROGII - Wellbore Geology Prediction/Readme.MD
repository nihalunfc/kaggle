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

## Submission Tracker

| Date | Notebook Name | Description / Key Features | Local CV (RMSE) | Public LB (RMSE) |
| :--- | :--- | :--- | :---: | :---: |
## 📊 Submission Tracker

| Date | Notebook Name | Description / Key Features | Local CV (RMSE) | Public LB (RMSE) |
| :--- | :--- | :--- | :---: | :---: |
| 2026-05-08 | `version_01` | High-speed Polars feature engineering + XGBoost baseline. Basic error handling. | 15.18 | 11551.95 |
| 2026-05-08 | `version_02` | Injected Beam Search & Particle Filters. Mapping bug caused global median fallback. | ~15.00 | 685.739 |
| 2026-05-09 | `version_08` | Fixed `row_index` mapping trap. Polars pipeline now successfully passing physics features to XGBoost. | 15.60 | 15.512 |
| 2026-05-09 | `version_09` | Introduced Heterogeneous Ensemble (XGB, LGBM, CAT). Used Nelder-Mead to optimize blend weights. | TBA | TBA |
| 2026-05-09 | `version_10` | Built bespoke Numba JIT physics engine. Added Dual-State PF (Z-Vel + ANCC) and Multi-Scale NCC. | TBA | TBA |

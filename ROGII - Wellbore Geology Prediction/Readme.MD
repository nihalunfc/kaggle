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
| 2026-May-08 | `v1` | High-speed Polars feature engineering + XGBoost baseline. Memory-safe T4 inference with safe fallback contingencies. | 15.18 | 11551.955 |
| 2026-May-08 | `v2` | Injected Beam Search & Particle Filters. Applied "Global Median" fallback for corrupted test wells. | 15.00 | 685.739 |
| YYYY-MM-DD | | | | |
| YYYY-MM-DD | | | | |
| YYYY-MM-DD | | | | |

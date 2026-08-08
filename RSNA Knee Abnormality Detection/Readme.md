# RSNA Knee Abnormality Detection 🦴

## Project Overview
This repository contains my applied research and modeling pipeline for the automated detection of knee abnormalities using medical imaging. The primary objective is to detect 12 distinct structural conditions—including ACL/MCL tears, meniscal lesions, and compartmental osteoarthritis—directly from Magnetic Resonance Imaging (MRI) studies. 

The extensive dataset utilized for this project is provided by the Radiological Society of North America (RSNA). It features a diverse, international mix of imaging sites, scanners, clinical protocols, and patient populations.

## Technical Challenge
The knee is one of the most complex joints in the human body, and diagnosing injuries requires inherently 3D spatial reasoning. A standard knee MRI examination is not a single image, but multiple volumetric series acquired with varying pulse sequences (such as fluid-sensitive or fat-suppressed) across different anatomical planes (sagittal, coronal, axial). 

This project tackles several complex data engineering and computer vision challenges:
- **DICOM Processing**: Standardizing raw, uncompressed multi-planar MRI series and transforming them for deep learning ingestion.
- **Data Sparsity & Imbalance**: Navigating nuanced diagnostic thresholds where clinical abnormalities vary greatly in prevalence.
- **Multimodal Integration**: (Upcoming) Leveraging Natural Language Processing (NLP) to incorporate unstructured, free-text radiology reports as weak supervision signals alongside the imaging data.

## Modeling Approach
- **Baseline Architecture**: A Convolutional Neural Network (ResNet18) implemented in PyTorch. The current pipeline processes normalized 2D representations of the fluid-sensitive sequences to predict a 12-dimensional multi-label output vector.
- **Objective Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) optimized for multi-label classification.
- **Optimizer**: Adam

## Tech Stack
- **Deep Learning**: PyTorch, TorchVision
- **Computer Vision & Imaging**: OpenCV, pydicom
- **Data Engineering**: NumPy, Pandas

## Version History & Updates
*Updates on model convergence, evaluation metrics, and architectural iterations are documented below.*

- **#v1**: Initial baseline implementation with custom PyTorch Dataset for DICOM processing.
- **#v2**: Added offline model weight loading for ResNet18.
- **#v3**: Configured dataset loading paths for the RSNA environment.
- **#v4**: Resolved PyTorch 2.6 checkpoint unpickling strictness for robust offline inference.

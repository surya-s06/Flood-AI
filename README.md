# FloodAI

## Multi-Modal Flood Intelligence System

FloodAI is an end-to-end artificial intelligence system designed to improve flood monitoring and disaster response by combining multiple sources of information into a single decision-making pipeline.

Unlike traditional flood prediction projects that rely on a single dataset or model, FloodAI integrates satellite imagery, weather observations, and environmental factors to produce a comprehensive flood risk assessment.

The long-term vision is to simulate an intelligent edge AI system capable of processing satellite imagery, analysing environmental conditions, and generating actionable flood intelligence while minimizing the amount of data transmitted.

---

# Project Objectives

- Detect flooded regions from satellite imagery.
- Predict flood risk using environmental data.
- Analyse weather conditions related to flooding.
- Fuse multiple AI models into a single flood intelligence system.
- Generate actionable flood risk reports.
- Simulate intelligent disaster monitoring workflows.

---

# System Architecture

```text
                    FloodAI

         ┌────────────────────────┐
         │ Satellite Module       │
         │ (Computer Vision)      │
         └────────────────────────┘
                    │
                    ▼
           Flood Segmentation

                    +

         ┌────────────────────────┐
         │ Weather Module         │
         │ (Time Series / ML)     │
         └────────────────────────┘
                    │
                    ▼
             Weather Risk

                    +

         ┌────────────────────────┐
         │ Environmental Module   │
         │ (Regression)           │
         └────────────────────────┘
                    │
                    ▼
            Flood Probability

                    ▼

          Multi-Modal Fusion Engine

                    ▼

         Comprehensive Flood Report
```

---

# Current Project Status

## Completed

- Project initialization
- Repository structure
- Development environment setup
- Dataset collection
- Exploratory Data Analysis (EDA)
- Environmental regression baseline
- Project architecture design

---

## In Progress

- Satellite image preprocessing
- Satellite dataset analysis
- Weather data preprocessing

---

## Planned

### Phase 1
Environmental Intelligence Module

### Phase 2
Satellite Flood Segmentation Module

### Phase 3
Weather Intelligence Module

### Phase 4
Multi-Modal Fusion Engine

### Phase 5
Interactive Web Application

### Phase 6
Edge AI Simulation for Satellite Deployment

---

# Datasets

## Dataset A (Development)

Synthetic environmental flood dataset.

Current purpose:
- Regression pipeline development
- Testing
- Debugging

A real-world environmental dataset will replace this module later.

## Current Status

**Paused**

### Reason

The initial environmental dataset turned out to be **synthetically generated**.

Although it was useful for testing the preprocessing pipeline, training a production model on synthetic values would not produce reliable results.

### Work Completed

- Data preprocessing pipeline
- Feature analysis
- Initial regression experiments
- Baseline training pipeline

### Current Decision

The environmental module is paused while a higher-quality real-world dataset is being sourced.

Development will resume once a suitable dataset is available.

---

## Dataset B Complete ✅

Satellite imagery and segmentation masks used for flood detection and segmentation.

## Data

- 290 satellite images
- 290 corresponding segmentation masks

# Vision Module V1

The first version established the complete computer vision pipeline.

## Improvements
- Baseline U-Net implementation
- Basic training pipeline
- Initial evaluation and prediction pipeline

### Results

| Metric | Score |
|---------|--------|
| IoU | 0.5715 |
| Dice Score | 0.7213 |
| Pixel Accuracy | 0.8148 |

### Conclusion

Version 1 successfully demonstrated that the model could learn meaningful flood segmentation.

However, the predictions contained:

- rough boundaries
- missed small flooded regions
- incomplete segmentation masks

---

# Vision Module V2

Version 2 focused on improving the training pipeline instead of changing the architecture.

## Improvements

- BCE + Dice Loss
- Paired data augmentation
- Learning rate scheduler
- Gradient clipping

## Results

| Metric | V1 | V2 |
|---------|------|------|
| IoU | 0.5715 | **0.6227** |
| Dice Score | 0.7213 | **0.7629** |
| Pixel Accuracy | 0.8148 | **0.8280** |

### Outcome

The improvements produced measurable gains across all evaluation metrics.

Visual inspection also showed cleaner segmentation masks and better detection of large flooded regions.

Despite the improvement, the model still struggles with:

- thin flood boundaries
- small flooded regions
- fine segmentation details

---

### Vision V3 (Final)

## Improvements
- Batch Normalization
- Early Stopping
- Gradient Clipping
- Validation IoU tracking
- Automatic model versioning
- Improved experiment tracking

Training was monitored using:
- Training Loss
- Validation Loss
- Validation IoU
- Learning Rate Scheduler
- Early Stopping

The final model converged successfully and stopped automatically at **Epoch 16**, indicating that further training was not providing meaningful improvements.

## Final Results

| Metric | Score |
|--------|------:|
| IoU | **0.6887** |
| Dice Score | **0.8132** |
| Pixel Accuracy | **0.8646** |

Compared to the initial baseline (Vision V1), the final model achieved:

- **+20.5% relative improvement in IoU**
- **+12.7% relative improvement in Dice Score**
- **Higher overall pixel accuracy**
- **Cleaner and more accurate flood segmentation masks**

---

## Dataset C

Historical weather and rainfall datasets used to estimate weather-related flood risk.

## Status

Not started.

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- PyTorch
- OpenCV
- Matplotlib
- Jupyter Notebook

---

# Planned Fusion Model

The final FLOOD AI system will combine:

Environmental Module
+
Vision Module
+
Weather Module

↓

Flood Intelligence Engine

↓

Flood Prediction & Analysis

---

# Repository Structure

```text
FloodAI/

│

├── app/

├── data/

│   ├── environmental/
│   ├── environmental_synthetic_data/
│   ├── satellite/
│   └── weather/

│

├── models/

│   ├── environmental/
│   ├── vision/
│   ├── weather/
│   └── fusion/

│

├── notebook/

├── outputs/

├── utils/

│

├── README.md

├── requirements.txt

└── .gitignore
```

---

# Disclaimer

FloodAI is an educational and research-oriented project exploring multi-modal artificial intelligence for disaster management. It is not intended for operational flood forecasting or emergency response without further validation.

## Vision Module

- Custom FloodDataset
- U-Net segmentation model
- 10-epoch training pipeline
- Prediction pipeline
- Evaluation metrics

# Experiments

| Version | Changes | IoU | Dice Score | Pixel Accuracy | Status |
|---------|---------|------:|------:|------:|---------|
| **V1** | Baseline U-Net + BCE Loss | **0.5715** | **0.7213** | **0.8148** | Archived |
| **V2** | BCE + Dice Loss + Paired Data Augmentation + Learning Rate Scheduler + Gradient Clipping | **0.6227** | **0.7629** | **0.8280** |  Current Best |
| **V3** | Batch Normalization + Early Stopping *(planned)* | — | — | — | 🚧 In Progress |

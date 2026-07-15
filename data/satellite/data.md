# Dataset B - Satellite

The dataset contains images of flood hit areas and corresponding mask images showing the water region.

There are 290 images and self annoted masks. The mask images were created using Label Studio, an open source data labelling software.

## Purpose

Satellite imagery and segmentation masks used for flood detection and segmentation.

## Link to Dataset 

https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation?

## Usage

Trained the model 3 times with these images

### Version 1 - 
- Baseline U-Net implementation
- Basic training pipeline
- Initial evaluation and prediction pipeline

### Version 2 - 
- BCE + Dice Loss
- Paired data augmentation
- Learning rate scheduler
- Gradient clipping

### Version 3 (Successful)
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

#### Concern
Overfitting was a concern because the size of the dataset is rather small.
But the diversity of the dataset helped in overcoming this problem.
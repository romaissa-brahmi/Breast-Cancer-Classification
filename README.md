# Breast Cancer Classification

## Project Overview

### Description
The goal of this project is to train a neural network to distinguish between benign and malignant breast 
tumors using the Wisconsin Diagnostic Breast Cancer dataset.
The workflow includes:
- Loading and balancing the dataset
- Preprocessing and standardizing the features
- Training and testing a neural network
- Visualizing performance (accuracy, loss, confusion matrix)

### Dataset
* Source: UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic) Data Set
* Features: 30 numerical features computed from digitized images of fine needle aspirate (FNA) of a breast mass
* Classes:
	* 0 → Benign
	* 1 → Malignant

In this project, a balanced subset of 212 samples per class is used.

### Model Architecture
A simple neural network with:
* Input layer: 30 features
* Hidden layer: one layer with ReLU activation
* Output layer: 2 neuron (multi-class classification)
* Loss: Cross Entropy Loss
* Optimizer: Adam

### Training
* 4 runs were performed to evaluate consistency
* The model reached 98.8% accuracy on the test set
* The best-performing run was used to compute the confusion matrix


## How to Run
- Open the Jupyter Notebook
```bash
jupyter notebook breast_cancer_classification_notebook.ipynb 
```
- Run the Python script
```bash
python3 breast_cancer_classification_nn.py 
```

## Folder Structure
```text
Breast Cancer Classification
├── README.md
├── breast_cancer_classification_notebook.ipynb
├── breast_cancer_classification_nn.py                               
└── results.csv                            
```

## Tools and Technologies
- Python3 (matplotlib, numpy)
- scikit-learn
- torch
- pandas


## Project Info
- **Author:** Romaïssa BRAHMI
- **Tested on:** macOS (M1 chip)

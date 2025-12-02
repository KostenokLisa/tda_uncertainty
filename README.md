# Uncertainty Estimation of Transformers' Predictions via Topological Analysis of the Attention Matrices
This repository contains an implementation of a novel uncertainty estimation approach for Transformer-based models. The method leverages **topological features derived from attention matrices** such as graph statistics, barcodes, and cross-barcodes to predict confidence scores for fine-tuned text classification models.

The goal of this work is to provide **reliable uncertainty estimates** for Transformer predictions *without modifying the underlying model architecture.*


## Pipeline

### 1. Feature Extraction

For each training sample:

- Compute BERT attention matrices across all layers and heads  
- Build graph representations of the attention patterns  
- Compute the following topological features:  
  - Graph statistics  
  - Barcodes  
  - Cross-barcodes (capturing interactions between layers and heads)  
- Aggregate these into a single feature vector per sample  

**Example workflow for BERT-base model trained on English CoLA bench:**  
`notebooks/end-to-end-feature_extraction.ipynb`

### 2. Score Predictor Training

A lightweight MLP is trained on the extracted feature vectors to predict confidence scores for model outputs.

Evaluation uses the **Area Under the Accuracy–Rejection Curve (AURC)**:  
A strong uncertainty estimator should allow the model to reject uncertain predictions while maintaining high accuracy on the retained instances.

**Example workflow for BERT-base model trained on English CoLA bench:**  
`notebooks/score_predictor_en_cola_no_fn.ipynb`

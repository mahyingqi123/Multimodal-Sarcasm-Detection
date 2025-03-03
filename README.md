# A Multi-Level Hybrid Fusion Architecture for Multimodal Sarcasm Detection

This repository contains the code and resources for our paper, **“A Multi-Level Hybrid Fusion Architecture for Multimodal Sarcasm Detection.”** Our goal is to detect sarcasm in video-based content by leveraging multiple modalities (text, audio, and video). The work introduces a unified framework that integrates:

1. **Early Fusion** (concatenating raw features),
2. **Intermediate Fusion** (using refined, pre-trained embeddings),
3. **Late Fusion** (aggregating modality-specific predictions),
4. **Intermodality Inconsistency Detection** (capturing mismatches across modalities).

By combining these strategies, our architecture captures subtle cross-modal cues crucial for recognizing sarcastic expressions.

---

## Overview

Sarcasm is inherently multimodal: text alone rarely conveys the full intent without tone of voice or visual cues. Our paper shows that **fusing textual, auditory, and visual data** significantly improves performance over unimodal approaches. Specifically, we propose:
- **Multi-Level Hybrid Fusion** that combines early, intermediate, and late fusion pipelines.
- **Intermodality Inconsistency Detector** to explicitly model mismatched sentiments (e.g., positive text and negative tone).

We evaluate the model on the **MUStARD** dataset (video snippets from TV shows), achieving state-of-the-art performance in detecting sarcastic utterances.

---

## Key Features
- **Scalable Feature Extraction**: Supports standard (ResNet, BERT, OpenSMILE) and more advanced (TimeSformer, BGE, AST) pre-trained models.
- **Flexible Fusion Methods**: Individual modules for early, intermediate, late, and inconsistency-based fusion—plus a final hybrid module.
- **Attention-Based Learning**: Uses an attention mechanism to dynamically weight each modality.
- **Extensive Ablation**: Shows how each fusion component contributes to overall performance.

---



## Results

We report **Precision**, **Recall**, and **F1 Score** on the MUStARD dataset. Below is a summary of our best performing model—**Multi-Level Hybrid Fusion**—versus popular baselines:

| Model             | Precision | Recall | F1    |
|-------------------|----------:|-------:|------:|
| Ning et al.       |     71.55 |  71.52 | 70.99 |
| MO-Sarcation      |     79.71 |  79.71 | 79.71 |
| MV-BART           |     80.56 |  82.86 | 81.69 |
| DyCR-Net          |     83.33 |  85.71 | 84.51 |
| **Hybrid Fusion** | **85.60** | **85.54** | **85.53** |

---


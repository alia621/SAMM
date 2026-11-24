# Modeling Multimodal Structures for Movie Genre Recognition

This repository provides the implementation of our proposed framework for multimodal movie genre classification, submitted to *Information Fusion*.

## Overview

The framework integrates visual, textual, and knowledge graph features for multi-label movie genre classification. It incorporates collaboration-aware knowledge graph representations, modality-specific mixture-of-experts, multi-view contrastive learning, and learnable genre clustering.

## Datasets

The experiments are conducted on MM-IMDb and MM-IMDb 2.0.

The MM-IMDb 2.0 dataset is available from the original IDKG project:

- **MM-IMDb 2.0:** [Download from Google Drive](https://drive.google.com/file/d/1fmU3ZKM3ieTDeTAeyK1uT9sFVvaGplIp/view?usp=sharing)
- **Original IDKG Repository:** https://github.com/aoluming/IDKG
- **embeddings**:https://drive.google.com/drive/folders/17w4daZvoF_1DqkLxGsGg_61xy6VAMshM?usp=drive_link

Please download the dataset from the original source and configure the corresponding data paths before running the code.

## Code

This repository contains the implementation of the proposed framework, including model architecture, training, and evaluation components.

## Citation

This work extends the IDKG framework published at ACM Multimedia 2023:

*Incorporating Domain Knowledge Graph into Multimodal Movie Genre Classification with Self-Supervised Attention and Contrastive Learning.*

DOI: https://doi.org/10.1145/3581783.3612085

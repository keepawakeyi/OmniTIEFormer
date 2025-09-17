## Overview
This repository contains supplementary materials for our paper titled "OmniTIEFormer: A Tri-Branch Transformer with Cross-Scale Transfer Learning for Multi-Scale Battery Life-Cycle Forecasting". The core code of OmniTIEFormer is not open-sourced at this time due to the unpublished status of the paper. However, we have provided the following resources:

## Contents
### 1. Autoformer Code
We have included the implementation of Autoformer, which was used as a baseline model in our experiments. This can be found in the `autoformer` directory.

### 2. Datasets
We have also made available two datasets used in our experiments:
- **Panasonic Dataset**: Located in the `data/panasonic` directory.
- **Gotion Dataset**: Located in the `data/gotion` directory.

These datasets can be used to reproduce the results of the baseline models and for further research purposes.

## How to Use
### Running Autoformer
To run the Autoformer code, please follow these steps:
1. Clone this repository.
2. Navigate to the `autoformer` directory.
3. Install the required dependencies (a `requirements.txt` file is provided).
4. Run the experiments using the provided scripts.
## Dependencies

The baseline experiments (e.g., Autoformer) were implemented using the following Python packages and versions. We recommend using a virtual environment (e.g., `conda` or `venv`) to ensure reproducibility.

| Package               | Version |
|-----------------------|---------|
| numpy                 | 1.21.6  |
| numba                 | 0.55.1  |
| matplotlib            | 3.3.4   |
| scipy                 | 1.8.0   |
| statsmodels           | 0.13.5  |
| pytorch-lightning     | 1.9.5   |
| pytorch-forecasting   | 0.10.3  |
| sympy                 | 1.12.1  |
| reformer_pytorch      | 1.4.4   |
| openpyxl              | 3.1.5   |
| einops                | 0.8.0   |
### Accessing the Datasets
The Panasonic and Gotion datasets are located in the `data` directory. You can use these datasets for your own experiments or to reproduce the results reported in our paper.

## Future Updates
Once our paper is published, we plan to release the full implementation of OmniTIEFormer along with detailed instructions on how to reproduce all the results presented in the paper.

## Contact
If you have any questions or need further assistance, please feel free to contact the authors via the email address provided in the paper.

---

This README provides a clear overview of what is currently available in the repository and sets expectations for future updates. It also ensures that users understand the current limitations and can still make use of the provided resources effectively.

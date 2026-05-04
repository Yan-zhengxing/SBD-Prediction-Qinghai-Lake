# Soil Bulk Density Prediction in Qinghai Lake Basin: A Machine Learning Approach with DSSCV

This repository contains the core implementation for predicting soil bulk density (SBD) in the **Qinghai Lake Basin**. It features a comparative analysis of three machine learning models and a robust spatial validation strategy tailored for high-altitude heterogeneous landscapes.

## 🌟 Key Features
- **Multi-Model Comparison**: Integration of Random Forest, XGBoost, and Cubist (via `rpy2`).
- **DSSCV Strategy**: Elevation-stratified (DEM-based) spatial cross-validation to mitigate spatial autocorrelation.
- **Hyperparameter Optimization**: Random search with 5-fold cross-validation for optimal model performance.
- **Reproducibility**: Pre-defined random seeds (Seed=42) and explicit environment configurations.
- **Interpretability**: Feature importance analysis for high-altitude soil environmental factors.

## 📂 Project Structure
```text
├── config.py               # Global configuration (parameters, paths, random seed)
├── utils.py                # Helper functions (data loading, evaluation, export)
├── spatial_cv.py           # Elevation-stratified K-means blocking (DSSCV)
├── rf_model.py             # Random Forest regression
├── xgb_model.py            # XGBoost regression
├── cubist_model.py         # Cubist rule-based regression (calls R 4.4.2 via rpy2)
├── main.py                 # Full pipeline execution
├── requirements.txt        # Detailed Python package versions
├── .gitignore              # Ignore temporary files and sensitive data
└── LICENSE                 # MIT License
🛠️ Requirements
Python: 3.9.12 (64-bit)

R: 4.4.2 (with Cubist package installed)

📦 Installation
Clone this repository to your local machine.

Install Python dependencies:
pip install -r requirements.txt
Usage
Data: Place your dataset and place it in the data/ folder.

Config: Modify model parameters and file paths in config.py.

Execute: Run the full prediction pipeline:
python main.py

## 📜 Citation
This repository is the official implementation of our research on soil bulk density prediction. 

**Citation information will be updated here once the paper is formally published.** For now, if you wish to refer to this methodology, please contact the author or check back later for the full reference.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

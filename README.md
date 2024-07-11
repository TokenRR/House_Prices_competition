# Housing price forecasting - Kaggle competition
This repository contains the code to participate in a Kaggle competition on predicting housing prices.  
Different machine learning models were used in the research to achieve better results

## Models used:
1. `LGBMRegressor`: Uses gradient boosting based on LightGBM
2. `XGBRegressor`: Uses gradient boosting based on XGBoost
3. `Ridge Regressor`: Linear regression with regularization using Ridge
4. `Support Vector Regressor (SVR)`: Uses the support vector method for regression
5. `Gradient Boosting Regressor`: Uses gradient boosting to improve model accuracy
6. `Random Forest Regressor`: Uses an ensemble of decision trees for prediction
7. `StackingCVRegressor`: A model that combines the predictions of several underlying models using cross-validation
8. `Blended Model`: Combining the predictions of different models to obtain a better generalized predictive power

## Repository contents
- `app/`: This directory contains the code for the windowed user interface program
- `data/`: The folder with the raw data and processed datasets
- `models/`: Saved models, after training
- `notebooks/`: A Jupyter notebook for data visualization, model experimentation, and analysis of results
- `requirements.txt`: List of required Python packages for installation

## Installation and use
#### Clone the repository:
```sh
git clone https://github.com/TokenRR/House_prices_competition.git
cd House_prices_competition
```

#### Install the required packages:
```sh
pip install -r requirements.txt
```

---
You can use the notebooks from the `notebooks/` folder to research and analyze the results.  
If you would like to contribute to this project, please create a **pull request** or open a new **issue**.
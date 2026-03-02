# Ames Housing Price Prediction

## Overview
End-to-end machine learning project predicting home sale prices using the Ames Housing dataset. The project covers data preprocessing, exploratory data analysis, feature engineering, and model comparison across multiple regression approaches. Results are translated into a business-facing dashboard showing what features drive home price, how closely the model tracks real sale prices, which modeling approach performed best, and where the most and least expensive neighborhoods are located.

---

## Dataset
Ames Housing dataset from Kaggle's House Prices competition. 1,460 training rows and 80 features covering structural, location, and quality attributes of homes sold in Ames, Iowa between 2006 and 2010.

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

---

## Project Structure
```
├── notebook.ipynb        # Full pipeline — preprocessing, EDA, modeling
└── model_dash.pbix       # Power BI dashboard
```

---

## Pipeline

### Preprocessing
- Dropped high null columns: Alley, PoolQC, MiscFeature, Fence
- Null imputation: None for absent features, 0 for absent areas, median for LotFrontage, mode for Electrical
- Ordinal encoding for quality and condition columns
- One-hot encoding for nominal categoricals
- Low variance, high multicollinearity, and low correlation features removed

### Feature Engineering
- TotalSF: combined basement, first, and second floor square footage
- TotalBath: combined all bathroom counts
- HouseAge: years from build to sale
- YearsSinceRemodel: years from remodel to sale
- HasGarage, HasFireplace, Has2ndFloor: binary flags

### Log Transformations
Applied log1p to skewed features: LotArea, LotFrontage, GrLivArea, TotalBsmtSF, 1stFlrSF, BsmtFinSF1, WoodDeckSF, OpenPorchSF, BsmtFinSF2, EnclosedPorch, SalePrice

---

## Models

| Model | CV RMSE | CV R2 | Kaggle Score |
|-------|---------|-------|--------------|
| Linear Regression | 0.1128 | 0.9195 | — |
| Ridge | 0.1125 | 0.9200 | — |
| Lasso | 0.1112 | 0.9218 | 0.12846 |
| XGBoost | 0.1205 | 0.9088 | 0.13000 |
| LightGBM | 0.1223 | 0.9062 | — |
| Ensemble (Lasso + XGBoost) | — | — | 0.12400 |

The ensemble of Lasso and XGBoost predictions achieved the best Kaggle score of 0.124.

---

## Key Findings
- Lasso outperformed gradient boosting methods on this dataset due to small sample size and strong linear signal from feature engineering
- TotalSF was the single most important feature by SHAP value, followed by ExterQual and KitchenQual
- NoRidge, NridgHt, and StoneBr were the highest value neighborhoods
- Ensembling Lasso and XGBoost by averaging predictions improved Kaggle score over either model alone

---

## Power BI Dashboard
Results translated into a business-facing dashboard showing: what features drive home price, how closely the model tracks real sale prices, which modeling approach performed best, and where the most and least expensive neighborhoods are located.

---

## Libraries
numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, shap

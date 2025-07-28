
# Walmart Sales Prediction Project

This project explores historical sales data from 45 Walmart stores across the U.S., with the goal of forecasting future weekly sales and understanding the factors that influence them.


## Project Objective

- Analyze weekly sales trends from 2010 to 2012  
- Build regression models to forecast future sales  
- Evaluate the impact of external factors like holidays, unemployment, and temperature


## Data Overview

- **Period:** 2010–2012  
- **Stores:** 45 Walmart locations  
- **Features:**  
  - Date  
  - Holiday events  
  - Temperature  
  - Fuel Price  
  - CPI (Consumer Price Index)  
  - Unemployment rate  
- **Target variable:** `Weekly_Sales`



##  EDA Highlights

- **2010** had the highest total sales, while **2012** had the lowest (fewer recorded weeks)  
- December consistently showed sales peaks — strong **holiday effect**  
- Sales vary significantly between stores  
- External economic indicators (CPI, Unemployment) had **weak correlation** with weekly sales

---

## Modeling Approach

### Linear Regression
- Trained on a cleaned dataset  
- Performance:  
  - R² = **0.96** on training data  
  - R² = **0.92** on test data  

###  Ridge Regression
- Reduced overfitting by shrinking large coefficients

###  Lasso Regression
- Improved model **interpretability**
- Automatically removed less impactful features by setting coefficients to zero

---

## Key Takeaways

- The models predict weekly sales with high accuracy  
- **Lasso regression** helps identify the most influential variables  
- Results are actionable for **inventory planning**, **seasonal promotions**, and **resource allocation**



##  Future Improvements

- Engineer new features: store type, competition, promotions  
- Explore non-linear models: **Random Forest**, **XGBoost**  
- Build a pipeline for **automated, real-time forecasting**

---

## 📁 File Structure

- `01-Walmart_sales.ipynb`: Full notebook (EDA + modeling)
- `Walmart_Store_sales.csv`: Original dataset
- `README.md`: Project documentation

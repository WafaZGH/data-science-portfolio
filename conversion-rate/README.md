# Conversion Rate Optimization Project


This project was part of my data science training. The goal was to build a model that predicts whether a website visitor would subscribe to a newsletter, based on a few behavioral features.



## The Challenge

We were given:
- `data_train.csv`: labeled data to train and validate our model  
- `data_test.csv`: unlabeled data for final predictions  


##  Context

The data came from **Data Science Weekly**, a popular newsletter platform.  
Their team wanted to explore:
- Predict who is likely to subscribe based on user behavior  
- Identify key features that influence subscription decisions 


The goal was to find insights and possibly boost their newsletter's **conversion rate**.



##  My Approach

- **Exploratory Data Analysis** to spot trends and clean issues  
- Data preprocessing: handling missing values, encoding, scaling  
- Tried multiple models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost

- **GridSearchCV** to find the best hyperparameters  
- Chose the best-performing model based on the **F1-score**, and generated predictions for submission



##  What I Learned

-- Simple models like logistic regression were easy to interpret but less accurate  
- **XGBoost** provided the best results and was used for the final submission  
- Feature engineering and preprocessing played a big role in model performance


##  Folder Contents

- `02-Conversion_rate_challenge final_version.ipynb`: Full notebook with EDA, modeling, and prediction  
- `data_train.csv`: Training data  
- `data_test.csv`: Testing data  
- `README.md`: This file


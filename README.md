# Housing Price Prediction - Exploratory Data Analysis and Predictive Model Creation
To view the notebook with visualizations, use this link: https://nbviewer.org/github/leeharry709/housing_price_prediction_model/blob/main/housing_price_exploration_and_prediction_model.ipynb

The purpose of this project was to practice common, yet influential, data exploration techniques as well as creating a predictive model complete with testing 4 different models, hyperparameter tuning, principal component analysis (PCA), feature engineering, and ensembling in order to find the best model for the data. The data used was from a Kaggle competition. After submitting, I got 1243 out of 4023 teams.

<p align="center">
  <img src="https://github.com/leeharry709/housing_price_prediction_model/blob/main/submission_placement.png?raw=true" width="75%">
</p>

## Explanation of Dataset
The dataset for the housing price prediction project consists of two main files: "train.csv" (the training set) and "test.csv" (the test set), along with a detailed data description in "data_description.txt." The dataset contains various features related to properties, including factors like building class, zoning classification, lot size, and property condition. The target variable for prediction is "SalePrice," representing the property's sale price in dollars. This dataset provides a comprehensive set of attributes to analyze and build predictive models for housing prices.

## Basic Data Exploration - Exploratory Data Analysis (EDA)
The data exploration was to understand the dataset, the differences between the categorical and numerical columns, and to understand how the metrics affect the target variable "SalePrice." In this section, we will explore the dependent variable as well as answer some questions one might have when looking at the dataset.

### Dependent Variable Exploration
Because the end goal was to create a predictive model using this data, I came into this phase thinking about what will need to be done: Normalization and reducing the impact of outliers. Not all datasets need normalization. But, a dataset such as housing information will require it because of scale. In this dataset, there are variables such as how many bathrooms there are (typically 1-3) vs. the sale price of a home (can be over 1 million). By normalizing the data, we will improve the final performance of the model.

### Research Questions
For this project, I had 8 questions that I visualized in the notebook using Plotly graphical objects:
<br>
<br><b>Q1. What is the distribution of dwelling types and their relation to Sale Price?</b>
<br><b>A1.</b> Single family homes (1Fam) are by far the most popular option for a home and also are the highest in price. Townhouse end units (TwnhsE) are the second most popular and cost similarly to single family homes. It is interesting to note that townhouse end units are far more expensive than standard townhouses.
        
<br><b>Q2. Does zoning impact Sale Price?</b>
<br><b>A2.</b> Zoning can grealy impact the sale price. Floating village residential (FV) are the most expensive, followed by residential low-density (RL). There are a few zones that are not listed, most likely due to not having a substantial amount of data.

<br><b>Q3. Does street and alley access types impact Sale Price?</b>
<br><b>A3.</b> Yes, paved streets and alleys are more expensive than unpaved streets and alleys.

<br><b>Q4. Does property shape and contour impact Sale Price?</b>
<br><b>A4.</b> Yes, a moderately irregular property shape is the most expensive, and the regular property shape is the least expensive. Hillside houses with significant slope from side to side are the most expensive property contour type.
  
<br><b>Q5. Is there a correlation between Property Age and Sale Price?</b>
<br><b>A5.</b> There is a negative correlation between Property Age and Sale Price. As Property Age goes up, Sale Price typically goes down.

<br><b>Q6. Is there a correlation between Living Area and Sale Price?</b>
<br><b>A6.</b> There is a strongly positive correlation between Living Area and Sale Price. As Living Area goes up, Sale Price typically goes up.

<br><b>Q7. Does price change year to year?</b>
<br><b>A7.</b> For the years 2006 to 2010, there was not a significant change in yearly average Sale Price of homes.

<br><b>Q8. What is the correlation between Sale Price and all numerical features?</b>
<br><b>A8.</b> Sale Price is most negatively correlated with kitchen above grade (ground) (KitchenAbvGr) and enclosed porch (EnclosedPorch). Sale Price is most positively correlated with living area above grade square footage (GrLivArea) and overall quality (OverallQual)

## Model Creation - Predictive Modeling
For experimentation, I used 4 different regression techniques: Linear Regression, Random Forest, Gradient Boosting via XGBoost, and Multi-Layer Perceptron (MLP). Some of the models work better for this project, while others do not. It is important to understand the strengths and weaknesses of each model while approaching the problem.
<br>
<br><b>Linear Regression</b> is a simple regression technique that is quick and easily interpretable and can test assumptions such as linearity and homoscedasticity, which describes the same variance between error terms across all values of the independent variables.
<br><b>Random Forest</b> is an ensemble method that can capture complex non-linear relationships between features and the target variable. It is able to handle categorical variables well.
<br><b>XGBoost (Extreme Gradient Boosting)</b> is known for its high predictive performance and efficiency, making it suitable for large datasets. It also ahs built-in support for handling missing data, assisting in final model performance.
<br><b>MLP (Multi-Layer Perceptron)</b> is a complex neural network-based regression technique that can automatically learn and extract relevant features from raw data, accommodating complex datasets.

### Data Preprocessing - Creating a Pipeline
For preprocessing the dataset, creating a pipeline greatly increased consistency by automating preprocessing steps such as missing value imputation, one-hot encoding for categorical variables, PCA component selection, and feature scaling. Creating a pipeline will also set up the project for hyperparameter tuning, and model comparison.

In this step, I implemented a data preprocessing pipeline to prepare the dataset for machine learning modeling. I defined separate transformers for numerical and categorical columns. For numerical data, I imputed missing values with the mean and scaled the data to ensure uniformity. For categorical data, I imputed missing values with a new category ('missing') and performed one-hot encoding to convert categorical variables into a binary format. I identified and separated the categorical and numerical columns in the dataset, and then combined these transformers using the ColumnTransformer. Finally, I created a comprehensive pipeline that applies these preprocessing steps to the dataset, preparing it for subsequent model training. Additionally, I normalized the dependent variable, 'SalePrice,' by taking its logarithm to address skewness observed during data exploration.

### Fitting and Hyperparameter Tuning Models
By fitting the models to the training data, I aimed to enable them to learn from the patterns in the data. Hyperparameter tuning allowed me to systematically search for the best combination of model settings, ensuring that each model performed at its highest potential, resulting in improved predictive accuracy and more reliable housing price predictions.

I split the data into training and testing sets to facilitate model evaluation. Then, I defined four regression models: Linear Regression, Random Forest, XGBoost, and MLP (Multi-Layer Perceptron), each with their respective hyperparameter grids. To assess model performance, I employed 3-fold cross-validation and utilized GridSearchCV to search for the best hyperparameters for each model. The results, including the best parameters and root mean squared error (RMSE) scores, were printed to the console. This process allows for the systematic comparison and selection of the most suitable regression model for the housing price prediction task.

### Principal Component Analysis (PCA)
Principal Component Analysis (PCA) reduces the dimensionality of the dataset. PCA helped me simplify the data by identifying the most important patterns and features while preserving as much variance as possible. This not only improved computational efficiency but also reduced the risk of overfitting, ultimately enhancing the accuracy and stability of the predictive models.

I started by fitting PCA to the preprocessed data and then calculated the cumulative explained variance to determine the optimal number of components to retain (in this case, enough to explain at least 95% of the variance). I integrated PCA into my data preprocessing pipeline and transformed the dataset accordingly. With the reduced-dimension dataset, I re-ran my regression models while performing hyperparameter tuning to optimize their performance. This approach aimed to improve computational efficiency and potentially enhance model accuracy by focusing on the most informative components of the data.

### Feature Engineering
Feature engineering is essential in machine learning because it allows you to create new, informative features from existing data, ultimately enhancing the performance of your models. By engineering features, you can capture relevant patterns and relationships that may not be apparent in the original data, leading to better predictive accuracy. Additionally, feature engineering can help mitigate issues like overfitting and improve the interpretability of models, making them more effective in solving complex real-world problems like housing price prediction.

After adding features to the dataset such as total square footage, total bathrooms, and month sold, I re-ran my regression models while performaning hyperparameter tuning using the PCA pipeline. Combining the PCA pipeline with feature engineering allows for the creation of more informative features while reducing dimensionality, resulting in improved model interpretability and predictive accuracy.

### Ensembling - Stacking Regression
By combining multiple diverse models such as Linear Regression, Random Forest, XGBoost, and MLP, we leverage their individual strengths and mitigate weaknesses. Stacking allows us to learn from the predictions of these base models, effectively capturing complex patterns and improving overall accuracy while reducing the risk of overfitting, thereby yielding more reliable and robust housing price predictions.

I implemented a StackingRegressor ensemble to combine the strengths of Linear Regression, Random Forest, XGBoost, and MLP.The goal was to create a powerful predictive model by leveraging the diverse strengths of these models, ultimately achieving improved housing price prediction accuracy. Finally, I evaluated the best stacking ensemble on the test data, providing an RMSE score as an indication of its predictive performance.

## Results
#### Normal Pipeline RMSE (without PCA or Feature Engineering):
1. **Linear Regression**: 482,591,986.50
   - The extremely high RMSE suggests poor performance for predicting housing prices with Linear Regression.
2. **RandomForest**: 0.1468
   - Reasonable performance with relatively low prediction error.
3. **XGBoost**: 0.1445
   - Good performance with predictions close to actual sale prices.
4. **MLP (Neural Network)**: 0.1480
   - Slightly higher RMSE but still reasonable for housing price prediction.  

#### PCA Pipeline RMSE (with PCA but without Feature Engineering):
1. **Linear Regression**: 0.1418
   - Improved performance after PCA but still relatively high RMSE.
2. **RandomForest**: 0.1525
   - Slight drop in performance after PCA.
3. **XGBoost**: 0.1453
   - Consistent performance after PCA.
4. **MLP (Neural Network)**: 0.1626
   - Significant performance drop after PCA.  

#### Feature Engineering and PCA Pipeline RMSE:
1. **Linear Regression**: 0.1425
   - Similar performance with feature engineering and PCA.
2. **RandomForest**: 0.1529
   - Minimal impact of feature engineering and PCA.
3. **XGBoost**: 0.1396
   - Improved performance after feature engineering and PCA.
4. **MLP (Neural Network)**: 0.1515
   - Improved performance with feature engineering and PCA.  

#### Stacking Ensemble with Feature Engineering and PCA Pipeline:
- Best RMSE: 0.1328
  - The stacking ensemble outperforms individual models and other pipelines, providing the most accurate predictions for housing prices (SalePrice).  

In summary, the stacking ensemble with feature engineering and PCA appears to be the most suitable choice for predicting housing prices (SalePrice). It outperforms individual models and other pipelines by providing the lowest RMSE, suggesting that it can make more accurate predictions for this specific task.

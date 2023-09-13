# Housing Price Prediction - Exploratory Data Analysis and Predictive Model Creation
The purpose of this project was to practice common, yet influential, data exploration techniques as well as creating a predictive model complete with testing 4 different models, hyperparameter tuning, principal component analysis (PCA), feature engineering, and ensembling in order to find the best model for the data. The data used was from a Kaggle competition. After submitting, I got 1243 out of 4023 teams.

To view the notebook with visualizations, use this link: https://nbviewer.org/github/leeharry709/housing_price_prediction_model/blob/main/housing_price_exploration_and_prediction_model.ipynb

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

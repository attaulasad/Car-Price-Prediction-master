import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
import pickle

# Load the data
df = pd.read_csv('car data.csv')

# Check the shape of the dataframe
print(df.shape)

# Display unique values in certain columns
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())

# Check for missing values
print(df.isnull().sum())

# Describe the dataset
print(df.describe())

# Create the final dataset with selected features
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
                    'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]

# Add a new column for the current year
final_dataset['Current Year'] = 2020

# Calculate the number of years the car has been used
final_dataset['no_year'] = final_dataset['Current Year'] - final_dataset['Year']

# Drop the 'Year' and 'Current Year' columns
final_dataset.drop(['Year', 'Current Year'], axis=1, inplace=True)

# Convert categorical variables into dummy/indicator variables
final_dataset = pd.get_dummies(final_dataset, drop_first=True)

# Visualize the correlation matrix
plt.figure(figsize=(20,20))
sns.heatmap(final_dataset.corr(), annot=True, cmap="RdYlGn")
plt.show()

# Define independent variables (X) and dependent variable (y)
X = final_dataset.iloc[:, 1:]
y = final_dataset.iloc[:, 0]

# Use ExtraTreesRegressor to identify important features
model = ExtraTreesRegressor()
model.fit(X, y)

# Plot the top 5 important features
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the hyperparameter grid for RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Instantiate RandomForestRegressor
rf = RandomForestRegressor()

# Randomized Search CV
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, 
                               scoring='neg_mean_squared_error', n_iter=10, cv=5, 
                               verbose=2, random_state=42, n_jobs=1)

# Fit the model
rf_random.fit(X_train, y_train)

# Print best parameters and score
print(rf_random.best_params_)
print(rf_random.best_score_)

# Predict on the test set
predictions = rf_random.predict(X_test)

# Visualize the residuals
sns.distplot(y_test - predictions)
plt.show()

plt.scatter(y_test, predictions)
plt.show()

# Print evaluation metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Save the model using pickle
with open('random_forest_regression_model.pkl', 'wb') as file:
    pickle.dump(rf_random, file)

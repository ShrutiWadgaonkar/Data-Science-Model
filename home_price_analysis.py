import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
home_prices = pd.read_csv('CSUSHPISA.csv')
interest_rates = pd.read_csv('DFF.csv')
gdp_rates = pd.read_csv('GDP.csv')
unemployment_rates = pd.read_csv('UNRATE.csv')

# Rename 'observation_date' to 'Date' in all datasets
home_prices.rename(columns={'observation_date': 'Date'}, inplace=True)
interest_rates.rename(columns={'observation_date': 'Date'}, inplace=True)
gdp_rates.rename(columns={'observation_date': 'Date'}, inplace=True)
unemployment_rates.rename(columns={'observation_date': 'Date'}, inplace=True)

# Merge datasets on 'Date'
combined_data = pd.merge(home_prices, interest_rates, on='Date', how='inner')
combined_data = pd.merge(combined_data, gdp_rates, on='Date', how='inner')
combined_data = pd.merge(combined_data, unemployment_rates, on='Date', how='inner')

# Rename columns for clarity
combined_data.rename(columns={'CSUSHPISA': 'HomePriceIndex', 'DFF': 'InterestRate', 
                              'UNRATE': 'UnemploymentRate', 'GDP': 'GDP'}, inplace=True)

# Handle missing values
combined_data.ffill(inplace=True)

# Convert 'Date' to datetime format for better handling
combined_data['Date'] = pd.to_datetime(combined_data['Date'])

# Filter data for the last 20 years
combined_data = combined_data[combined_data['Date'] >= '2004-01-01']

# Plot home prices over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=combined_data, x='Date', y='HomePriceIndex')
plt.title('Home Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Home Price Index')
plt.xticks(rotation=45)
plt.show()

# Print summary for home prices
print("Summary: Over the last 20 years, home prices have shown significant variation. "
      "There was a notable dip during the 2008 financial crisis, followed by a steady recovery and growth over time.")

# Correlation Matrix
correlation_matrix = combined_data[['HomePriceIndex', 'InterestRate', 'UnemploymentRate', 'GDP']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Print correlation insights
print("\nCorrelation Insights:")
print("1. Interest rates and home prices often show a negative correlation, as lower rates make borrowing cheaper.\n"
      "2. GDP positively correlates with home prices, indicating economic growth supports higher prices.\n"
      "3. Higher unemployment rates may depress home prices due to reduced demand.")

# Define features and target variable
X = combined_data[['InterestRate', 'UnemploymentRate', 'GDP']]
y = combined_data['HomePriceIndex']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model performance
print("\nModel Summary:")
print(f"The linear regression model achieved an R-squared of {r2:.2f}, indicating that approximately {r2*100:.2f}% "
      "of the variance in home prices is explained by the model. "
      f"The mean squared error (MSE) of the model is {mse:.2f}, showing the average squared difference between predicted "
      "and actual values.")

# Display feature importance (coefficients)
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nFeature Importance:")
print(coefficients.to_string(index=False))

# Plot actual vs predicted home prices
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Home Prices')
plt.xlabel('Actual Home Prices')
plt.ylabel('Predicted Home Prices')
plt.show()

# Summary of actual vs predicted
print("\nSummary of Predictions:")
print("The scatter plot shows the relationship between actual and predicted home prices. "
      "A strong alignment along the red diagonal line indicates a well-performing model.")

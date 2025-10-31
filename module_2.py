import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor,plot_tree
from scipy.stats import pearsonr
from scipy.stats import ttest_ind, f_oneway
#from sklearn.tree import plot_tree
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Load the training and testing datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
predict_df = pd.read_csv("predict.csv")


# Select the three most important variables for descriptive statistics
selected_features = ['population_density', 'ticket_price', 'distance_km']

# Descriptive Statistics for the selected variables
train_desc = train_df[selected_features].describe()

# Hypothesis 4: Combined effect of ticket price, distance, and population density (ANOVA)
f_stat, p_value_anova = f_oneway(train_df['ticket_price'], train_df['distance_km'], train_df['population_density'])
print(f"\n🔬 Hypothesis Test: Joint Effect of Ticket Price, Distance, and Population Density (ANOVA)\nF-statistic: {f_stat:.4f}, P-value: {p_value_anova:.4f}")

if p_value_anova < 0.05:
    print("✅ Statistical Conclusion: We reject H0, Ticket prices, travel distance, and socioeconomic factors have little impact on passenger flow.")
else:
    print("❌ Statistical Conclusion: We fail to reject H1, Ticket prices, travel distance, and socioeconomic factors greatly influence passenger flow.")

# Hypothesis 1: Impact of Ticket Prices
# Null Hypothesis (H₀₁): Ticket prices do not have a significant effect on passenger flow.
# Alternative Hypothesis (H₁₁): Ticket prices have a significant effect on passenger flow.

f_stat_ticket, p_value_ticket = f_oneway(train_df['ticket_price'], train_df['passenger_count'])
print(f"\n🔬 Hypothesis Test: Impact of Ticket Prices\nF-statistic: {f_stat_ticket:.4f}, P-value: {p_value_ticket:.4f}")

if p_value_ticket < 0.05:
    print("✅ Reject H0: Ticket prices do not have a significant effect on passenger flow.")
else:
    print("❌ Fail to reject H1: Ticket prices have a significant effect on passenger flow.")

# Hypothesis 2: Impact of Travel Distance
# Null Hypothesis (H₀₂): Travel distance does not have a significant effect on passenger flow.
# Alternative Hypothesis (H₁₂): Travel distance has a significant effect on passenger flow.

f_stat_distance, p_value_distance = f_oneway(train_df['distance_km'], train_df['passenger_count'])
print(f"\n🔬 Hypothesis Test: Impact of Travel Distance\nF-statistic: {f_stat_distance:.4f}, P-value: {p_value_distance:.4f}")

if p_value_distance < 0.05:
    print("✅ Reject H0: Travel distance does not has a significant effect on passenger flow.")
else:
    print("❌ Fail to reject H1: Travel distance have a significant effect on passenger flow.")

# Hypothesis 3: Impact of Socioeconomic Factors
# Null Hypothesis (H₀₃): Socioeconomic factors do not have a significant effect on passenger flow.
# Alternative Hypothesis (H₁₃): Socioeconomic factors have a significant effect on passenger flow.

f_stat_socio, p_value_socio = f_oneway(train_df['population_density'], train_df['passenger_count'])
print(f"\n🔬 Hypothesis Test: Impact of Socioeconomic Factors\nF-statistic: {f_stat_socio:.4f}, P-value: {p_value_socio:.4f}")

if p_value_socio < 0.05:
    print("✅ Reject H0: Socioeconomic factors do not have a significant effect on passenger flow.")
else:
    print("❌ Fail to reject H1: Socioeconomic factors  have a significant effect on passenger flow.")

# Visualization of the ANOVA Results
# Create a bar chart to visualize the p-values for the three factors
p_values = [p_value_ticket, p_value_distance, p_value_socio]
factors = ['Ticket Prices', 'Travel Distance', 'Socioeconomic Factors']

plt.figure(figsize=(10, 6))
plt.bar(factors, p_values, color=['blue', 'green', 'orange'])
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Level (0.05)')
plt.xlabel('Factors')
plt.ylabel('P-value')
plt.title('ANOVA Test: Impact of Different Factors on Passenger Flow')
plt.legend()
plt.show()

# Visualization of Descriptive Statistics (Mean, Std, Min, Max)
plt.figure(figsize=(10, 5))
train_desc.loc[['mean', 'std', 'min', 'max']].T.plot(kind='bar', colormap='coolwarm', alpha=0.75)
plt.title("Descriptive Statistics (Mean, Std, Min, Max) for Selected Features")
plt.ylabel("Values")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Get all possible columns from train, test, and predict datasets
all_columns = set(train_df.columns) | set(test_df.columns) | set(predict_df.columns)

# Ensure all datasets have the same columns
train_df = train_df.reindex(columns=all_columns, fill_value=0)
test_df = test_df.reindex(columns=all_columns, fill_value=0)
predict_df = predict_df.reindex(columns=all_columns, fill_value=0)

# Define feature columns
features = ['distance_km', 'ticket_price', 'population_density', 'avg_income', 'daily_transport',
            'metro_usage', 'primary_metro_use', 'off_peak_pricing', 'new_metro_use', 'fare_impact',
            'faster_commute', 'last_mile_connectivity', 'more_stations', 'better_amenities',
            'workplace_connection']

# One-Hot Encode categorical variables
train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)
predict_df = pd.get_dummies(predict_df, drop_first=True)

# Define Independent (X) and Dependent (y) Variables
X_train = train_df.drop(columns=['passenger_count'])
y_train = train_df['passenger_count']
X_test = test_df.drop(columns=['passenger_count'])
y_test = test_df['passenger_count']

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predict using both models
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Hybrid Model: Weighted Averaging
w_rf = 0.5   # Weight for Random Forest
w_xgb = 0.5  # Weight for XGBoost
y_pred_hybrid = (w_rf * y_pred_rf) + (w_xgb * y_pred_xgb)

# Evaluate Hybrid Model Performance
mae_hybrid = mean_absolute_error(y_test, y_pred_hybrid)
mse_hybrid = mean_squared_error(y_test, y_pred_hybrid)
rmse_hybrid = np.sqrt(mse_hybrid)
r2_hybrid = r2_score(y_test, y_pred_hybrid)

print("\n📊 Hybrid Model Performance (Random Forest + XGBoost):")
print(f"  - MAE: {mae_hybrid:.2f}")
print(f"  - MSE: {mse_hybrid:.2f}")
print(f"  - RMSE: {rmse_hybrid:.2f}")
print(f"  - R² Score: {r2_hybrid:.2f}")

# Hypothesis Testing
alpha = 0.05  # 5% significance level

if r2_hybrid > 0.1:
    print("\n✅ Reject the Null Hypothesis (H₀): Ticket pricing, distance, and socioeconomic factors significantly impact passenger flow.")
else:
    print("\n❌ Fail to Reject the Null Hypothesis (H₀): Ticket pricing, distance, and socioeconomic factors have minimal impact on passenger flow.")

# Time Series Forecasting for Passenger Flow in 2026
monthly_passenger_data = {
    'year': [2023] * 12 + [2024] * 12 + [2025] * 2,
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
              'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
              'jan', 'feb'],
    'passenger_count': [
        6607458, 6379282, 6999341, 6719510, 7299093, 7394696, 8254561, 8590212, 8437182, 8550435,
        8001700, 8123281, 8466310, 8618114, 8682459, 8087712, 8421620, 8433837, 9535019, 9552680, 9278183, 9024220,
        8361492, 8738393,
        8699359, 8665803
    ]
}

# Convert data into DataFrame
df_time_series = pd.DataFrame(monthly_passenger_data)

# Create a 'date' column combining 'year' and 'month'
df_time_series['date'] = pd.to_datetime(df_time_series['month'] + '-' + df_time_series['year'].astype(str),
                                        format='%b-%Y')
df_time_series.set_index('date', inplace=True)
df_time_series = df_time_series.drop(columns=['year', 'month'])

# Create lag features for time series forecasting
df_time_series['lag_1'] = df_time_series['passenger_count'].shift(1)
df_time_series['lag_2'] = df_time_series['passenger_count'].shift(2)

# Drop missing values caused by lagging
df_time_series = df_time_series.dropna()

# Prepare the features and target for model training
X_time_series = df_time_series[['lag_1', 'lag_2']]
y_time_series = df_time_series['passenger_count']

# Train the XGBoost model for time series
model_ts = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
model_ts.fit(X_time_series, y_time_series)

# Forecasting for 2026 (12 months)
last_2_months = df_time_series.iloc[-2:][['passenger_count']]
forecast_2026 = []

for i in range(12):  # Forecast for each month in 2026
    lag_1 = last_2_months.iloc[-1, 0]
    lag_2 = last_2_months.iloc[-2, 0]
    X_new = np.array([[lag_1, lag_2]])
    y_new = model_ts.predict(X_new)[0]
    forecast_2026.append(y_new)

    # Update last 2 months
    new_data = pd.DataFrame({'passenger_count': [y_new]})
    last_2_months = pd.concat([last_2_months, new_data], ignore_index=True).tail(2)

# Create a forecast dataframe for 2026
forecast_2026_df = pd.DataFrame({
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'forecasted_passenger_count': forecast_2026
})

# Convert forecasted months to proper datetime
forecast_2026_df['date'] = pd.to_datetime(forecast_2026_df['month'] + '-2026', format='%b-%Y')

# Visualization of Forecast vs Actual Data
plt.figure(figsize=(12, 6))
plt.plot(df_time_series.index, df_time_series['passenger_count'], label='Actual Data (2023-2024)', color='blue')
plt.plot(forecast_2026_df['date'], forecast_2026_df['forecasted_passenger_count'], label='Forecasted Data (2026)',
         color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.title('Passenger Flow Forecast for 2026')
plt.legend()
plt.grid(True)
plt.show()

# Scatter Plot - Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_hybrid, alpha=0.6, color='blue')  # Scatter plot of actual vs predicted values
plt.plot(y_test, y_test, color='red', linestyle='--')  # Perfect prediction line (y = x)
plt.xlabel("Actual Passenger Count")
plt.ylabel("Predicted Passenger Count")
plt.title("Actual vs. Predicted Passenger Flow (Hybrid Model)")
plt.show()


# Predict Passenger Flow for New Metro Lines
predict_df = predict_df.reindex(columns=X_train.columns, fill_value=0)
predict_df = predict_df.iloc[:3]
predict_df_scaled = scaler.transform(predict_df)
predicted_passengers = xgb_model.predict(predict_df_scaled)

forecast_data = {
    'Metro Line': ['Purple Line', 'Orange Line', 'Red Line'],
    'Predicted Passenger Flow': predicted_passengers
}
# Create a forecast dataframe for 2026
forecast_2026_df = pd.DataFrame({
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'forecasted_passenger_count': forecast_2026
})

# Display forecast for 2026
print("\n📊 Forecasted Passenger Flow for 2026:")
print(forecast_2026_df)

forecast_df = pd.DataFrame(forecast_data)
print(forecast_df)

# Visualization of Predicted Passenger Flow for New Metro Lines
plt.figure(figsize=(10, 6))
sns.barplot(x='Metro Line', y='Predicted Passenger Flow', data=forecast_df, hue='Metro Line', legend=False)
plt.title('Predicted Passenger Flow for New Metro Lines')
plt.ylabel('Predicted Passenger Flow')
plt.xlabel('Metro Line')
plt.show()

# Convert daily predictions to monthly ridership
predicted_passengers_monthly = predicted_passengers * 30  # Assuming 30 days in a month

# Sum up the total ridership for all three lines
total_ridership_monthly = np.sum(predicted_passengers_monthly)

# Create a new forecast dataframe with total ridership
forecast_data = {
    'Metro Line': ['Purple Line', 'Orange Line', 'Red Line'],
    'Daily Predicted Passenger Flow': predicted_passengers,
    'Monthly Predicted Passenger Flow': predicted_passengers_monthly
}

forecast_df = pd.DataFrame(forecast_data)

# Print individual and total ridership
print("\n📊 Forecasted Passenger Flow for Each Metro Line (2026):")
print(forecast_df)

print(f"\n🚆 Total Forecasted Monthly Ridership for All Three Lines: {total_ridership_monthly:,.0f} passengers")

# Visualization of Predicted Passenger Flow for New Metro Lines
plt.figure(figsize=(10, 6))
sns.barplot(x='Metro Line', y='Monthly Predicted Passenger Flow', data=forecast_df, hue='Metro Line', legend=False)
plt.title('Predicted Monthly Passenger Flow for New Metro Lines')
plt.ylabel('Monthly Predicted Passenger Flow')
plt.xlabel('Metro Line')
plt.show()

# Feature Importance
feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Extract only the required variables
selected_features = ['ticket_price', 'distance_km', 'population_density']
importance_df_selected = importance_df[importance_df['Feature'].isin(selected_features)].copy()

# Normalize importance to percentage
total_importance = importance_df_selected['Importance'].sum()
importance_df_selected['Percentage Impact'] = (importance_df_selected['Importance'] / total_importance) * 100

# Print percentage impact
print("\n📊 Percentage Impact of Key Variables on Passenger Flow:")
print(importance_df_selected[['Feature', 'Percentage Impact']])

# Plot feature importance for selected variables
plt.figure(figsize=(10, 5))
sns.barplot(x='Percentage Impact', y='Feature', data=importance_df_selected, hue='Feature', palette='coolwarm', legend=False)
plt.title('Impact of Ticket Price, Distance, and Population Density on Passenger Flow')
plt.xlabel('Percentage Impact (%)')
plt.ylabel('Feature')
plt.xlim(0, 100)
plt.show()

'''# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
rf.fit(X_train, y_train)

# Plot one of the trees from the forest
plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], feature_names=X_train.columns, filled=True, rounded=True)
plt.show()'''


# Visualizing an individual tree from the XGBoost model
plt.figure(figsize=(20, 10))
plot_tree(xgb_model, num_trees=0, rankdir='LR')  # Change num_trees to visualize different trees
plt.title("XGBoost Decision Tree Visualization")
plt.show()

# Model performance metrics
metrics = ['MAE', 'MSE', 'RMSE', 'R² Score']
values = [923.12, 1198366.58, 1094.70, 0.96]

# Convert MSE to log scale to balance visualization
log_values = [np.log10(v) if v > 1 else v for v in values]

# Custom colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure(figsize=(8, 5))
sns.barplot(x=metrics, y=log_values, hue=metrics, palette=colors, legend=False)


# Convert log values back to readable format for display
for i, v in enumerate(log_values):
    plt.text(i, v + 0.05, f"{values[i]:,.2f}", ha='center', fontsize=10, fontweight='bold')

# Labels & Title
plt.ylabel('Log Scale (Base 10)')
plt.xlabel('Performance Metrics')
plt.title('Hybrid Model Performance (Random Forest + XGBoost)')

plt.show()



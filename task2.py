import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import plotly.express as px

# Importing the dataset
# Start from 1 Aug 2021 (first record of vaccination rate)
df = pd.read_excel('Covid-19 SG.xlsx', skiprows=range(1, 557))

st.title("Predicting Singapore's Straits Times Index (STI) growth amidst COVID-19 using Multiple Linear Regression")



# Processing with:
# - Date
# - Still Hospitalised
# - Phase
# - 7 days Moving Average
# - Percentage Vaccinated
sg_population = 5.686 * 10**6

# Processing NaN values
df = df[pd.notnull(df['Phase'])]
df['Cumulative Individuals Vaccinated'] = df['Cumulative Individuals Vaccinated'].fillna(0) # drop na if we want to remove vacine
df['Percentage Vaccinated'] = df['Cumulative Individuals Vaccinated'].divide(sg_population)


# # Convert to Date time object for easier processing
# df['Date'] = pd.to_datetime(df['Date'])

# # Convert Date to numerical value
# df['Date'] = df['Date'].map(dt.datetime.toordinal)

# Find 7 days Moving Average as another feature
df['7 days Moving Average'] = df['Daily Confirmed'].rolling(window=7).mean()

# Replace NaN values
df['7 days Moving Average'].fillna(df['Daily Confirmed'], inplace=True)

# Display table
initial_columns = ['Date', 'Still Hospitalised','Phase','7 days Moving Average', 'Percentage Vaccinated']
df = df[initial_columns]
st.write("Data we will be using:")
st.write(df)


# Convert to Date time object for easier processing
df['Date'] = pd.to_datetime(df['Date'])

# Convert Date to numerical value
df['Date'] = df['Date'].map(dt.datetime.toordinal)

# Getting useful columns
new_columns = ['Date', 'Still Hospitalised','Phase','7 days Moving Average', 'Percentage Vaccinated']
df = df.reindex(columns=new_columns)

df = df[new_columns]

# st.write("Table we will be using:")
# st.write(df)



# Preprocessing SGX Data for Y Axis
sgx_df = pd.read_csv('./HistoricalPrices.csv', skipfooter=43)

sgx_df['Date'] = pd.to_datetime(sgx_df['Date'])
sgx_df['Date'] = sgx_df['Date'].map(dt.datetime.toordinal)
sgx_df = sgx_df.rename(columns={' Open':'STI Price'})

# Narrowing down to just open instead of high, low and close
sgx_df = sgx_df[['Date','STI Price']]

# Merge with dataset on Date
merged_df = pd.merge(df,sgx_df, how='inner', on='Date')




X = merged_df.iloc[:, :-1].values # selects all the columns excluding STI price
Y = merged_df.iloc[:, -1].values # STI price column


##### Line graph #####
dataframe = merged_df[['Date', 'Still Hospitalised', '7 days Moving Average', 'Percentage Vaccinated', 'STI Price']]
dataframe = dataframe.set_index('Date')
# Line chart
st.write("Line chart:")
st.line_chart(dataframe)
# st.altair_chart(dataframe)



# Encoding categorical data which is the phase
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2]) # phase column
ct = ColumnTransformer([("Phase", OneHotEncoder(), [2])], remainder="passthrough")
X = ct.fit_transform(X)



# Avoiding the Dummy Variable Trap (dummy variables: binary variables for categorical data)
X = X[:, 1:] # avoid one of the dummy variables

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear Regression in the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)
# print(regressor.intercept_)
# print(regressor.coef_)

# Predicting the Test set results
Y_Pred = regressor.predict(X_Test)



##### OPTIMISATION #####

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((59, 1)).astype(int), values = X, axis = 1)

X_Optimal = X[:, [0,1,2,3,4,5]]
X_Optimal = np.array(X_Optimal, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
regressor_OLS.summary()

X_Optimal = X[:, [0,1,2,4,5]]
X_Optimal = np.array(X_Optimal, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
regressor_OLS.summary()

X_Optimal = X[:, [0,1,4,5]]
X_Optimal = np.array(X_Optimal, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
regressor_OLS.summary()

X_Optimal = X[:, [0,1,4]]
X_Optimal = np.array(X_Optimal, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
regressor_OLS.summary()

# Fitting the Multiple Linear Regression in the Optimal Training set
X_Optimal_Train, X_Optimal_Test = train_test_split(X_Optimal,test_size = 0.2, random_state = 0)
regressor.fit(X_Optimal_Train, Y_Train)

# Predicting the Optimal Test set results
Y_Optimal_Pred = regressor.predict(X_Optimal_Test)



##### EVALUATING THE MODEL #####
# optimized with bw elimation
X_Optimal = X[:, [0,1,2,3,4,5]]
X_Optimal = np.array(X_Optimal, dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
# print(regressor_OLS.summary())



# # importing r2_score module
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error


# # predicting the accuracy score
# score=r2_score(Y_Test,Y_Pred)
# print(f"r2 score is {score}")
# print(f"mean_sqrd_error is == {mean_squared_error(Y_Test,Y_Pred)}")
# print(f"root_mean_squared error of is == {np.sqrt(mean_squared_error(Y_Test,Y_Pred))}")


# # After Optimization with BE
# print("============================")
# print("After Optimization with Backwards Elimination")
# # predicting the accuracy score
# score=r2_score(Y_Test,Y_Optimal_Pred)
# print(f"r2 score is {score}")
# print(f"mean_sqrd_error is == {mean_squared_error(Y_Test,Y_Pred)}")
# print(f"root_mean_squared error of is == {np.sqrt(mean_squared_error(Y_Test,Y_Pred))}")
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime

# Get the stock symbol from the user
symbol = input("Enter the stock symbol: ")

# Get the current date
now = datetime.datetime.now().strftime("%Y-%m-%d")

# Download the stock data from Yahoo Finance
data = yf.download(symbol, start="2010-01-01", end=now)

# Print the head of the data
print(data.head())

# Plot the close price over time
plt.plot(data['Close'])
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.title(symbol + ' Close Price Over Time')
plt.show()

# Plot the volume over time
plt.plot(data['Volume'])
plt.xlabel('Year')
plt.ylabel('Volume')
plt.title(symbol + ' Volume Over Time')
plt.show()

# Split the data into training and testing sets
train_data = data[:'2020-01-01']
test_data = data['2020-01-01':]

# Train a linear regression model on the training data
X_train = train_data.index.values.reshape(-1, 1)
y_train = train_data['Close'].values.reshape(-1, 1)
reg = LinearRegression().fit(X_train, y_train)

# Make predictions on the test data
X_test = test_data.index.values.reshape(-1, 1)
predictions = reg.predict(X_test)

# Plot the predictions
plt.plot(test_data.index, predictions, label='Prediction')
plt.plot(test_data.index, test_data['Close'], label='Actual')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.title(symbol + ' Close Price Prediction')
plt.legend()
plt.show()

# Calculate the highest close price in the data
highest_price = data['Close'].max()
print(f"The highest close price in the data is {highest_price:.2f}")

# Calculate the lowest close price in the data
lowest_price = data['Close'].min()
print(f"The lowest close price in the data is {lowest_price:.2f}")

# Calculate the percentage increase or decrease in the close price over the years
percent_change = (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100
print(f"The close price has changed by {percent_change:.2f}% over the years")
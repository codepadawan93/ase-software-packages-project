import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

# Declare a function to get the rolling statistics


def get_trend(y, window):
    rollmean = y.rolling(window=window, center=False).mean()
    rollstd = y.rolling(window=window, center=False).std()
    return rollmean, rollstd


# Read in a CSV file containing site usage data for company (time series, 90 days)
raw_df = pd.read_csv('./resources/data.csv')

# Check data format
print(raw_df.head())

# Copy raw data to useable format
df = raw_df[['Visitors', 'Bounce Rate', 'Cart Abandon Rate']]
df.set_index(pd.to_datetime(raw_df['Date'],
                            infer_datetime_format=True), inplace=True)

# Check format again, then plot one metric
print(df.head())
plt.plot(df['Visitors'], color='blue', label='Daily visitors')
plt.legend(loc='best')
plt.show()

# Plot the other metrics too
plt.plot(df['Bounce Rate'], color='blue', label='Bounce rate')
plt.plot(df['Cart Abandon Rate'], color='red', label='Cart Abandon Rate')
plt.legend(loc='best')
plt.show()

# Show trends (rolling avg, rolling std dev) over several windows: 1 week, 2 weeks, 1 month
windows = [7, 14, 30]
for window in windows:
    trendline = get_trend(df['Visitors'], window=window)
    plt.plot(df['Visitors'], color='blue', label='Visitors (original)')
    plt.plot(trendline[0], color='red', label=f'Trend {window} days (avg)')
    plt.plot(trendline[1], color='black',
             label=f'Trend {window} days (std. dev.)')
    plt.legend(loc='best')
    plt.show()

# Create a linear regression model check if the bounce
# rate is related to the cart abandon rate using a OLS
# regression model
model = sm.OLS(df['Bounce Rate'], df['Cart Abandon Rate'])
res = model.fit()
print(res.summary())

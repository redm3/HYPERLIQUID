import pandas as pd
import datetime
import time

# Define the URL for fetching recent price data
symbol = "BTC-USD"
url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?interval=1d&events=history&includeAdjustedClose=true"

# Define the path to the CSV file
csv_path = 'C:/Users/macmw/Documents/GitHub/HyperLiquid/BTC-USD-actual.csv'

# Define the time of day to run the script (in 24-hour format)
#run_time = "00:00:00"  # set this to the desired time

# Continuously loop until the script is interrupted
#while True:
    # Get the current time
    #current_time = datetime.datetime.now().strftime("%H:%M:%S")

    # Check if it's time to run the script
    #if current_time == run_time:
        # Read recent price data from the URL
df = pd.read_csv(url)

        # Append the most recent row to the existing file
df.tail(1).to_csv(csv_path, header=False, index=False, mode='a')

        # Print a message indicating that the script has run
        #print(f"Data updated at {current_time}.")

        # Wait for 24 hours before running the script again
        #time.sleep(24 * 60 * 60)
    #else:
        # Print a message indicating that the script is waiting
        #print(f"Waiting for {run_time} to update the script data.")

        # Wait for 60 seconds before checking the time again
        #time.sleep(60)
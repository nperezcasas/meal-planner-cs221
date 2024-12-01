import pandas as pd
import re

# Load the CSV file with the correct separator
data_df = pd.read_csv('100_ingredient_costs_fromChatGPT.csv', 
                      sep=',',
                      names=['ingredient', 'price_range'])

# Initialize lists to store the parsed data
names = []
units = []
average_prices = []

# Function to calculate average price
def calculate_average_price(price_range):
    if price_range.lower() == 'free':
        return 0.0
    prices = price_range.replace('$', '').split(' - ')
    return (float(prices[0]) + float(prices[1])) / 2

# Parse each row
for _, row in data_df.iterrows():
    # Extract name and unit using regex
    match = re.match(r'^(.*?)\s*\((.*?)\)$', row['ingredient'])
    if match:
        name = match.group(1).strip()
        unit = match.group(2).strip()
    else:
        name = row['ingredient'].strip()
        unit = ''
    
    average_price = calculate_average_price(row['price_range'])
    
    names.append(name)
    units.append(unit)
    average_prices.append(average_price)

# Create a DataFrame
result_df = pd.DataFrame({
    'Name': names,
    'Unit': units,
    'Avg Cost': average_prices
})

# Save the DataFrame to a CSV file
result_df.to_csv('100_ingredient_costs.csv', index=False)

# Display the DataFrame
print(result_df.head())
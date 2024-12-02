import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = {
    'Total Cost': [49.5, 49.6, 48.8, 48.81, 50.0, 49.31, 49.86, 47.85, 49.51, 49.1, 49.72, 49.65, 28.52, 30.94, 14.31],
    'Total Calories': [2050.6, 5057.1, 760.6, 733.8, 1813.1, 2596.8, 7803.2, 4228.0, 4347.9, 9101.1, 5628.8, 3556.7, 3213.5, 3188.2, 2349.6],
    'Total Protein (g)': [67.0, 209.4, 13.5, 17.8, 33.2, 58.6, 150.4, 76.3, 232.9, 97.7, 123.5, 62.3, 104.9, 93.4, 77.2],
    'Cost Efficiency (%)': [99.0, 99.2, 97.6, 97.62, 100.0, 98.62, 99.72, 95.7, 99.02, 98.2, 99.44, 99.3, 28.52, 30.94, 14.31],
    'Nutritional Balance (%)': [21.84380952380953, 60.59269841269842, 6.6171428571428565, 7.624126984126985, 16.857142857142858, 27.320317460317465,
                                79.58253968253968, 38.37746031746031, 60.31714285714285, 85.89555555555556, 52.36444444444444, 28.685714285714283, 
                                180.44057539682535, 179.01996031746032, 131.93190476190475],
    'Recipe Variety (%)': [79.3103448275862, 100.0, 88.57142857142857, 85.71428571428571, 85.71428571428571, 82.14285714285714, 
                            89.28571428571429, 92.5925925925926, 92.0, 65.38461538461539, 88.88888888888889, 84.84848484848484, 
                            77.77777777777779, 81.81818181818183, 73.33333333333333],
    'source': ['baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'meal_planner', 'meal_planner', 'meal_planner'],
    'timestamp': ['2024-12-01 16:32:41.483954', '2024-12-01 16:32:56.102877', '2024-12-01 16:32:58.076778', '2024-12-01 16:32:59.911264', '2024-12-01 16:33:01.002760', 
                  '2024-12-01 16:33:01.898516', '2024-12-01 16:33:02.742819', '2024-12-01 16:33:03.594285', '2024-12-01 16:33:04.426203', '2024-12-01 16:33:05.251616', 
                  '2024-12-01 16:33:06.185556', '2024-12-01 16:33:07.245517', '2024-12-01 16:34:21.522277', '2024-12-01 16:35:53.702247', '2024-12-01 16:37:46.000164']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Set up the seaborn style
sns.set(style="whitegrid")

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# Scatter plot 1: Total Cost vs Total Calories
sns.scatterplot(x='Total Cost', y='Total Calories', hue='source', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Total Cost vs Total Calories')

# Scatter plot 2: Total Protein vs Cost Efficiency
sns.scatterplot(x='Total Protein (g)', y='Cost Efficiency (%)', hue='source', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Total Protein vs Cost Efficiency')

# Scatter plot 3: Nutritional Balance vs Recipe Variety
sns.scatterplot(x='Nutritional Balance (%)', y='Recipe Variety (%)', hue='source', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Nutritional Balance vs Recipe Variety')

# Scatter plot 4: Total Cost vs Recipe Variety
sns.scatterplot(x='Total Cost', y='Recipe Variety (%)', hue='source', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Total Cost vs Recipe Variety')

# Scatter plot 5: Total Protein vs Nutritional Balance
sns.scatterplot(x='Total Protein (g)', y='Nutritional Balance (%)', hue='source', data=df, ax=axes[2, 0])
axes[2, 0].set_title('Total Protein vs Nutritional Balance')

# Adjust layout to make room for titles
plt.tight_layout()

# Show the plot
plt.show()

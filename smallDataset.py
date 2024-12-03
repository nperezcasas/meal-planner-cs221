import pandas as pd

# Load the full dataset
full_recipes_df = pd.read_csv('files/recipes_data.csv')

# Randomly sample 1000 recipes
small_recipes_df = full_recipes_df.sample(n=100, random_state=42)  # Set a random seed for reproducibility

# Save the smaller dataset
#small_recipes_df.to_csv('small_recipes_dataset.csv', index=False)

small_recipes_df.to_csv('files/100_recipes_dataset.csv', index=False)
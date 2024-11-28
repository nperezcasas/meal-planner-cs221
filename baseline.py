import random
import pandas as pd
import numpy as np

# Load the cleaned dataset
recipes_df = pd.read_csv('100_combined_recipe_data.csv')

# Normalize the nutritional values per dollar
recipes_df['protein_per_dollar'] = recipes_df['Total_Protein'] / recipes_df['Estimated_Cost']
recipes_df['calories_per_dollar'] = recipes_df['Total_Calories'] / recipes_df['Estimated_Cost']

# Function to calculate health score
def calculate_health_score(row):
    # Protein is positive, excess fats/carbs are negative
    protein_ratio = row['Total_Protein'] / row['Total_Calories'] if row['Total_Calories'] > 0 else 0
    fat_ratio = row['Total_Fat'] / row['Total_Calories'] if row['Total_Calories'] > 0 else 0
    carb_ratio = row['Total_Carbs'] / row['Total_Calories'] if row['Total_Calories'] > 0 else 0
    
    # Ideal macronutrient ratios (approximately 30% protein, 30% fat, 40% carbs)
    health_score = (
        protein_ratio * 2.0 -  # Favor protein
        abs(fat_ratio - 0.3) -  # Penalize deviation from ideal fat ratio
        abs(carb_ratio - 0.4)   # Penalize deviation from ideal carb ratio
    )
    return health_score

# Weights for scoring
w1 = 0.7  # Weight for health score
w2 = 0.3  # Weight for cost

# Calculate scores
recipes_df['health_score'] = recipes_df.apply(calculate_health_score, axis=1)
recipes_df['combined_score'] = (w1 * recipes_df['health_score'] - 
                              w2 * (recipes_df['Estimated_Cost'] / recipes_df['Total_Calories']))

# Random baseline function
def random_baseline(recipes_df, budget, num_recipes):
    selected_recipes = []
    total_cost = 0
    
    # Keep randomly selecting recipes until budget or number limit is met
    while len(selected_recipes) < num_recipes and total_cost <= budget:
        recipe = recipes_df.sample().iloc[0]
        if recipe['Name'] not in [r['Name'] for r in selected_recipes]:
            if total_cost + recipe['Estimated_Cost'] <= budget:
                selected_recipes.append(recipe)
                total_cost += recipe['Estimated_Cost']
    
    return selected_recipes, total_cost

# Example usage
budget = 50
num_recipes = 5
selected_recipes, total_cost = random_baseline(recipes_df, budget, num_recipes)

# Create dataframe of selected recipes
selected_recipes_df = pd.DataFrame(selected_recipes)

# Display results
print("\nSelected Recipes:")
print(selected_recipes_df[['Name', 'Total_Protein', 'Total_Calories', 'Estimated_Cost', 'combined_score']])
print(f"\nTotal Cost: ${total_cost:.2f}")
print(f"Total Calories: {selected_recipes_df['Total_Calories'].sum():.0f}")
print(f"Total Protein: {selected_recipes_df['Total_Protein'].sum():.1f}g")

# Save the selected recipes to a CSV file
selected_recipes_df.to_csv('baseline_selected_recipes.csv', index=False)
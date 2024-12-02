import random
import pandas as pd
import numpy as np

# Load the cleaned dataset
recipes_df = pd.read_csv('combined_recipe_data.csv')

# Normalize the nutritional values per dollar
recipes_df['protein_per_dollar'] = recipes_df['ProteinContent'] / recipes_df['Estimated_Cost']
recipes_df['calories_per_dollar'] = recipes_df['Calories'] / recipes_df['Estimated_Cost']

# Function to calculate health score
def calculate_health_score(row):
    # Protein is positive, excess fats/carbs are negative
    protein_ratio = row['ProteinContent'] / row['Calories'] if row['Calories'] > 0 else 0
    fat_ratio = row['FatContent'] / row['Calories'] if row['Calories'] > 0 else 0
    carb_ratio = row['CarbohydrateContent'] / row['Calories'] if row['Calories'] > 0 else 0
    
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
                              w2 * (recipes_df['Estimated_Cost'] / recipes_df['Calories']))

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

def display_recipe(recipe, servings):
    """Display recipe information with adjusted servings"""
    print(f"\n{recipe['Name']}")
    print(f"Calories: {recipe['Calories'] * servings:.0f} total ({recipe['Calories']:.0f} per serving)")
    print(f"Cost: ${recipe['Estimated_Cost'] * servings:.2f} total (${recipe['Estimated_Cost']:.2f} per serving)")
    print(f"Rating: {recipe['AggregatedRating']:.1f}")

def calculate_cost_efficiency(total_cost, user_budget):
    """
    Calculate the cost efficiency of the meal plan.
    
    Parameters:
    - total_cost (float): The total cost of the meal plan.
    - user_budget (float): The user's budget.
    
    Returns:
    - float: The cost efficiency as a percentage.
    """
    return (total_cost / user_budget) * 100

def calculate_nutritional_balance(actual_intake, recommended_intake):
    """
    Calculate the nutritional balance for a nutrient.
    
    Parameters:
    - actual_intake (float): The actual intake of the nutrient.
    - recommended_intake (float): The recommended intake of the nutrient.
    
    Returns:
    - float: The nutritional balance as a percentage.
    """
    return (actual_intake / recommended_intake) * 100

def calculate_overall_nutritional_balance(nutrient_intakes, recommended_intakes):
    """
    Calculate the overall nutritional balance for all nutrients.
    
    Parameters:
    - nutrient_intakes (dict): A dictionary of actual intakes for each nutrient.
    - recommended_intakes (dict): A dictionary of recommended intakes for each nutrient.
    
    Returns:
    - float: The overall nutritional balance as a percentage.
    """
    scores = []
    for nutrient, actual_intake in nutrient_intakes.items():
        recommended_intake = recommended_intakes.get(nutrient, 1)  # Avoid division by zero
        scores.append(calculate_nutritional_balance(actual_intake, recommended_intake))
    return sum(scores) / len(scores)

def calculate_recipe_variety(unique_ingredients, total_ingredients):
    """
    Calculate the recipe variety of the meal plan.
    
    Parameters:
    - unique_ingredients (int): The number of unique ingredients in the meal plan.
    - total_ingredients (int): The total number of ingredients in the meal plan.
    
    Returns:
    - float: The recipe variety as a percentage.
    """
    return (unique_ingredients / total_ingredients) * 100

# Example usage
budget = 50
num_recipes = 5
selected_recipes, total_cost = random_baseline(recipes_df, budget, num_recipes)

# Create dataframe of selected recipes
selected_recipes_df = pd.DataFrame(selected_recipes)

# Display results
print("\nSelected Recipes:")
for _, recipe in selected_recipes_df.iterrows():
    display_recipe(recipe, 1)

print(f"\nTotal Cost: ${total_cost:.2f}")
print(f"Total Calories: {selected_recipes_df['Calories'].sum():.0f}")
print(f"Total Protein: {selected_recipes_df['ProteinContent'].sum():.1f}g")

# Calculate evaluation metrics
cost_efficiency = calculate_cost_efficiency(total_cost, budget)
nutrient_intakes = {
    'protein': selected_recipes_df['ProteinContent'].sum(),
    'carbs': selected_recipes_df['CarbohydrateContent'].sum(),
    'fat': selected_recipes_df['FatContent'].sum()
}
recommended_intakes = {
    'protein': 50.0 * num_recipes,  # Example: 50g protein per day per recipe
    'carbs': 300.0 * num_recipes,   # Example: 300g carbs per day per recipe
    'fat': 70.0 * num_recipes       # Example: 70g fat per day per recipe
}
nutritional_balance = calculate_overall_nutritional_balance(nutrient_intakes, recommended_intakes)
unique_ingredients = len(set(ingredient for ingredients in selected_recipes_df['Ingredients'] for ingredient in eval(ingredients)))
total_ingredients = sum(len(eval(ingredients)) for ingredients in selected_recipes_df['Ingredients'])
recipe_variety = calculate_recipe_variety(unique_ingredients, total_ingredients)

print(f"\nCost Efficiency: {cost_efficiency:.2f}%")
print(f"Nutritional Balance: {nutritional_balance:.2f}%")
print(f"Recipe Variety: {recipe_variety:.2f}%")

# Function to save results to an evaluation file
def save_results_to_eval_file(results, source):
    results['source'] = source
    results['timestamp'] = pd.to_datetime('now')  # Add timestamp for tracking
    try:
        eval_results = pd.read_csv('eval_results.csv')
        eval_results = pd.concat([eval_results, results], ignore_index=True)
    except FileNotFoundError:
        eval_results = results
    eval_results.to_csv('eval_results.csv', index=False)

# Save baseline results to the evaluation file
baseline_results = pd.DataFrame({
    'Total Cost': [total_cost],
    'Total Calories': [selected_recipes_df['Calories'].sum()],
    'Total Protein (g)': [selected_recipes_df['ProteinContent'].sum()],
    'Cost Efficiency (%)': [cost_efficiency],
    'Nutritional Balance (%)': [nutritional_balance],
    'Recipe Variety (%)': [recipe_variety],
})
save_results_to_eval_file(baseline_results, source='baseline')

# Save the selected recipes to a CSV file
selected_recipes_df.to_csv('baseline_selected_recipes.csv', index=False)
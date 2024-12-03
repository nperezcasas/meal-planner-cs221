import pandas as pd
import ast
import re

# Load all datasets
nutrition_df = pd.read_csv('files/100_ingredient_nutrition.csv', index_col=0)
costs_df = pd.read_csv('files/100_ingredient_costs.csv')
recipes_df = pd.read_csv('files/100_recipes_dataset.csv')

def clean_ingredient_name(name):
    # Remove quantities and units
    name = re.sub(r'^[\d\s/]+', '', name)  # Remove leading numbers
    name = re.sub(r'\(.*?\)', '', name)    # Remove parentheses content
    name = name.split(',')[0]              # Remove anything after comma
    return name.strip().lower()

def get_nutrition_for_ingredient(ingredient_name, nutrition_df):
    # Try exact match first
    if ingredient_name in nutrition_df.index:
        return nutrition_df.loc[ingredient_name]
    
    # Try partial match
    for idx in nutrition_df.index:
        if idx in ingredient_name or ingredient_name in idx:
            return nutrition_df.loc[idx]
    
    return pd.Series({'Protein': 0, 'Fat': 0, 'Carbohydrates': 0, 'Calories': 0})

def get_cost_for_ingredient(ingredient_name, costs_df):
    # Try exact match first
    match = costs_df[costs_df['Name'].str.lower() == ingredient_name]
    if not match.empty:
        return match['Avg Cost'].iloc[0]
    
    # Try partial match
    for _, row in costs_df.iterrows():
        if row['Name'].lower() in ingredient_name or ingredient_name in row['Name'].lower():
            return row['Avg Cost']
    
    return 0

def process_recipe(row):
    try:
        # Parse ingredients list from string representation
        ingredients = ast.literal_eval(row['ingredients'])
        
        # Initialize totals
        total_protein = 0
        total_fat = 0
        total_carbs = 0
        total_calories = 0
        total_cost = 0
        
        # Process each ingredient
        for ingredient in ingredients:
            clean_name = clean_ingredient_name(ingredient)
            
            # Get nutrition
            nutrition = get_nutrition_for_ingredient(clean_name, nutrition_df)
            total_protein += nutrition['Protein']
            total_fat += nutrition['Fat']
            total_carbs += nutrition['Carbohydrates']
            total_calories += nutrition['Calories']
            
            # Get cost
            cost = get_cost_for_ingredient(clean_name, costs_df)
            total_cost += cost
        
        return pd.Series({
            'Name': row['title'],
            'Ingredients': ingredients,
            'Total_Protein': round(total_protein, 2),
            'Total_Fat': round(total_fat, 2),
            'Total_Carbs': round(total_carbs, 2),
            'Total_Calories': round(total_calories, 2),
            'Estimated_Cost': round(total_cost, 2)
        })
    except:
        return pd.Series({
            'Name': row['title'],
            'Ingredients': [],
            'Total_Protein': 0,
            'Total_Fat': 0,
            'Total_Carbs': 0,
            'Total_Calories': 0,
            'Estimated_Cost': 0
        })

# Process all recipes
result_df = recipes_df.apply(process_recipe, axis=1)

# Save to CSV
result_df.to_csv('files/100_combined_recipe_data.csv', index=False)

# Display first few results
print(result_df.head())
import pandas as pd
import re
from datasets import load_dataset

# Load ingredient prices
prices_df = pd.read_csv('ingredients_with_price.csv', delimiter=r'\s{2,}', engine='python')

# Clean up price column - extract numeric value and convert to float
prices_df['Price'] = prices_df['Cost per Unit ($)'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
prices_df['Ingredient'] = prices_df['Ingredient'].str.strip().str.lower()

# Load recipe dataset from Hugging Face
print("Loading recipe dataset...")
ds = load_dataset("AkashPS11/recipes_data_food.com")
recipes_df = ds['train'].to_pandas()

def clean_ingredient_name(name):
    # Remove quantities and units
    name = re.sub(r'^[\d\s/]+', '', name)  # Remove leading numbers
    name = re.sub(r'\(.*?\)', '', name)    # Remove parentheses content
    name = name.split(',')[0]              # Remove anything after comma
    return name.strip().lower()

def get_cost_for_ingredient(ingredient_name, prices_df):
    # Try exact match first
    match = prices_df[prices_df['Ingredient'] == ingredient_name]
    if not match.empty:
        return match['Price'].iloc[0]
    
    # Try partial match
    for _, row in prices_df.iterrows():
        if row['Ingredient'] in ingredient_name or ingredient_name in row['Ingredient']:
            return row['Price']
    
    return None

def process_recipe(row):
    try:
        # Parse ingredients list from string representation
        ingredients_str = row['RecipeIngredientParts']
        if pd.isna(ingredients_str):
            return None
            
        # Remove c() wrapper and split by commas, handling the R-style format
        ingredients = [
            ing.strip(' "\'') 
            for ing in ingredients_str.strip('c()').split(',')
        ]
        
        # Initialize tracking variables
        total_cost = 0
        missing_ingredients = False
        found_ingredients = []
        missing_ingredients_list = []
        
        # Process each ingredient
        for ingredient in ingredients:
            clean_name = clean_ingredient_name(ingredient)
            cost = get_cost_for_ingredient(clean_name, prices_df)
            
            if cost is None:
                missing_ingredients = True
                missing_ingredients_list.append(clean_name)
            else:
                total_cost += cost
                found_ingredients.append(clean_name)
        
        if missing_ingredients:
            # Optionally print missing ingredients for debugging
            # print(f"Recipe {row['Name']} missing ingredients: {missing_ingredients_list}")
            return None
            
        return pd.Series({
            'RecipeId': row['RecipeId'],
            'Name': row['Name'],
            'Ingredients': ingredients,
            'Found_Ingredients': found_ingredients,
            'Calories': row['Calories'],
            'ProteinContent': row['ProteinContent'],
            'FatContent': row['FatContent'],
            'CarbohydrateContent': row['CarbohydrateContent'],
            'Estimated_Cost': round(total_cost, 2),
            'ReviewCount': row['ReviewCount'],
            'AggregatedRating': row['AggregatedRating']
        })
        
    except Exception as e:
        print(f"Error processing recipe {row['Name']}: {str(e)}")
        return None

# Process all recipes
print("Processing recipes...")
result_df = recipes_df.apply(process_recipe, axis=1)
result_df = result_df.dropna()  # Remove recipes where processing failed

# Save to CSV
print(f"Saving {len(result_df)} valid recipes...")
result_df.to_csv('combined_recipe_data.csv', index=False)

# Display some statistics
print(f"\nTotal recipes processed: {len(recipes_df)}")
print(f"Recipes with complete ingredient prices: {len(result_df)}")
print(f"Recipes dropped: {len(recipes_df) - len(result_df)}")
print("\nSample of results:")
print(result_df[['Name', 'Estimated_Cost', 'AggregatedRating']].head())
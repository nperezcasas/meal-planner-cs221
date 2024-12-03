import pandas as pd
from collections import Counter
from datasets import load_dataset

ds = load_dataset("AkashPS11/recipes_data_food.com")
recipes_raw = ds['train'].to_pandas()

# Filter out recipes with "Dessert" in their keywords
filtered_recipes = recipes_raw[~recipes_raw['Keywords'].str.contains('Dessert', na=False)]

# Extract all ingredients from RecipeIngredientParts
all_ingredients = []
for ingredients_str in filtered_recipes['RecipeIngredientParts']:
    if ingredients_str is not None:
        # Remove the c() wrapper and split by commas
        # Strip quotes and whitespace from each ingredient
        ingredients_list = [
            ing.strip(' "\'') 
            for ing in ingredients_str.strip('c()').split(',')
        ]
        all_ingredients.extend(ingredients_list)

# Count ingredient frequencies
ingredient_counts = Counter(all_ingredients)

# Get top 5000 most common ingredients
top_5000_ingredients = dict(ingredient_counts.most_common(5000))

# Create a DataFrame with the ingredients and their counts
ingredients_df = pd.DataFrame({
    'Ingredient': list(top_5000_ingredients.keys()),
    'Count': list(top_5000_ingredients.values())
})

# Save to CSV
ingredients_df.to_csv('files/top_5000_ingredients.csv', index=False)

# Print some statistics
print(f"Total unique ingredients: {len(ingredient_counts)}")
print("\nTop 5000 most common ingredients:")
for ing in list(top_5000_ingredients.items()):
    print(f"{ing}")
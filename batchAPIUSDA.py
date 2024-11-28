import pandas as pd
import aiohttp
import asyncio
import json
from tqdm import tqdm

API_KEY = 'twqFFbyj6WE3I3LjKuZmGbkycrO7bhZdpG8ApQAs'
nutrition_data = {}

# Function to fetch nutrition data from USDA API with retries and improved error handling
async def fetch_nutrition(session, ingredient, retries=3):
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={ingredient}&api_key={API_KEY}"
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return ingredient, data
                else:
                    print(f"Error {response.status} for {ingredient}. Retrying...")
                    await asyncio.sleep(1)  # Backoff before retrying
        except aiohttp.ClientError as e:
            print(f"Request failed for {ingredient}: {str(e)}. Retrying...")
            await asyncio.sleep(1)
    return ingredient, None

# Helper function to extract nutrition information from the API response
def extract_nutrition_info(first_food):
    nutrients = first_food.get('foodNutrients', [])
    nutrition_info = {}
    
    # Mapping USDA nutrient IDs to desired fields
    nutrient_map = {
        1003: 'Protein',        # Protein
        1004: 'Fat',            # Total fat
        1005: 'Carbohydrates',  # Carbohydrates
        1008: 'Calories',       # Energy (kcal)
    }
    
    for nutrient in nutrients:
        nutrient_id = nutrient.get('nutrientId')
        if nutrient_id in nutrient_map:
            nutrient_name = nutrient_map[nutrient_id]
            nutrition_info[nutrient_name] = nutrient.get('value', 'N/A')
    
    return nutrition_info

# Main asynchronous function to fetch data for all ingredients
async def main(ingredients):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_nutrition(session, ingredient) for ingredient in ingredients]
        
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing ingredients"):
            ingredient, data = await future
            if data and 'foods' in data and data['foods']:
                first_food = data['foods'][0]
                # Extract relevant nutrition information and handle missing values
                nutrition_data[ingredient] = extract_nutrition_info(first_food)
            else:
                print(f"Failed to fetch data for {ingredient}")

# Read the simplified ingredients list
simplified_ingredients_df = pd.read_csv('100_simplified_ingredients_list.csv')
ingredients = simplified_ingredients_df['NER'].str.lower().str.strip().tolist()  # Preprocess ingredient names

# Run the main function
asyncio.run(main(ingredients))

# Save the final fetched data to a JSON file
with open('nutrition_data.json', 'w') as f:
    json.dump(nutrition_data, f)

print(f"\nCompleted: Processed {len(nutrition_data)} ingredients")

# Save to CSV after ensuring the data is in a proper format
nutrition_df = pd.DataFrame.from_dict(nutrition_data, orient='index')
nutrition_df.fillna(0, inplace=True)  # Replace missing values with 0
#nutrition_df.to_csv('ingredient_nutrition.csv')
nutrition_df.to_csv('100_ingredient_nutrition.csv')

import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import defaultdict
from preference_learner import PreferenceLearner, swap_recipes

def normalize_score(series):
    """Normalize a series to range [0,1]"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series.apply(lambda x: 1)
    return (series - min_val) / (max_val - min_val)

def get_user_preferences(recipes_df):
    print("\nPlease rate the importance of each factor (1-5, 5 being most important):")
    preferences = {
        'health': float(input("Health importance: ")),
        'cost': float(input("Cost importance: ")),
        'time': float(input("Time importance: ")),
        'rating': float(input("Rating importance: ")),
        'budget': float(input("\nWhat's your budget? $")),
        'calories': float(input("Desired calories per serving: ")),
        'max_time': float(input("Maximum cooking time (in minutes, enter 0 for no limit): ")),
        'recipes_needed': int(input("How many recipes do you need for the week? ")),
        'servings_needed': int(input("How many people are you cooking for? ")),
    }
    
    return preferences

def compile_shopping_list(selected_recipes):
    shopping_list = defaultdict(float)
    
    for _, recipe in selected_recipes.iterrows():
        ingredients = eval(recipe['Ingredients'])  # Convert string representation to list
        for ingredient in ingredients:
            shopping_list[ingredient.lower()] += 1
    
    return dict(shopping_list)

def parse_time(time_str):
    if pd.isna(time_str):
        return None
    
    total_minutes = 0
    if 'PT' in time_str:
        time_str = time_str.replace('PT', '')
        if 'H' in time_str:
            hours = int(time_str.split('H')[0])
            total_minutes += hours * 60
            time_str = time_str.split('H')[1]
        if 'M' in time_str:
            minutes = int(time_str.split('M')[0])
            total_minutes += minutes
    return total_minutes

def calculate_health_score(row):
    """
    Calculate health score based on standard nutritional guidelines:
    - Daily calories: 2000-2500 calories
    - Protein: 50-60g per day (0.8g per kg of body weight)
    - Carbs: 45-65% of calories
    - Fat: 20-35% of calories
    """
    try:
        calories = row['Calories'] if not pd.isna(row['Calories']) else 0
        protein = row['ProteinContent'] if not pd.isna(row['ProteinContent']) else 0
        carbs = row['CarbohydrateContent'] if not pd.isna(row['CarbohydrateContent']) else 0
        fat = row['FatContent'] if not pd.isna(row['FatContent']) else 0
        servings = row['RecipeServings'] if not pd.isna(row['RecipeServings']) else 1

        # Calculate per serving
        calories_per_serving = calories / servings
        protein_per_serving = protein / servings
        carbs_per_serving = carbs / servings
        fat_per_serving = fat / servings

        # Ideal ranges per serving (assuming 3 meals per day)
        IDEAL_CALORIES_PER_MEAL = 600  # Based on 1800 daily calories
        IDEAL_PROTEIN_PER_MEAL = 20    # Based on 60g daily protein
        IDEAL_CARBS_RATIO = 0.5        # 50% of calories from carbs
        IDEAL_FAT_RATIO = 0.3          # 30% of calories from fat

        # Calculate scores for each component
        calorie_score = 1 - min(abs(calories_per_serving - IDEAL_CALORIES_PER_MEAL) / IDEAL_CALORIES_PER_MEAL, 1)
        protein_score = 1 - min(abs(protein_per_serving - IDEAL_PROTEIN_PER_MEAL) / IDEAL_PROTEIN_PER_MEAL, 1)

        # Calculate actual macronutrient ratios
        total_calories_from_macros = (carbs_per_serving * 4) + (protein_per_serving * 4) + (fat_per_serving * 9)
        if total_calories_from_macros > 0:
            carbs_ratio = (carbs_per_serving * 4) / total_calories_from_macros
            fat_ratio = (fat_per_serving * 9) / total_calories_from_macros
            
            # Score macronutrient balance
            carbs_balance = 1 - min(abs(carbs_ratio - IDEAL_CARBS_RATIO) / IDEAL_CARBS_RATIO, 1)
            fat_balance = 1 - min(abs(fat_ratio - IDEAL_FAT_RATIO) / IDEAL_FAT_RATIO, 1)
        else:
            carbs_balance = 0
            fat_balance = 0

        # Combine scores with weights
        health_score = (
            0.3 * calorie_score +      # 30% weight for calorie appropriateness
            0.3 * protein_score +      # 30% weight for protein content
            0.2 * carbs_balance +      # 20% weight for carbs ratio
            0.2 * fat_balance         # 20% weight for fat ratio
        )

        return health_score

    except Exception as e:
        return 0

def collect_initial_feedback(recipes_df, preference_learner):
    """Collect initial feedback on some recipes to bootstrap the model"""
    print("\n=== Initial Preference Collection ===")
    print("Please rate these sample recipes to help us learn your preferences (1-5, or 0 to skip):")
    
    # Sample diverse recipes for initial feedback
    sample_recipes = recipes_df.sample(min(10, len(recipes_df)))  # Increase sample size to 10
    
    for _, recipe in sample_recipes.iterrows():
        print(f"\n{recipe['Name']}")
        print(f"Calories: {recipe['Calories_per_serving']:.0f} per serving")
        print(f"Cost: ${recipe['Cost_per_serving']:.2f} per serving")
        print(f"Time: {recipe['TotalTime_minutes']:.0f} minutes")
        
        try:
            rating = float(input("Your rating (1-5, 0 to skip): "))
            if 1 <= rating <= 5:
                preference_learner.add_feedback(recipe, liked=(rating >= 4))
        except ValueError:
            continue
    
    # Initial training
    preference_learner.train()

def select_initial_recipes(filtered_df, prefs, preference_learner):
    """Select initial recipes using both base scores and learned preferences"""
    
    # Calculate preference scores if we have training data
    if preference_learner.train():
        filtered_df['preference_score'] = filtered_df.apply(
            lambda x: preference_learner.predict_score(x), axis=1
        )
        # Combine with original scoring
        filtered_df['final_score'] = (
            filtered_df['combined_score'] * 0.3 + 
            filtered_df['preference_score'] * 0.7
        )
        filtered_df = filtered_df.sort_values('final_score', ascending=False)
    else:
        # Fall back to original scoring
        filtered_df = filtered_df.sort_values('combined_score', ascending=False)
    
    # Select recipes within budget
    selected_recipes = []
    total_cost = 0
    
    for _, recipe in filtered_df.iterrows():
        if len(selected_recipes) >= prefs['recipes_needed']:
            break
        if total_cost + recipe['Estimated_Cost'] <= prefs['budget']:
            selected_recipes.append(recipe)
            total_cost += recipe['Estimated_Cost']
    
    return pd.DataFrame(selected_recipes)

def display_recipe(recipe, servings):
    """Display recipe information with adjusted servings"""
    print(f"\n{recipe['Name']}")
    print(f"Calories: {recipe['Calories_per_serving'] * servings:.0f} total ({recipe['Calories_per_serving']:.0f} per serving)")
    print(f"Cost: ${recipe['Estimated_Cost'] * servings:.2f} total (${recipe['Estimated_Cost']:.2f} per serving)")
    print(f"Time: {recipe['TotalTime_minutes']:.0f} minutes")
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

def save_results_to_eval_file(results, source):
    results['source'] = source
    results['timestamp'] = pd.to_datetime('now')  # Add timestamp for tracking
    try:
        eval_results = pd.read_csv('files/eval_results.csv')
        eval_results = pd.concat([eval_results, results], ignore_index=True)
    except FileNotFoundError:
        eval_results = results
    eval_results.to_csv('files/eval_results.csv', index=False)

def main():
    # Load and prepare datasets
    print("Loading datasets...")
    recipes_with_costs = pd.read_csv('files/combined_recipe_data.csv')
    ds = load_dataset("AkashPS11/recipes_data_food.com")
    recipes_df = ds['train'].to_pandas()

    # Merge datasets
    recipes_df = pd.merge(recipes_df, recipes_with_costs[['RecipeId', 'Estimated_Cost', 'Ingredients']], 
                         left_on='RecipeId', right_on='RecipeId', how='inner')

    # Extract unique ingredients from the dataset
    all_ingredients = set()
    for ingredients in recipes_df['Ingredients']:
        all_ingredients.update(eval(ingredients))
    ingredient_list = sorted(all_ingredients)

    # Initialize preference learner with the ingredient list
    preference_learner = PreferenceLearner(ingredient_list)
    
    # Add per-serving calculations BEFORE collecting feedback
    print("Processing recipes...")
    recipes_df['TotalTime_minutes'] = recipes_df['TotalTime'].apply(parse_time)

    # Add these calculations before collect_initial_feedback
    recipes_df['Calories_per_serving'] = recipes_df.apply(
        lambda x: x['Calories'] / x['RecipeServings'] if x['RecipeServings'] > 0 else x['Calories'],
        axis=1
    )
    recipes_df['Protein_per_serving'] = recipes_df.apply(
        lambda x: x['ProteinContent'] / x['RecipeServings'] if x['RecipeServings'] > 0 else x['ProteinContent'],
        axis=1
    )
    recipes_df['Fat_per_serving'] = recipes_df.apply(
        lambda x: x['FatContent'] / x['RecipeServings'] if x['RecipeServings'] > 0 else x['FatContent'],
        axis=1
    )
    recipes_df['Carbs_per_serving'] = recipes_df.apply(
        lambda x: x['CarbohydrateContent'] / x['RecipeServings'] if x['RecipeServings'] > 0 else x['CarbohydrateContent'],
        axis=1
    )
    recipes_df['Cost_per_serving'] = recipes_df.apply(
        lambda x: x['Estimated_Cost'] / x['RecipeServings'] if x['RecipeServings'] > 0 else x['Estimated_Cost'],
        axis=1
    )

    # Filter out recipes with unrealistic values
    recipes_df = recipes_df[
        (recipes_df['Calories_per_serving'].between(100, 1200)) &  # Reasonable calories per serving
        (recipes_df['Protein_per_serving'].between(0, 100)) &      # Reasonable protein per serving
        (recipes_df['Fat_per_serving'].between(0, 100)) &          # Reasonable fat per serving
        (recipes_df['Carbs_per_serving'].between(0, 200))          # Reasonable carbs per serving
    ]

    # Calculate health score (same as before)
    recipes_df['health_score'] = recipes_df.apply(calculate_health_score, axis=1)
    recipes_df['health_score_norm'] = normalize_score(recipes_df['health_score'])

    # Add cost per serving calculation before normalizing scores
    recipes_df['Cost_per_serving'] = recipes_df.apply(
        lambda x: x['Estimated_Cost'] / x['RecipeServings'] if x['RecipeServings'] > 0 else x['Estimated_Cost'],
        axis=1
    )

    # Calculate and normalize scores
    recipes_df['cost_score_norm'] = 1 - normalize_score(recipes_df['Cost_per_serving'])
    recipes_df['time_score_norm'] = 1 - normalize_score(recipes_df['TotalTime_minutes'].fillna(recipes_df['TotalTime_minutes'].mean()))
    recipes_df['rating_score_norm'] = normalize_score(recipes_df['AggregatedRating'])

    prefs = {
        'health': 0.25,  # Example weight for health score
        'cost': 0.25,    # Example weight for cost
        'time': 0.25,    # Example weight for preparation time
        'rating': 0.25   # Example weight for rating
    }

    # Calculate combined score
    recipes_df['combined_score'] = (
        prefs['health'] * recipes_df['health_score_norm'] +
        prefs['cost'] * recipes_df['cost_score_norm'] +
        prefs['time'] * recipes_df['time_score_norm'] +
        prefs['rating'] * recipes_df['rating_score_norm']
    ) / (prefs['health'] + prefs['cost'] + prefs['time'] + prefs['rating'])

    # Collect initial feedback to train the model
    collect_initial_feedback(recipes_df, preference_learner)

    # Get user preferences
    prefs = get_user_preferences(recipes_df)

    # Filter recipes
    filtered_df = recipes_df.copy()
    if prefs['max_time'] > 0:
        filtered_df = filtered_df[
            (filtered_df['TotalTime_minutes'].isna()) | 
            (filtered_df['TotalTime_minutes'] <= prefs['max_time'])
        ]

    # Select initial recipes using learned preferences
    selected_recipes_df = select_initial_recipes(filtered_df, prefs, preference_learner)

    # Display initial selection
    print("\n=== Selected Recipes ===")
    for _, recipe in selected_recipes_df.iterrows():
        display_recipe(recipe, prefs['servings_needed'])
        # Add initial selections as positive examples
        preference_learner.add_feedback(recipe, liked=True)
    
    # Retrain model with initial selections
    preference_learner.train()

    # Allow recipe swapping with preference learning
    selected_recipes_df = swap_recipes(selected_recipes_df, filtered_df, prefs, preference_learner)

    # Generate final shopping list
    shopping_list = compile_shopping_list(selected_recipes_df)

    # Display final summary
    print("\n=== Final Meal Plan ===")
    total_cost = 0
    total_rating = 0
    total_time = 0
    total_calories = 0

    for _, recipe in selected_recipes_df.iterrows():
        display_recipe(recipe, prefs['servings_needed'])
        total_cost += recipe['Estimated_Cost'] * max(1, prefs['servings_needed'] / recipe['RecipeServings'])
        total_rating += recipe['AggregatedRating']
        total_time += recipe['TotalTime_minutes'] if not pd.isna(recipe['TotalTime_minutes']) else 0
        total_calories += recipe['Calories_per_serving']

    print(f"\nTotal Cost (adjusted for {prefs['servings_needed']} people): ${total_cost:.2f}")
    print(f"Average Rating: {total_rating/len(selected_recipes_df):.1f}")
    print(f"Average Time: {total_time/len(selected_recipes_df):.0f} minutes")
    print(f"Average Calories per Serving: {total_calories/len(selected_recipes_df):.0f}")
    print("\n=== Shopping List ===")
    for item, count in shopping_list.items():
        print(f"- {item}: {count:.0f}")

    # Save weekly meal plan
    selected_recipes_df.to_csv('files/weekly_meal_plan.csv', index=False)

    # Save shopping list to a file
    with open('files/shopping_list.txt', 'w') as f:
        for ingredient, quantity in shopping_list.items():
            f.write(f"{ingredient}: {quantity}\n")

    # Calculate evaluation metrics
    cost_efficiency = calculate_cost_efficiency(total_cost, prefs['budget'])
    nutrient_intakes = {
        'protein': total_calories * 0.15 / 4,  # Example calculation for protein intake
        'carbs': total_calories * 0.55 / 4,    # Example calculation for carbs intake
        'fat': total_calories * 0.30 / 9       # Example calculation for fat intake
    }
    recommended_intakes = {
        'protein': 50.0,
        'carbs': 300.0,
        'fat': 70.0
    }
    nutritional_balance = calculate_overall_nutritional_balance(nutrient_intakes, recommended_intakes)
    recipe_variety = calculate_recipe_variety(len(shopping_list), sum(shopping_list.values()))

    print(f"\nCost Efficiency: {cost_efficiency:.2f}%")
    print(f"Nutritional Balance: {nutritional_balance:.2f}%")
    print(f"Recipe Variety: {recipe_variety:.2f}%")

    # Save baseline results to the evaluation file
    baseline_results = pd.DataFrame({
        'Total Cost': [total_cost],
        'Total Calories': [selected_recipes_df['Calories'].sum()],
        'Total Protein (g)': [selected_recipes_df['ProteinContent'].sum()],
        'Cost Efficiency (%)': [cost_efficiency],
        'Nutritional Balance (%)': [nutritional_balance],
        'Recipe Variety (%)': [recipe_variety],
    })
    save_results_to_eval_file(baseline_results, source='meal_planner')

if __name__ == "__main__":
    main()
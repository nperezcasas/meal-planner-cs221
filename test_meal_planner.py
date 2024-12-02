import pytest
from unittest.mock import patch
import pandas as pd

# Assuming you have these functions in the meal_planner module
from meal_planner import get_user_preferences, main as meal_planner

# Sample test data for recipes_df
test_data = """Name,Total_Protein,Total_Calories,Total_Fat,Total_Carbs,Estimated_Cost,Ingredients,TotalTime,RecipeServings,AggregatedRating
Recipe1,10,200,5,30,3,"['ingredient1', 'ingredient2']", "PT30M", 4, 4.5
Recipe2,15,300,10,40,4,"['ingredient3', 'ingredient4']", "PT45M", 4, 4.0
Recipe3,5,100,2,20,2,"['ingredient5', 'ingredient6']", "PT20M", 4, 3.5
Recipe4,20,500,15,60,6,"['ingredient7', 'ingredient8']", "PT60M", 4, 5.0
"""

@pytest.fixture
def recipes_df():
    # Create a DataFrame from the sample test data
    from io import StringIO
    return pd.read_csv(StringIO(test_data))

# Function to test `get_user_preferences` with mocked input
def test_get_user_preferences(recipes_df):
    test_cases = [
        {
            'inputs': ['5', '4', '3', '2', '20', '500', '60', '7', '4'],
            'expected': {
                'health': 5.0,
                'cost': 4.0,
                'time': 3.0,
                'rating': 2.0,
                'budget': 20.0,
                'calories': 500.0,
                'max_time': 60.0,
                'recipes_needed': 7,
                'servings_needed': 4
            }
        },
        {
            'inputs': ['3', '5', '2', '4', '30', '600', '30', '3', '6'],
            'expected': {
                'health': 3.0,
                'cost': 5.0,
                'time': 2.0,
                'rating': 4.0,
                'budget': 30.0,
                'calories': 600.0,
                'max_time': 30.0,
                'recipes_needed': 3,
                'servings_needed': 6
            }
        }
    ]

    for case in test_cases:
        with patch('builtins.input', side_effect=case['inputs']):
            preferences = get_user_preferences(recipes_df)
            for key in preferences:
                assert preferences[key] == case['expected'][key], f"Expected {key} to be {case['expected'][key]}, but got {preferences[key]}"

# Function to test `meal_planner` with different user preferences
def test_meal_planner_with_preferences(recipes_df):
    test_cases = [
        {
            'preferences': {
                'health': 5.0,
                'cost': 4.0,
                'time': 3.0,
                'rating': 2.0,
                'budget': 20.0,
                'calories': 500.0,
                'max_time': 60.0,
                'recipes_needed': 2,
                'servings_needed': 4
            },
            'expected_num_recipes': 2  # Based on your planner logic, adjust accordingly
        },
        {
            'preferences': {
                'health': 3.0,
                'cost': 5.0,
                'time': 2.0,
                'rating': 4.0,
                'budget': 30.0,
                'calories': 600.0,
                'max_time': 30.0,
                'recipes_needed': 3,
                'servings_needed': 6
            },
            'expected_num_recipes': 3  # Based on your planner logic, adjust accordingly
        }
    ]

    for case in test_cases:
        with patch('meal_planner.get_user_preferences', return_value=case['preferences']):
            selected_recipes_df, total_cost = meal_planner(recipes_df, case['preferences'])
            
            # Test that the number of selected recipes is as expected
            assert len(selected_recipes_df) == case['expected_num_recipes'], \
                f"Expected {case['expected_num_recipes']} recipes, but got {len(selected_recipes_df)}"
            
            # Additional assertions can be added to check for cost, calories, or other factors based on preferences

if __name__ == "__main__":
    pytest.main()
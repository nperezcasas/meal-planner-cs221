# Meal Planner Project

This project is a meal planner that combines recipe data with ingredient costs to estimate the cost of recipes. The project uses data from Hugging Face and ingredient costs generated with ChatGPT.

## Getting Started

### Prerequisites

- Miniconda (Python 3.8 or higher)

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/nperezcasas/meal-planner-cs221.git
    cd meal-planner-cs221
    ```

2. **Set up a Conda environment**:
    ```sh
    conda create -n meal-planner python=3.8
    conda activate meal-planner
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset
The dataset used in this project is from Hugging Face. You can download it using the `datasets` library.

## Instructions

1. **Generate the Top 5000 Ingredients**:
    - Run `ingredients.py` to generate `top_5000_ingredients.csv`:
      ```sh
      python ingredients.py
      ```

2. **Generate Ingredient Costs**:
    - Use ChatGPT or another method to generate the ingredient costs and save them in `ingredients_with_price.csv`.

3. **Combine Recipe Data**:
    - Run `combineRecipeData.py` to combine the recipe data with ingredient costs:
      ```sh
      python combineRecipeData.py
      ```

4. **Generate Weekly Meal Plan**:
    - Run `meal_planner.py` to generate the meal plan using `preference_learner` to learn user preferences and recommend recipes:
      ```sh
      python meal_planner.py
      ```

## Files
- `ingredients.py`: Script to generate the top 5000 ingredients.
- `combineRecipeData.py`: Script to combine recipe data with ingredient costs.
- `meal_planner.py`: Script to generate the meal plan using user preferences.
- `ingredients_with_price.csv`: Ingredient costs generated with ChatGPT.

## License
This project is licensed under the MIT License.

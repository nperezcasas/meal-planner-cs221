import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def create_recipe_features(recipe, ingredient_list):
    """Extract relevant features from a recipe"""
    ingredient_vector = np.array([1 if ingredient in recipe['Ingredients'] else 0 for ingredient in ingredient_list])
    return np.concatenate((
        ingredient_vector,
        np.array([
            recipe['health_score'],
            recipe['TotalTime_minutes'] if not pd.isna(recipe['TotalTime_minutes']) else 0
        ])
    ))

class PreferenceLearner:
    def __init__(self, ingredient_list):
        self.scaler = StandardScaler()
        self.positive_examples = []
        self.negative_examples = []
        self.model = NearestNeighbors(algorithm='ball_tree')
        self.min_examples = 3  # Minimum examples needed for training
        self.ingredient_list = ingredient_list
        
    def add_feedback(self, recipe, liked=True):
        """Add a recipe to the learning set"""
        features = create_recipe_features(recipe, self.ingredient_list)
        if liked:
            self.positive_examples.append(features)
        else:
            self.negative_examples.append(features)
        
        # Automatically train if we have enough examples
        if len(self.positive_examples) >= self.min_examples:
            self.train()
    
    def train(self):
        """Train the model based on collected feedback"""
        if len(self.positive_examples) < self.min_examples:
            return False
            
        # Combine positive examples
        X = np.array(self.positive_examples)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Adjust n_neighbors based on the number of positive examples
        n_neighbors = min(5, len(self.positive_examples))
        self.model.set_params(n_neighbors=n_neighbors)
        
        # Train nearest neighbors model
        self.model.fit(X_scaled)
        return True
    
    def predict_score(self, recipe):
        """Predict how well a recipe matches learned preferences"""
        if len(self.positive_examples) < self.min_examples:
            return 0.5  # Default score when insufficient training data
            
        features = create_recipe_features(recipe, self.ingredient_list)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get distance to nearest positive examples
        distances, _ = self.model.kneighbors(features_scaled)
        
        # Convert distance to similarity score (0 to 1)
        similarity = np.exp(-distances.mean())
        
        # Penalize if similar to negative examples
        if self.negative_examples:
            neg_features = np.array(self.negative_examples)
            neg_features_scaled = self.scaler.transform(neg_features)
            neg_distances = np.linalg.norm(features_scaled - neg_features_scaled, axis=1)
            neg_similarity = np.exp(-neg_distances.mean())
            similarity *= (1 - 0.3 * neg_similarity)  # Lighter penalty
            
        return similarity

def swap_recipes(current_recipes, all_recipes, preferences, learner):
    """
    Allow user to swap recipes in the meal plan with preference learning.
    """
    while True:
        response = input("\nWould you like to swap any recipes? (y/n): ").lower()
        if response != 'y':
            # Add all kept recipes as positive examples
            for _, recipe in current_recipes.iterrows():
                learner.add_feedback(recipe, liked=True)
            return current_recipes
            
        # Display current recipes with indices
        print("\nCurrent recipes:")
        for i, (_, recipe) in enumerate(current_recipes.iterrows()):
            print(f"{i+1}. {recipe['Name']} (Rating: {recipe['AggregatedRating']:.1f})")
            
        try:
            idx = int(input("\nEnter the number of the recipe you want to replace (0 to cancel): ")) - 1
            if idx < 0:
                return current_recipes
            if idx >= len(current_recipes):
                print("Invalid recipe number")
                continue
            
            # Add replaced recipe as negative example
            replaced_recipe = current_recipes.iloc[idx]
            learner.add_feedback(replaced_recipe, liked=False)
                
            # Show alternative recipes, ranked by learned preferences
            print("\nTop 5 alternative recipes:")
            alternatives = all_recipes[~all_recipes['Name'].isin(current_recipes['Name'])]
            
            # Train model with current feedback
            if learner.train():
                # Score alternatives using learned preferences
                alternatives['preference_score'] = alternatives.apply(
                    lambda x: learner.predict_score(x), axis=1
                )
                # Combine with other scoring factors
                alternatives['final_score'] = (
                    alternatives['combined_score'] * 0.7 + 
                    alternatives['preference_score'] * 0.3
                )
                alternatives = alternatives.sort_values('final_score', ascending=False).head()
            else:
                # Fall back to original scoring if no training data
                alternatives = alternatives.sort_values('combined_score', ascending=False).head()
            
            for i, (_, recipe) in enumerate(alternatives.iterrows()):
                print(f"{i+1}. {recipe['Name']} (Rating: {recipe['AggregatedRating']:.1f})")
                
            alt_idx = int(input("\nEnter the number of the recipe you want to use (0 to cancel): ")) - 1
            if alt_idx < 0:
                continue
            if alt_idx >= len(alternatives):
                print("Invalid recipe number")
                continue
                
            # Add selected alternative as positive example
            selected_recipe = alternatives.iloc[alt_idx]
            learner.add_feedback(selected_recipe, liked=True)
            
            # Replace the recipe
            current_recipes.iloc[idx] = selected_recipe
            print("\nRecipe swapped successfully!")
            
        except ValueError:
            print("Please enter a valid number")
            continue
            
    return current_recipes
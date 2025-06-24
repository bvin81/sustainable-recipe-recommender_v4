#!/usr/bin/env python3
"""
Enhanced Content-Based Filtering módú a fenntartható recept ajánlórendszerhez
Kompatibilis a meglévő Flask alkalmazással és adatbázis struktúrával
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import re
import json
from pathlib import Path

# Conditional imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from scipy.spatial.distance import euclidean
    from scipy.stats import pearsonr
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn libraries loaded")
except ImportError as e:
    print(f"⚠️ scikit-learn not available: {e}")
    print("🔧 Using fallback implementations")
    SKLEARN_AVAILABLE = False

class FallbackTfidfVectorizer:
    """Fallback TF-IDF implementáció ha sklearn nem elérhető"""
    def __init__(self, max_features=1000, ngram_range=(1,1), stop_words=None, min_df=1, max_df=1.0):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = set(stop_words) if stop_words == 'english' else set(stop_words or [])
        self.vocabulary_ = {}
        self.idf_ = {}
    
    def fit_transform(self, documents):
        """Basic TF-IDF implementation"""
        # Simple tokenization and term counting
        all_terms = set()
        doc_term_matrix = []
        
        for doc in documents:
            terms = self._extract_terms(doc)
            doc_term_matrix.append(terms)
            all_terms.update(terms)
        
        # Create vocabulary
        self.vocabulary_ = {term: i for i, term in enumerate(sorted(all_terms)[:self.max_features])}
        
        # Calculate TF-IDF matrix (simplified)
        matrix = np.zeros((len(documents), len(self.vocabulary_)))
        for i, terms in enumerate(doc_term_matrix):
            for term in terms:
                if term in self.vocabulary_:
                    matrix[i, self.vocabulary_[term]] += 1
        
        return matrix
    
    def transform(self, documents):
        """Transform new documents"""
        matrix = np.zeros((len(documents), len(self.vocabulary_)))
        for i, doc in enumerate(documents):
            terms = self._extract_terms(doc)
            for term in terms:
                if term in self.vocabulary_:
                    matrix[i, self.vocabulary_[term]] += 1
        return matrix
    
    def _extract_terms(self, text):
        """Simple term extraction"""
        if not text:
            return []
        text = str(text).lower()
        terms = re.findall(r'\b\w+\b', text)
        return [term for term in terms if term not in self.stop_words and len(term) > 2]

def fallback_cosine_similarity(X, Y=None):
    """Fallback cosine similarity ha sklearn nem elérhető"""
    if Y is None:
        Y = X
    
    # Normalize vectors
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    
    return np.dot(X_norm, Y_norm.T)

# Use sklearn or fallback implementations
if SKLEARN_AVAILABLE:
    TfidfVectorizer = TfidfVectorizer
    cosine_similarity = cosine_similarity
    StandardScaler = StandardScaler
else:
    TfidfVectorizer = FallbackTfidfVectorizer
    cosine_similarity = fallback_cosine_similarity
    StandardScaler = lambda: None

class AdvancedFeatureExtractor:
    """Továbbfejlesztett jellemzőkinyerő osztály"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Vectorizers
        self.ingredient_vectorizer = TfidfVectorizer(
            max_features=self.config['ingredient_max_features'],
            ngram_range=self.config['ingredient_ngram_range'],
            stop_words='english',
            min_df=self.config['min_df'],
            max_df=self.config['max_df']
        )
        
        self.instruction_vectorizer = TfidfVectorizer(
            max_features=self.config['instruction_max_features'],
            ngram_range=(1, 1),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Scalers for numeric features
        if SKLEARN_AVAILABLE:
            self.numeric_scaler = StandardScaler()
        else:
            self.numeric_scaler = None
        
        self.fitted = False
    
    def _default_config(self) -> Dict[str, Any]:
        """Default konfiguráció"""
        return {
            'ingredient_max_features': 2000,
            'ingredient_ngram_range': (1, 2),
            'instruction_max_features': 1000,
            'min_df': 2,
            'max_df': 0.95,
            'numeric_features': ['HSI', 'ESI', 'PPI'],
            'weights': {
                'ingredients': 0.6,
                'instructions': 0.2,
                'numeric': 0.2
            }
        }
    
    def fit_transform(self, recipes_data: List[Dict[str, Any]]) -> np.ndarray:
        """Feature extraction és transform egyben"""
        print(f"🔧 Extracting features from {len(recipes_data)} recipes...")
        
        # Szöveg jellemzők előkészítése
        ingredients_text = []
        instructions_text = []
        numeric_features_list = []
        
        for recipe in recipes_data:
            # Ingredients processing
            ingredients = recipe.get('ingredients', '')
            if isinstance(ingredients, list):
                ingredients = ' '.join(ingredients)
            ingredients_text.append(str(ingredients))
            
            # Instructions processing (ha van)
            instructions = recipe.get('instructions', recipe.get('directions', ''))
            if isinstance(instructions, list):
                instructions = ' '.join(instructions)
            instructions_text.append(str(instructions))
            
            # Numeric features
            numeric_row = []
            for feature in self.config['numeric_features']:
                value = recipe.get(feature, 0)
                # Handle various numeric formats
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0
                numeric_row.append(float(value))
            numeric_features_list.append(numeric_row)
        
        # Feature vektorizálás
        print("📊 Vectorizing ingredients...")
        ingredient_matrix = self.ingredient_vectorizer.fit_transform(ingredients_text)
        
        print("📊 Vectorizing instructions...")
        instruction_matrix = self.instruction_vectorizer.fit_transform(instructions_text)
        
        print("📊 Scaling numeric features...")
        numeric_array = np.array(numeric_features_list)
        if self.numeric_scaler and SKLEARN_AVAILABLE:
            numeric_matrix = self.numeric_scaler.fit_transform(numeric_array)
        else:
            # Fallback normalization
            numeric_matrix = self._normalize_features(numeric_array)
        
        # Feature kombinálás
        combined_features = self._combine_features(
            ingredient_matrix, instruction_matrix, numeric_matrix
        )
        
        self.fitted = True
        print(f"✅ Feature extraction complete. Shape: {combined_features.shape}")
        return combined_features
    
    def transform(self, recipes_data: List[Dict[str, Any]]) -> np.ndarray:
        """Transform új receptekhez (már fitted vectorizer-rel)"""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Ugyanaz a feldolgozás mint fit_transform-nál
        ingredients_text = []
        instructions_text = []
        numeric_features_list = []
        
        for recipe in recipes_data:
            ingredients = recipe.get('ingredients', '')
            if isinstance(ingredients, list):
                ingredients = ' '.join(ingredients)
            ingredients_text.append(str(ingredients))
            
            instructions = recipe.get('instructions', recipe.get('directions', ''))
            if isinstance(instructions, list):
                instructions = ' '.join(instructions)
            instructions_text.append(str(instructions))
            
            numeric_row = []
            for feature in self.config['numeric_features']:
                value = recipe.get(feature, 0)
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0
                numeric_row.append(float(value))
            numeric_features_list.append(numeric_row)
        
        # Transform fitted vectorizer-ekkel
        ingredient_matrix = self.ingredient_vectorizer.transform(ingredients_text)
        instruction_matrix = self.instruction_vectorizer.transform(instructions_text)
        
        numeric_array = np.array(numeric_features_list)
        if self.numeric_scaler and SKLEARN_AVAILABLE:
            numeric_matrix = self.numeric_scaler.transform(numeric_array)
        else:
            numeric_matrix = self._normalize_features(numeric_array)
        
        return self._combine_features(ingredient_matrix, instruction_matrix, numeric_matrix)
    
    def _normalize_features(self, array: np.ndarray) -> np.ndarray:
        """Fallback normalization sklearn nélkül"""
        if array.size == 0:
            return array
        
        # Min-max normalization
        result = np.zeros_like(array, dtype=float)
        for i in range(array.shape[1]):
            col = array[:, i]
            min_val, max_val = np.min(col), np.max(col)
            if max_val - min_val > 0:
                result[:, i] = (col - min_val) / (max_val - min_val)
            else:
                result[:, i] = 0.5  # Konstans értékek esetén
        
        return result
    
    def _combine_features(self, ingredient_matrix, instruction_matrix, numeric_matrix) -> np.ndarray:
        """Feature mátrixok kombinálása súlyozott módon"""
        weights = self.config['weights']
        
        # Convert to dense arrays if sparse
        if hasattr(ingredient_matrix, 'toarray'):
            ingredient_matrix = ingredient_matrix.toarray()
        if hasattr(instruction_matrix, 'toarray'):
            instruction_matrix = instruction_matrix.toarray()
        
        # Weighted combination
        features = []
        
        # Ingredients (főbb súly)
        if ingredient_matrix.size > 0:
            features.append(ingredient_matrix * weights['ingredients'])
        
        # Instructions (kisebb súly)
        if instruction_matrix.size > 0:
            features.append(instruction_matrix * weights['instructions'])
        
        # Numeric features (közepes súly)
        if numeric_matrix.size > 0:
            features.append(numeric_matrix * weights['numeric'])
        
        # Concatenate all features
        if features:
            combined = np.hstack(features)
        else:
            # Fallback ha nincs feature
            combined = np.zeros((len(ingredient_matrix), 1))
        
        return combined

class MultiMetricSimilarity:
    """Többféle hasonlósági metrika kombinálása"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'cosine': 0.7,
            'euclidean': 0.2,
            'correlation': 0.1
        }
    
    def calculate_similarity(self, query_features: np.ndarray, recipe_features: np.ndarray) -> np.ndarray:
        """Kombinált hasonlóság számítás"""
        if query_features.size == 0 or recipe_features.size == 0:
            return np.zeros(recipe_features.shape[0])
        
        similarities = []
        
        # Cosine similarity
        if self.weights.get('cosine', 0) > 0:
            cosine_sim = cosine_similarity(query_features.reshape(1, -1), recipe_features).flatten()
            similarities.append(('cosine', cosine_sim))
        
        # Euclidean distance based similarity
        if self.weights.get('euclidean', 0) > 0:
            euclidean_sim = self._euclidean_similarity(query_features, recipe_features)
            similarities.append(('euclidean', euclidean_sim))
        
        # Correlation based similarity (ha sklearn elérhető)
        if self.weights.get('correlation', 0) > 0 and SKLEARN_AVAILABLE:
            corr_sim = self._correlation_similarity(query_features, recipe_features)
            similarities.append(('correlation', corr_sim))
        
        # Weighted combination
        if not similarities:
            return np.zeros(recipe_features.shape[0])
        
        final_similarity = np.zeros(recipe_features.shape[0])
        total_weight = 0
        
        for metric_name, sim_scores in similarities:
            weight = self.weights.get(metric_name, 0)
            final_similarity += weight * sim_scores
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            final_similarity /= total_weight
        
        return np.clip(final_similarity, 0, 1)
    
    def _euclidean_similarity(self, query: np.ndarray, recipes: np.ndarray) -> np.ndarray:
        """Euclidean distance alapú hasonlóság (0-1 skálán)"""
        distances = []
        for recipe in recipes:
            try:
                if SKLEARN_AVAILABLE:
                    dist = euclidean(query, recipe)
                else:
                    dist = np.sqrt(np.sum((query - recipe) ** 2))
                distances.append(dist)
            except:
                distances.append(1.0)  # Max distance fallback
        
        distances = np.array(distances)
        # Convert distance to similarity: similarity = 1 / (1 + distance)
        similarities = 1 / (1 + distances)
        return similarities
    
    def _correlation_similarity(self, query: np.ndarray, recipes: np.ndarray) -> np.ndarray:
        """Pearson korreláció alapú hasonlóság"""
        correlations = []
        for recipe in recipes:
            try:
                if len(set(query)) > 1 and len(set(recipe)) > 1:
                    corr, _ = pearsonr(query, recipe)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # Absolute correlation
                    else:
                        correlations.append(0.0)
                else:
                    correlations.append(0.0)
            except:
                correlations.append(0.0)
        
        return np.array(correlations)

class EnhancedContentBasedRecommender:
    """Továbbfejlesztett Content-Based ajánlórendszer"""
    
    def __init__(self, recipes_data: List[Dict[str, Any]], config: Dict[str, Any] = None):
        self.recipes_data = recipes_data
        self.config = config or self._default_config()
        
        # Initialize components
        self.feature_extractor = AdvancedFeatureExtractor(self.config.get('feature_extraction', {}))
        self.similarity_calculator = MultiMetricSimilarity(self.config.get('similarity_weights', {}))
        
        # Extract features from all recipes
        self.recipe_features = self.feature_extractor.fit_transform(recipes_data)
        
        print(f"✅ Enhanced Content-Based Recommender initialized with {len(recipes_data)} recipes")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default konfiguráció"""
        return {
            'sustainability_boost': True,
            'sustainability_weights': {
                'ESI': 0.4,  # Környezeti
                'HSI': 0.3,  # Egészségügyi
                'PPI': 0.3   # Preferencia
            },
            'diversity_factor': 0.1,  # Diverzitás növeléshez
            'min_similarity_threshold': 0.01
        }
    
    def recommend(self, 
                  user_preferences: Dict[str, Any], 
                  n_recommendations: int = 10,
                  exclude_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Fő ajánlási függvény"""
        
        print(f"🎯 Generating {n_recommendations} recommendations...")
        
        # User preferences vektorizálása
        query_features = self._vectorize_user_preferences(user_preferences)
        
        # Hasonlóság számítás
        similarities = self.similarity_calculator.calculate_similarity(
            query_features, self.recipe_features
        )
        
        # Fenntarthatósági boost alkalmazása
        if self.config['sustainability_boost']:
            sustainability_scores = self._calculate_sustainability_boost()
            final_scores = similarities * (1 + sustainability_scores)
        else:
            final_scores = similarities
        
        # Exclude certain recipes if needed
        if exclude_ids:
            exclude_indices = []
            for i, recipe in enumerate(self.recipes_data):
                recipe_id = str(recipe.get('id', recipe.get('recipeid', i)))
                if recipe_id in exclude_ids:
                    exclude_indices.append(i)
            
            for idx in exclude_indices:
                if idx < len(final_scores):
                    final_scores[idx] = 0
        
        # Top N ajánlások kiválasztása
        top_indices = np.argsort(final_scores)[::-1][:n_recommendations * 2]  # Extra candidates for diversity
        
        # Diverzitás alkalmazása
        final_indices = self._apply_diversity(top_indices, final_scores, n_recommendations)
        
        # Ajánlások formázása
        recommendations = self._format_recommendations(final_indices, final_scores, similarities)
        
        print(f"✅ Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _vectorize_user_preferences(self, preferences: Dict[str, Any]) -> np.ndarray:
        """Felhasználói preferenciák vektorizálása"""
        # Create a dummy recipe-like object from user preferences
        dummy_recipe = {
            'ingredients': preferences.get('ingredients', preferences.get('search_query', '')),
            'instructions': preferences.get('instructions', ''),
            'HSI': preferences.get('health_preference', 70),
            'ESI': preferences.get('environmental_preference', 70),
            'PPI': preferences.get('taste_preference', 70)
        }
        
        # Transform using fitted feature extractor
        query_features = self.feature_extractor.transform([dummy_recipe])
        return query_features[0]
    
    def _calculate_sustainability_boost(self) -> np.ndarray:
        """Fenntarthatósági pontszám alapú boost számítás"""
        weights = self.config['sustainability_weights']
        boosts = []
        
        for recipe in self.recipes_data:
            esi = float(recipe.get('ESI', 50))
            hsi = float(recipe.get('HSI', 50))
            ppi = float(recipe.get('PPI', 50))
            
            # Normalize to 0-1 scale
            esi_norm = esi / 100.0
            hsi_norm = hsi / 100.0
            ppi_norm = ppi / 100.0
            
            # Weighted boost (0-1 scale, ahol 1 = maximális boost)
            boost = (
                weights['ESI'] * esi_norm +
                weights['HSI'] * hsi_norm +
                weights['PPI'] * ppi_norm
            )
            
            boosts.append(boost * 0.5)  # Max 50% boost
        
        return np.array(boosts)
    
    def _apply_diversity(self, candidate_indices: np.ndarray, scores: np.ndarray, n_final: int) -> List[int]:
        """Diverzitás alkalmazása az ajánlásokban"""
        if len(candidate_indices) <= n_final:
            return candidate_indices.tolist()
        
        selected = [candidate_indices[0]]  # Start with highest scoring
        candidates = candidate_indices[1:].tolist()
        
        diversity_factor = self.config.get('diversity_factor', 0.1)
        
        while len(selected) < n_final and candidates:
            best_candidate = None
            best_score = -1
            
            for candidate in candidates:
                # Base score
                base_score = scores[candidate]
                
                # Diversity bonus (distance from already selected)
                diversity_bonus = 0
                for selected_idx in selected:
                    if candidate < len(self.recipe_features) and selected_idx < len(self.recipe_features):
                        sim = cosine_similarity(
                            self.recipe_features[candidate].reshape(1, -1),
                            self.recipe_features[selected_idx].reshape(1, -1)
                        )[0][0]
                        diversity_bonus += (1 - sim)  # Higher bonus for less similar items
                
                diversity_bonus = diversity_bonus / len(selected) if selected else 0
                
                # Combined score
                combined_score = base_score + (diversity_factor * diversity_bonus)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _format_recommendations(self, indices: List[int], final_scores: np.ndarray, similarities: np.ndarray) -> List[Dict[str, Any]]:
        """Ajánlások formázása a meglévő rendszerrel kompatibilis módon"""
        recommendations = []
        
        for i, idx in enumerate(indices):
            if idx >= len(self.recipes_data):
                continue
                
            recipe = self.recipes_data[idx].copy()
            
            # Add scoring information
            recipe['similarity_score'] = float(similarities[idx])
            recipe['final_score'] = float(final_scores[idx])
            recipe['recommendation_rank'] = i + 1
            
            # Ensure sustainability scores are present
            recipe['sustainability_score'] = (
                float(recipe.get('ESI', 50)) * 0.4 +
                float(recipe.get('HSI', 50)) * 0.3 +
                float(recipe.get('PPI', 50)) * 0.3
            )
            
            # Kompatibilitás a meglévő rendszerrel
            if 'composite_score' not in recipe:
                recipe['composite_score'] = recipe['sustainability_score']
            
            recommendations.append(recipe)
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Feature fontosság visszaadása debugging-hoz"""
        return {
            'total_features': self.recipe_features.shape[1],
            'recipes_count': len(self.recipes_data),
            'ingredient_vocab_size': len(getattr(self.feature_extractor.ingredient_vectorizer, 'vocabulary_', {})),
            'instruction_vocab_size': len(getattr(self.feature_extractor.instruction_vectorizer, 'vocabulary_', {}))
        }

# Utility funkciók a meglévő rendszerrel való kompatibilitáshoz
def create_enhanced_recommender(recipes_data: List[Dict[str, Any]], config: Dict[str, Any] = None) -> EnhancedContentBasedRecommender:
    """Factory function az Enhanced Recommender létrehozásához"""
    return EnhancedContentBasedRecommender(recipes_data, config)

def convert_old_recipe_format(old_recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Konvertálja a régi recept formátumot az új rendszerrel kompatibilissé"""
    converted = []
    
    for recipe in old_recipes:
        converted_recipe = recipe.copy()
        
        # Ensure required fields exist
        if 'ingredients' not in converted_recipe:
            converted_recipe['ingredients'] = ''
        
        if 'HSI' not in converted_recipe:
            converted_recipe['HSI'] = converted_recipe.get('hsi_normalized', 50)
        
        if 'ESI' not in converted_recipe:
            converted_recipe['ESI'] = converted_recipe.get('esi_inverted', 50)
        
        if 'PPI' not in converted_recipe:
            converted_recipe['PPI'] = converted_recipe.get('ppi_normalized', 50)
        
        converted.append(converted_recipe)
    
    return converted

if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing Enhanced Content-Based Filtering...")
    
    # Sample data for testing
    sample_recipes = [
        {
            'id': '1',
            'name': 'Sustainable Veggie Bowl',
            'ingredients': 'quinoa spinach tomatoes olive oil',
            'HSI': 85,
            'ESI': 90,
            'PPI': 75
        },
        {
            'id': '2', 
            'name': 'Chicken Pasta',
            'ingredients': 'chicken pasta cheese butter',
            'HSI': 60,
            'ESI': 40,
            'PPI': 85
        }
    ]
    
    # Create recommender
    recommender = create_enhanced_recommender(sample_recipes)
    
    # Test recommendation
    user_prefs = {
        'ingredients': 'vegetables healthy',
        'health_preference': 80,
        'environmental_preference': 85
    }
    
    recommendations = recommender.recommend(user_prefs, n_recommendations=2)
    
    print(f"📊 Generated {len(recommendations)} test recommendations")
    for rec in recommendations:
        print(f"   - {rec['name']}: similarity={rec['similarity_score']:.2f}, sustainability={rec['sustainability_score']:.1f}")
    
    print("✅ Enhanced Content-Based Filtering test completed!")

#!/usr/bin/env python3
"""
Enhanced Routes Integration - A meglévő routes.py kiterjesztése
Enhanced Content-Based Filtering és Evaluation Metrics integrálásával
"""

import os
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Existing imports (ezek már megvannak a routes.py-ban)
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify

# Új modulok importálása
try:
    from enhanced_content_based import EnhancedContentBasedRecommender, create_enhanced_recommender, convert_old_recipe_format
    from evaluation_metrics import RecommendationEvaluator, MetricsTracker, create_evaluator
    ENHANCED_MODULES_AVAILABLE = True
    print("✅ Enhanced modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced modules not available: {e}")
    print("🔧 Falling back to original recommendation system")
    ENHANCED_MODULES_AVAILABLE = False

class EnhancedRecommendationEngine:
    """Továbbfejlesztett ajánló motor a meglévő rendszerrel kompatibilis interface-szel"""
    
    def __init__(self, recipes_data: List[Dict[str, Any]]):
        self.recipes_data = recipes_data
        self.enhanced_recommender = None
        self.evaluator = None
        self.metrics_tracker = MetricsTracker() if ENHANCED_MODULES_AVAILABLE else None
        
        # Initialize enhanced components if available
        if ENHANCED_MODULES_AVAILABLE:
            try:
                # Convert old format to new format
                converted_recipes = convert_old_recipe_format(recipes_data)
                
                # Create enhanced recommender
                self.enhanced_recommender = create_enhanced_recommender(converted_recipes)
                
                # Create evaluator
                self.evaluator = create_evaluator()
                
                print(f"✅ Enhanced Recommendation Engine initialized with {len(recipes_data)} recipes")
            except Exception as e:
                print(f"❌ Failed to initialize enhanced components: {e}")
                print("🔧 Falling back to original system")
                ENHANCED_MODULES_AVAILABLE = False
        
        # Fallback to original recommendation logic
        self.original_recipes = recipes_data
    
    def get_recommendations(self, 
                          user_input: str = "", 
                          user_preferences: Dict[str, Any] = None,
                          version: str = 'v3', 
                          n_recommendations: int = 5,
                          session_id: str = None) -> Dict[str, Any]:
        """
        Főbb ajánlási metódus - enhanced vagy original implementáció
        """
        
        # Prepare user preferences
        if user_preferences is None:
            user_preferences = {}
        
        # Add search query to preferences
        if user_input:
            user_preferences['search_query'] = user_input
            user_preferences['ingredients'] = user_input
        
        # Try enhanced recommendation first
        if ENHANCED_MODULES_AVAILABLE and self.enhanced_recommender:
            try:
                return self._get_enhanced_recommendations(
                    user_preferences, version, n_recommendations, session_id
                )
            except Exception as e:
                print(f"❌ Enhanced recommendation failed: {e}")
                print("🔧 Falling back to original system")
        
        # Fallback to original system
        return self._get_original_recommendations(user_input, version, n_recommendations)
    
    def _get_enhanced_recommendations(self, 
                                    user_preferences: Dict[str, Any],
                                    version: str,
                                    n_recommendations: int,
                                    session_id: str) -> Dict[str, Any]:
        """Enhanced recommendation system használata"""
        
        print(f"🚀 Using Enhanced Content-Based Filtering...")
        
        # Get recommendations from enhanced system
        enhanced_recs = self.enhanced_recommender.recommend(
            user_preferences=user_preferences,
            n_recommendations=n_recommendations
        )
        
        # Add version-specific information (kompatibilitás a meglévő rendszerrel)
        formatted_recs = []
        for rec in enhanced_recs:
            formatted_rec = rec.copy()
            
            if version == 'v1':
                formatted_rec['show_scores'] = False
                formatted_rec['show_explanation'] = False
                formatted_rec['explanation'] = ""
            elif version == 'v2':
                formatted_rec['show_scores'] = True
                formatted_rec['show_explanation'] = False
                formatted_rec['explanation'] = ""
            elif version == 'v3':
                formatted_rec['show_scores'] = True
                formatted_rec['show_explanation'] = True
                formatted_rec['explanation'] = self._generate_enhanced_explanation(formatted_rec, user_preferences)
            
            formatted_recs.append(formatted_rec)
        
        # Evaluate recommendations
        evaluation_metrics = None
        if self.evaluator:
            try:
                evaluation_metrics = self.evaluator.evaluate_recommendations(
                    recommendations=enhanced_recs,
                    ground_truth=None,  # TODO: Add ground truth if available
                    user_profile=user_preferences,
                    session_id=session_id
                )
                
                # Track metrics
                if self.metrics_tracker:
                    self.metrics_tracker.track_recommendation(session_id or 'anonymous', evaluation_metrics)
                
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
        
        return {
            'recommendations': formatted_recs,
            'algorithm': 'enhanced_content_based',
            'metrics': evaluation_metrics,
            'feature_importance': self.enhanced_recommender.get_feature_importance() if self.enhanced_recommender else None,
            'version': version,
            'total_recipes': len(self.recipes_data)
        }
    
    def _get_original_recommendations(self, user_input: str, version: str, n_recommendations: int) -> Dict[str, Any]:
        """Eredeti ajánló rendszer használata fallback-ként"""
        
        print(f"🔄 Using Original Recommendation System...")
        
        # Simple search in original recipes (simplified implementation)
        if user_input.strip():
            # Basic keyword search
            candidates = []
            search_terms = user_input.lower().split()
            
            for recipe in self.original_recipes:
                ingredients = str(recipe.get('ingredients', '')).lower()
                name = str(recipe.get('name', '')).lower()
                
                score = 0
                for term in search_terms:
                    if term in ingredients:
                        score += 2
                    if term in name:
                        score += 1
                
                if score > 0:
                    recipe_copy = recipe.copy()
                    recipe_copy['similarity_score'] = score / len(search_terms)
                    candidates.append(recipe_copy)
            
            # Sort by score
            candidates.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            recommendations = candidates[:n_recommendations]
        else:
            # Top recipes by composite score
            sorted_recipes = sorted(
                self.original_recipes, 
                key=lambda x: x.get('composite_score', 0), 
                reverse=True
            )
            recommendations = sorted_recipes[:n_recommendations]
        
        # Add version-specific formatting
        for rec in recommendations:
            if version == 'v1':
                rec['show_scores'] = False
                rec['show_explanation'] = False
                rec['explanation'] = ""
            elif version == 'v2':
                rec['show_scores'] = True
                rec['show_explanation'] = False
                rec['explanation'] = ""
            elif version == 'v3':
                rec['show_scores'] = True
                rec['show_explanation'] = True
                rec['explanation'] = self._generate_original_explanation(rec, user_input)
        
        return {
            'recommendations': recommendations,
            'algorithm': 'original_hybrid',
            'metrics': None,
            'version': version,
            'total_recipes': len(self.original_recipes)
        }
    
    def _generate_enhanced_explanation(self, recipe: Dict[str, Any], user_preferences: Dict[str, Any]) -> str:
        """Enhanced explanation generation"""
        similarity = recipe.get('similarity_score', 0)
        sustainability = recipe.get('sustainability_score', 70)
        esi = recipe.get('ESI', 70)
        hsi = recipe.get('HSI', 70)
        
        explanation = f"🎯 {similarity:.2f} hasonlóság alapján ajánljuk "
        explanation += f"({sustainability:.1f}/100 fenntarthatósági pontszám). "
        
        if esi >= 80:
            explanation += "🌱 Kiváló környezeti hatás! "
        elif esi >= 60:
            explanation += "🌿 Jó környezeti értékelés. "
        else:
            explanation += "🔸 Közepes környezeti hatás. "
        
        if hsi >= 80:
            explanation += "💚 Kiváló tápértékkel. "
        elif hsi >= 60:
            explanation += "⚖️ Kiegyensúlyozott összetevőkkel. "
        
        # Add search relevance explanation
        search_query = user_preferences.get('search_query', '')
        if search_query:
            explanation += f"✨ Illeszkedik a '{search_query}' kereséshez."
        
        return explanation
    
    def _generate_original_explanation(self, recipe: Dict[str, Any], user_input: str) -> str:
        """Original explanation generation (kompatibilitás)"""
        composite = recipe.get('composite_score', 70)
        esi = recipe.get('ESI', 70)
        hsi = recipe.get('HSI', 70)
        
        explanation = f"🎯 Ezt a receptet {composite:.1f}/100 összpontszám alapján ajánljuk. "
        
        if esi >= 80:
            explanation += "🌱 Kiváló környezeti értékeléssel! "
        elif esi >= 60:
            explanation += "🌿 Jó környezeti értékeléssel. "
        else:
            explanation += "🔸 Közepes környezeti hatással. "
            
        if hsi >= 80:
            explanation += "💚 Kiváló tápanyag-értékkel. "
        elif hsi >= 60:
            explanation += "⚖️ Kiegyensúlyozott összetevőkkel. "
        
        if user_input.strip():
            explanation += f"✨ Illeszkedik a '{user_input}' kereséshez."
        
        return explanation
    
    def get_metrics_dashboard_data(self) -> Dict[str, Any]:
        """Dashboard metrikák lekérdezése"""
        if self.metrics_tracker:
            return self.metrics_tracker.get_dashboard_data()
        return {'message': 'Enhanced metrics not available'}
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Értékelési összefoglaló"""
        if self.evaluator:
            return self.evaluator.get_evaluation_summary()
        return {'message': 'Enhanced evaluation not available'}

# Flask route módosítások a meglévő routes.py integrálásához
def create_enhanced_routes():
    """Enhanced route-ok létrehozása"""
    
    # Globals for the enhanced system
    enhanced_engine = None
    
    def initialize_enhanced_engine(recipes_data):
        """Enhanced engine inicializálása"""
        global enhanced_engine
        if enhanced_engine is None:
            enhanced_engine = EnhancedRecommendationEngine(recipes_data)
        return enhanced_engine
    
    # New API endpoints for enhanced features
    def add_enhanced_routes(app_or_blueprint):
        """Add enhanced routes to existing Flask app or blueprint"""
        
        @app_or_blueprint.route('/api/enhanced-recommendations', methods=['POST'])
        def enhanced_recommendations():
            """Enhanced recommendations API endpoint"""
            try:
                data = request.get_json() or {}
                
                # Get parameters
                user_input = data.get('search_query', '')
                user_preferences = data.get('user_preferences', {})
                version = data.get('version', 'v3')
                n_recommendations = data.get('n_recommendations', 5)
                session_id = session.get('user_id', 'anonymous')
                
                # Get enhanced engine (assuming recipes are loaded globally)
                # This would need to be adapted to your specific recipe loading mechanism
                engine = initialize_enhanced_engine([])  # You'll need to pass actual recipes here
                
                # Get recommendations
                results = engine.get_recommendations(
                    user_input=user_input,
                    user_preferences=user_preferences,
                    version=version,
                    n_recommendations=n_recommendations,
                    session_id=session_id
                )
                
                return jsonify({
                    'status': 'success',
                    'data': results
                })
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @app_or_blueprint.route('/api/metrics-dashboard')
        def metrics_dashboard():
            """Metrics dashboard API endpoint"""
            try:
                if enhanced_engine:
                    dashboard_data = enhanced_engine.get_metrics_dashboard_data()
                    return jsonify({
                        'status': 'success',
                        'data': dashboard_data
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Enhanced metrics not available'
                    }), 503
                    
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @app_or_blueprint.route('/api/evaluation-summary')
        def evaluation_summary():
            """Evaluation summary API endpoint"""
            try:
                if enhanced_engine:
                    summary = enhanced_engine.get_evaluation_summary()
                    return jsonify({
                        'status': 'success',
                        'data': summary
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Enhanced evaluation not available'
                    }), 503
                    
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @app_or_blueprint.route('/dashboard')
        def metrics_dashboard_page():
            """Metrics dashboard page"""
            try:
                dashboard_data = {}
                evaluation_summary = {}
                
                if enhanced_engine:
                    dashboard_data = enhanced_engine.get_metrics_dashboard_data()
                    evaluation_summary = enhanced_engine.get_evaluation_summary()
                
                return render_template('metrics_dashboard.html', 
                                     dashboard_data=dashboard_data,
                                     evaluation_summary=evaluation_summary,
                                     enhanced_available=ENHANCED_MODULES_AVAILABLE)
                                     
            except Exception as e:
                return render_template('error.html', 
                                     error_message=f"Dashboard error: {e}")
    
    return add_enhanced_routes

# Módosított RecommendationEngine osztály a meglévő routes.py-hoz
class ModifiedRecommendationEngine:
    """
    Módosított verzió a meglévő RecommendationEngine osztálynak
    Enhanced funkcionalitással kiegészítve
    """
    
    def __init__(self, recipes):
        # Original initialization
        self.recipes = recipes
        
        # Enhanced initialization
        self.enhanced_engine = None
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.enhanced_engine = EnhancedRecommendationEngine(recipes)
                print("✅ Enhanced engine integrated successfully")
            except Exception as e:
                print(f"⚠️ Enhanced engine integration failed: {e}")
    
    def recommend(self, search_query="", n_recommendations=5, version='v3'):
        """
        Módosított recommend metódus enhanced funkcionalitással
        Backward compatible a meglévő kóddal
        """
        
        # Try enhanced recommendations first
        if self.enhanced_engine:
            try:
                session_id = session.get('user_id', 'anonymous') if 'session' in globals() else 'anonymous'
                
                results = self.enhanced_engine.get_recommendations(
                    user_input=search_query,
                    version=version,
                    n_recommendations=n_recommendations,
                    session_id=session_id
                )
                
                # Return in original format for compatibility
                recommendations = results['recommendations']
                
                # Add original format compatibility
                for rec in recommendations:
                    if 'show_scores' not in rec:
                        rec['show_scores'] = version in ['v2', 'v3']
                    if 'show_explanation' not in rec:
                        rec['show_explanation'] = version == 'v3'
                
                print(f"✅ Enhanced recommendations: {len(recommendations)} items")
                return recommendations
                
            except Exception as e:
                print(f"❌ Enhanced recommendation failed: {e}")
                print("🔧 Falling back to original implementation")
        
        # Fallback to original implementation
        return self._original_recommend(search_query, n_recommendations, version)
    
    def _original_recommend(self, search_query="", n_recommendations=5, version='v3'):
        """Original recommendation implementation"""
        
        if not self.recipes:
            print("❌ No recipes available")
            return []
        
        print(f"🔍 Getting recommendations: {len(self.recipes)} total recipes available")
        
        # Search or top recipes (original logic)
        if search_query.strip():
            candidates = self._search_recipes(search_query, max_results=20)
            recommendations = candidates[:n_recommendations]
            print(f"🔍 Search '{search_query}' found {len(recommendations)} matches")
        else:
            # Best composite score recipes
            sorted_recipes = sorted(self.recipes, key=lambda x: x.get('composite_score', 0), reverse=True)
            recommendations = sorted_recipes[:n_recommendations]
            print(f"🏆 Top {len(recommendations)} recipes by score")
        
        if not recommendations:
            recommendations = self.recipes[:n_recommendations]
            print(f"⚠️ Fallback: using first {len(recommendations)} recipes")
        
        # Deep copy to avoid modifying original data
        recommendations = [recipe.copy() for recipe in recommendations]
        
        # Version-specific information
        for rec in recommendations:
            if version == 'v1':
                rec['show_scores'] = False
                rec['show_explanation'] = False
                rec['explanation'] = ""
            elif version == 'v2':
                rec['show_scores'] = True
                rec['show_explanation'] = False  
                rec['explanation'] = ""
            elif version == 'v3':
                rec['show_scores'] = True
                rec['show_explanation'] = True
                rec['explanation'] = self._generate_explanation(rec, search_query)
        
        print(f"✅ Returning {len(recommendations)} recommendations")
        return recommendations
    
    def _search_recipes(self, search_query, max_results=20):
        """Simple search implementation (original)"""
        candidates = []
        search_terms = search_query.lower().split()
        
        for recipe in self.recipes:
            ingredients = str(recipe.get('ingredients', '')).lower()
            name = str(recipe.get('name', '')).lower()
            
            score = 0
            for term in search_terms:
                if term in ingredients:
                    score += 2
                if term in name:
                    score += 1
            
            if score > 0:
                recipe_copy = recipe.copy()
                recipe_copy['similarity_score'] = score / len(search_terms)
                candidates.append(recipe_copy)
        
        candidates.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return candidates[:max_results]
    
    def _generate_explanation(self, recipe, search_query=""):
        """Original explanation generation"""
        composite = recipe.get('composite_score', 70)
        esi = recipe.get('ESI', 70)
        hsi = recipe.get('HSI', 70)
        category = recipe.get('category', '')
        
        explanation = f"🎯 Ezt a receptet {composite:.1f}/100 összpontszám alapján ajánljuk. "
        
        if esi >= 80:
            explanation += "🌱 Kiváló környezeti értékeléssel! "
        elif esi >= 60:
            explanation += "🌿 Jó környezeti értékeléssel. "
        else:
            explanation += "🔸 Közepes környezeti hatással. "
            
        if hsi >= 80:
            explanation += "💚 Kiváló tápanyag-értékkel. "
        elif hsi >= 60:
            explanation += "⚖️ Kiegyensúlyozott összetevőkkel. "
        
        if category:
            explanation += f"🏷️ Kategória: {category}. "
        
        if search_query.strip():
            explanation += f"✨ Illeszkedik a '{search_query}' kereséshez."
        
        return explanation
    
    def get_enhanced_metrics(self):
        """Enhanced metrikák lekérdezése"""
        if self.enhanced_engine:
            return self.enhanced_engine.get_metrics_dashboard_data()
        return None

# Integration helper functions
def integrate_enhanced_system(existing_app, recipes_data):
    """
    Helper function az enhanced system integrálásához a meglévő alkalmazásba
    """
    
    # Add enhanced routes
    enhanced_routes = create_enhanced_routes()
    enhanced_routes(existing_app)
    
    # Replace or enhance existing recommendation engine
    # This would be done in the main routes.py file where the engine is created
    
    print("✅ Enhanced system integrated successfully")
    
    return {
        'enhanced_available': ENHANCED_MODULES_AVAILABLE,
        'routes_added': ['api/enhanced-recommendations', 'api/metrics-dashboard', '/dashboard'],
        'features': [
            'Enhanced Content-Based Filtering',
            'Advanced Similarity Metrics',
            'Comprehensive Evaluation',
            'Real-time Metrics Dashboard'
        ]
    }

# Configuration for enhanced features
ENHANCED_CONFIG = {
    'feature_extraction': {
        'ingredient_max_features': 2000,
        'ingredient_ngram_range': (1, 2),
        'instruction_max_features': 1000,
        'min_df': 2,
        'max_df': 0.95,
        'weights': {
            'ingredients': 0.6,
            'instructions': 0.2,
            'numeric': 0.2
        }
    },
    'similarity_weights': {
        'cosine': 0.7,
        'euclidean': 0.2,
        'correlation': 0.1
    },
    'sustainability_boost': True,
    'sustainability_weights': {
        'ESI': 0.4,
        'HSI': 0.3,
        'PPI': 0.3
    },
    'evaluation': {
        'relevance_threshold': 4.0,
        'k_values': [5, 10, 20]
    }
}

if __name__ == "__main__":
    # Test integration
    print("🧪 Testing Enhanced Routes Integration...")
    
    # Sample recipes for testing
    sample_recipes = [
        {
            'id': '1',
            'name': 'Green Sustainability Bowl',
            'ingredients': 'quinoa, spinach, avocado, tomatoes',
            'HSI': 90,
            'ESI': 85,
            'PPI': 80,
            'composite_score': 85
        },
        {
            'id': '2',
            'name': 'Traditional Pasta',
            'ingredients': 'pasta, cheese, cream, herbs',
            'HSI': 55,
            'ESI': 40,
            'PPI': 85,
            'composite_score': 60
        }
    ]
    
    # Test enhanced engine
    engine = EnhancedRecommendationEngine(sample_recipes)
    
    # Test recommendations
    results = engine.get_recommendations(
        user_input="healthy vegetables",
        version='v3',
        n_recommendations=2,
        session_id='test_session'
    )
    
    print(f"📊 Test results:")
    print(f"   - Algorithm: {results['algorithm']}")
    print(f"   - Recommendations: {len(results['recommendations'])}")
    print(f"   - Metrics available: {results['metrics'] is not None}")
    
    # Test metrics
    dashboard_data = engine.get_metrics_dashboard_data()
    print(f"   - Dashboard metrics: {len(dashboard_data)} items")
    
    print("✅ Enhanced Routes Integration test completed!")

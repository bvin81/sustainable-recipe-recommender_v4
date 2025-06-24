#!/usr/bin/env python3
"""
Evaluation Metrics modulú a fenntartható recept ajánlórendszer teljesítményének mérésére
Precision, Recall, F1-score, Cosine Similarity, Diversity, Coverage metrikák
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import json
from datetime import datetime

# Conditional imports with fallbacks
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available, using fallback implementations")
    SKLEARN_AVAILABLE = False

def fallback_cosine_similarity(X, Y=None):
    """Fallback cosine similarity implementation"""
    if Y is None:
        Y = X
    
    # Ensure arrays are 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    
    # Normalize vectors
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    
    return np.dot(X_norm, Y_norm.T)

class RecommendationEvaluator:
    """Comprehensive ajánlórendszer értékelő osztály"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.evaluation_history = []
        
        # Use appropriate similarity function
        if SKLEARN_AVAILABLE:
            self.cosine_similarity = cosine_similarity
        else:
            self.cosine_similarity = fallback_cosine_similarity
    
    def _default_config(self) -> Dict[str, Any]:
        """Default evaluation configuration"""
        return {
            'relevance_threshold': 4.0,  # Minimum rating for relevance
            'k_values': [5, 10, 20],     # K values for @K metrics
            'diversity_feature_weights': {
                'category': 0.3,
                'ingredients': 0.4,
                'sustainability': 0.3
            }
        }
    
    def evaluate_recommendations(self, 
                                recommendations: List[Dict[str, Any]], 
                                ground_truth: Dict[str, float] = None,
                                user_profile: Dict[str, Any] = None,
                                session_id: str = None) -> Dict[str, Any]:
        """
        Komprehenzív ajánlás értékelés
        
        Args:
            recommendations: Lista az ajánlott receptekről
            ground_truth: Felhasználó valós értékelései {recipe_id: rating}
            user_profile: Felhasználói profil hasonlóság számításhoz
            session_id: Session azonosító tracking-hez
        
        Returns:
            Dict értékelési metrikákkal
        """
        
        print(f"📊 Evaluating {len(recommendations)} recommendations...")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'n_recommendations': len(recommendations)
        }
        
        # Basic metrics (always calculated)
        metrics.update(self._calculate_basic_metrics(recommendations))
        
        # Precision/Recall/F1 (csak ha van ground truth)
        if ground_truth:
            metrics.update(self._calculate_precision_recall_metrics(recommendations, ground_truth))
        
        # Diversity metrics
        metrics.update(self._calculate_diversity_metrics(recommendations))
        
        # Coverage metrics  
        metrics.update(self._calculate_coverage_metrics(recommendations))
        
        # Sustainability metrics
        metrics.update(self._calculate_sustainability_metrics(recommendations))
        
        # User profile similarity (ha van user profile)
        if user_profile:
            metrics.update(self._calculate_user_similarity_metrics(recommendations, user_profile))
        
        # Save evaluation
        self.evaluation_history.append(metrics)
        
        print(f"✅ Evaluation completed: {len(metrics)} metrics calculated")
        return metrics
    
    def _calculate_basic_metrics(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Alapszintű metrikák számítása"""
        if not recommendations:
            return {
                'avg_similarity_score': 0.0,
                'avg_final_score': 0.0,
                'avg_sustainability_score': 0.0
            }
        
        similarity_scores = [rec.get('similarity_score', 0) for rec in recommendations]
        final_scores = [rec.get('final_score', 0) for rec in recommendations]
        sustainability_scores = [rec.get('sustainability_score', 0) for rec in recommendations]
        
        return {
            'avg_similarity_score': np.mean(similarity_scores),
            'avg_final_score': np.mean(final_scores),
            'avg_sustainability_score': np.mean(sustainability_scores),
            'max_similarity_score': np.max(similarity_scores),
            'min_similarity_score': np.min(similarity_scores),
            'std_similarity_score': np.std(similarity_scores)
        }
    
    def _calculate_precision_recall_metrics(self, 
                                          recommendations: List[Dict[str, Any]], 
                                          ground_truth: Dict[str, float]) -> Dict[str, Any]:
        """Precision, Recall, F1-score számítása különböző K értékekre"""
        metrics = {}
        
        for k in self.config['k_values']:
            if k > len(recommendations):
                continue
                
            # Top-K recommendations
            top_k_recs = recommendations[:k]
            
            # Get recommended recipe IDs
            recommended_ids = []
            for rec in top_k_recs:
                recipe_id = str(rec.get('id', rec.get('recipeid', '')))
                recommended_ids.append(recipe_id)
            
            # Calculate precision@k
            precision_k = self.precision_at_k(recommended_ids, ground_truth, k)
            recall_k = self.recall_at_k(recommended_ids, ground_truth, k)
            f1_k = self.f1_score_at_k(precision_k, recall_k)
            
            metrics[f'precision_at_{k}'] = precision_k
            metrics[f'recall_at_{k}'] = recall_k
            metrics[f'f1_score_at_{k}'] = f1_k
        
        return metrics
    
    def precision_at_k(self, recommended_ids: List[str], ground_truth: Dict[str, float], k: int) -> float:
        """Precision@K számítás"""
        if not recommended_ids or k == 0:
            return 0.0
        
        threshold = self.config['relevance_threshold']
        relevant_items = set(
            item_id for item_id, rating in ground_truth.items() 
            if rating >= threshold
        )
        
        recommended_set = set(recommended_ids[:k])
        
        if not recommended_set:
            return 0.0
        
        relevant_recommended = len(relevant_items & recommended_set)
        return relevant_recommended / len(recommended_set)
    
    def recall_at_k(self, recommended_ids: List[str], ground_truth: Dict[str, float], k: int) -> float:
        """Recall@K számítás"""
        if not recommended_ids or k == 0:
            return 0.0
        
        threshold = self.config['relevance_threshold']
        relevant_items = set(
            item_id for item_id, rating in ground_truth.items() 
            if rating >= threshold
        )
        
        if not relevant_items:
            return 0.0
        
        recommended_set = set(recommended_ids[:k])
        relevant_recommended = len(relevant_items & recommended_set)
        
        return relevant_recommended / len(relevant_items)
    
    def f1_score_at_k(self, precision: float, recall: float) -> float:
        """F1-score számítás precision és recall alapján"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_diversity_metrics(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Diverzitás metrikák számítása"""
        if len(recommendations) < 2:
            return {
                'intra_list_diversity': 1.0,
                'category_diversity': 1.0,
                'ingredient_diversity': 1.0
            }
        
        # Kategória diverzitás
        categories = [rec.get('category', 'unknown') for rec in recommendations]
        category_diversity = len(set(categories)) / len(categories)
        
        # Összetevő diverzitás (egyedi összetevők száma)
        all_ingredients = []
        for rec in recommendations:
            ingredients = rec.get('ingredients', '')
            if isinstance(ingredients, str):
                # Simple tokenization
                ingredients_list = ingredients.lower().split()
                all_ingredients.extend(ingredients_list)
            elif isinstance(ingredients, list):
                all_ingredients.extend([str(ing).lower() for ing in ingredients])
        
        unique_ingredients = len(set(all_ingredients))
        total_ingredients = len(all_ingredients)
        ingredient_diversity = unique_ingredients / max(total_ingredients, 1)
        
        # Intra-list diversity (átlagos hasonlóság fordítottja)
        intra_diversity = self._calculate_intra_list_diversity(recommendations)
        
        return {
            'intra_list_diversity': intra_diversity,
            'category_diversity': category_diversity,
            'ingredient_diversity': ingredient_diversity,
            'unique_categories': len(set(categories)),
            'unique_ingredients': unique_ingredients
        }
    
    def _calculate_intra_list_diversity(self, recommendations: List[Dict[str, Any]]) -> float:
        """Intra-list diverzitás számítás feature-alapú hasonlóság alapján"""
        if len(recommendations) < 2:
            return 1.0
        
        # Create simple feature vectors for each recommendation
        feature_vectors = []
        for rec in recommendations:
            # Simple features: normalized sustainability scores
            features = [
                rec.get('HSI', 50) / 100.0,
                rec.get('ESI', 50) / 100.0,
                rec.get('PPI', 50) / 100.0,
                rec.get('similarity_score', 0.5)
            ]
            feature_vectors.append(features)
        
        feature_matrix = np.array(feature_vectors)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(feature_matrix)):
            for j in range(i + 1, len(feature_matrix)):
                sim = self.cosine_similarity(
                    feature_matrix[i].reshape(1, -1),
                    feature_matrix[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        return max(0.0, 1.0 - avg_similarity)
    
    def _calculate_coverage_metrics(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Lefedettség metrikák számítása"""
        # Kategória lefedettség
        categories = set(rec.get('category', 'unknown') for rec in recommendations)
        
        # Összetevő lefedettség 
        ingredients = set()
        for rec in recommendations:
            rec_ingredients = rec.get('ingredients', '')
            if isinstance(rec_ingredients, str):
                ingredients.update(rec_ingredients.lower().split())
            elif isinstance(rec_ingredients, list):
                ingredients.update(str(ing).lower() for ing in rec_ingredients)
        
        return {
            'category_coverage': len(categories),
            'ingredient_coverage': len(ingredients),
            'recipe_ids_coverage': len(set(str(rec.get('id', rec.get('recipeid', i))) 
                                          for i, rec in enumerate(recommendations)))
        }
    
    def _calculate_sustainability_metrics(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fenntarthatósági metrikák számítása"""
        if not recommendations:
            return {
                'avg_hsi': 0.0,
                'avg_esi': 0.0,
                'avg_ppi': 0.0,
                'sustainability_distribution': {}
            }
        
        hsi_scores = [rec.get('HSI', 50) for rec in recommendations]
        esi_scores = [rec.get('ESI', 50) for rec in recommendations]
        ppi_scores = [rec.get('PPI', 50) for rec in recommendations]
        
        # Sustainability distribution (high/medium/low)
        sustainability_levels = []
        for rec in recommendations:
            avg_score = (rec.get('HSI', 50) + rec.get('ESI', 50) + rec.get('PPI', 50)) / 3
            if avg_score >= 80:
                sustainability_levels.append('high')
            elif avg_score >= 60:
                sustainability_levels.append('medium')
            else:
                sustainability_levels.append('low')
        
        sustainability_dist = Counter(sustainability_levels)
        
        return {
            'avg_hsi': np.mean(hsi_scores),
            'avg_esi': np.mean(esi_scores),
            'avg_ppi': np.mean(ppi_scores),
            'std_hsi': np.std(hsi_scores),
            'std_esi': np.std(esi_scores),
            'std_ppi': np.std(ppi_scores),
            'sustainability_distribution': dict(sustainability_dist),
            'high_sustainability_ratio': sustainability_dist.get('high', 0) / len(recommendations)
        }
    
    def _calculate_user_similarity_metrics(self, 
                                         recommendations: List[Dict[str, Any]], 
                                         user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Felhasználói profil hasonlóság metrikák"""
        if not recommendations or not user_profile:
            return {'user_profile_similarity': 0.0}
        
        # Create user profile vector
        user_vector = [
            user_profile.get('health_preference', 70) / 100.0,
            user_profile.get('environmental_preference', 70) / 100.0,
            user_profile.get('taste_preference', 70) / 100.0
        ]
        
        # Calculate similarity with each recommendation
        similarities = []
        for rec in recommendations:
            rec_vector = [
                rec.get('HSI', 50) / 100.0,
                rec.get('ESI', 50) / 100.0,
                rec.get('PPI', 50) / 100.0
            ]
            
            sim = self.cosine_similarity(
                np.array(user_vector).reshape(1, -1),
                np.array(rec_vector).reshape(1, -1)
            )[0][0]
            
            similarities.append(sim)
        
        return {
            'user_profile_similarity': np.mean(similarities),
            'max_user_similarity': np.max(similarities),
            'min_user_similarity': np.min(similarities)
        }
    
    def get_evaluation_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Utolsó N értékelés összefoglalása"""
        if not self.evaluation_history:
            return {'message': 'No evaluations available'}
        
        recent_evaluations = self.evaluation_history[-last_n:]
        
        # Aggregate metrics
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'recent_evaluations': len(recent_evaluations),
            'average_metrics': {}
        }
        
        # Calculate averages for numeric metrics
        numeric_metrics = [
            'avg_similarity_score', 'avg_final_score', 'avg_sustainability_score',
            'precision_at_10', 'recall_at_10', 'f1_score_at_10',
            'intra_list_diversity', 'category_diversity', 'ingredient_diversity',
            'avg_hsi', 'avg_esi', 'avg_ppi', 'high_sustainability_ratio'
        ]
        
        for metric in numeric_metrics:
            values = [eval_data.get(metric, 0) for eval_data in recent_evaluations 
                     if metric in eval_data]
            if values:
                summary['average_metrics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary
    
    def export_evaluation_data(self, filepath: str = None) -> Dict[str, Any]:
        """Értékelési adatok exportálása"""
        export_data = {
            'config': self.config,
            'evaluation_history': self.evaluation_history,
            'export_timestamp': datetime.now().isoformat(),
            'total_evaluations': len(self.evaluation_history)
        }
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Evaluation data exported to {filepath}")
            except Exception as e:
                print(f"❌ Export failed: {e}")
        
        return export_data

class MetricsTracker:
    """Valós idejű metrika tracking az alkalmazáshoz"""
    
    def __init__(self):
        self.session_metrics = {}
        self.global_metrics = {
            'total_recommendations': 0,
            'total_sessions': 0,
            'avg_metrics': {}
        }
    
    def track_recommendation(self, session_id: str, metrics: Dict[str, Any]):
        """Ajánlás metrikák tracking-je"""
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = []
            self.global_metrics['total_sessions'] += 1
        
        self.session_metrics[session_id].append(metrics)
        self.global_metrics['total_recommendations'] += 1
        
        # Update running averages
        self._update_global_averages(metrics)
    
    def _update_global_averages(self, new_metrics: Dict[str, Any]):
        """Globális átlagok frissítése"""
        numeric_keys = [k for k, v in new_metrics.items() 
                       if isinstance(v, (int, float)) and not k.endswith('_count')]
        
        for key in numeric_keys:
            if key not in self.global_metrics['avg_metrics']:
                self.global_metrics['avg_metrics'][key] = {
                    'sum': 0,
                    'count': 0,
                    'average': 0
                }
            
            self.global_metrics['avg_metrics'][key]['sum'] += new_metrics[key]
            self.global_metrics['avg_metrics'][key]['count'] += 1
            self.global_metrics['avg_metrics'][key]['average'] = (
                self.global_metrics['avg_metrics'][key]['sum'] / 
                self.global_metrics['avg_metrics'][key]['count']
            )
    
    def get_session_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        """Session metrikák lekérdezése"""
        return self.session_metrics.get(session_id, [])
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Globális metrikák lekérdezése"""
        return self.global_metrics
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Dashboard-hoz optimalizált metrika adatok"""
        avg_metrics = self.global_metrics.get('avg_metrics', {})
        
        dashboard_data = {
            'total_recommendations': self.global_metrics['total_recommendations'],
            'total_sessions': self.global_metrics['total_sessions'],
            'key_metrics': {}
        }
        
        # Key metrics for dashboard
        key_metric_names = {
            'avg_similarity_score': 'Átlagos Hasonlóság',
            'precision_at_10': 'Precision@10',
            'intra_list_diversity': 'Diverzitás',
            'avg_sustainability_score': 'Fenntarthatóság',
            'high_sustainability_ratio': 'Magas Fenntarthatóság %'
        }
        
        for metric_key, display_name in key_metric_names.items():
            if metric_key in avg_metrics:
                dashboard_data['key_metrics'][display_name] = {
                    'value': round(avg_metrics[metric_key]['average'], 3),
                    'count': avg_metrics[metric_key]['count']
                }
        
        return dashboard_data

# Utility functions
def create_evaluator(config: Dict[str, Any] = None) -> RecommendationEvaluator:
    """Factory function az evaluator létrehozásához"""
    return RecommendationEvaluator(config)

def quick_evaluate(recommendations: List[Dict[str, Any]], 
                  ground_truth: Dict[str, float] = None) -> Dict[str, Any]:
    """Gyors értékelés utility function"""
    evaluator = create_evaluator()
    return evaluator.evaluate_recommendations(recommendations, ground_truth)

if __name__ == "__main__":
    # Test the evaluation system
    print("🧪 Testing Evaluation Metrics...")
    
    # Sample recommendations
    sample_recommendations = [
        {
            'id': '1',
            'name': 'Green Bowl',
            'similarity_score': 0.85,
            'final_score': 0.9,
            'sustainability_score': 88,
            'HSI': 90,
            'ESI': 85,
            'PPI': 75,
            'category': 'healthy',
            'ingredients': 'quinoa spinach tomatoes'
        },
        {
            'id': '2',
            'name': 'Pasta Dish',
            'similarity_score': 0.7,
            'final_score': 0.75,
            'sustainability_score': 65,
            'HSI': 60,
            'ESI': 70,
            'PPI': 80,
            'category': 'comfort',
            'ingredients': 'pasta cheese herbs'
        }
    ]
    
    # Sample ground truth
    ground_truth = {'1': 4.5, '2': 3.8}
    
    # Test evaluation
    evaluator = create_evaluator()
    results = evaluator.evaluate_recommendations(sample_recommendations, ground_truth)
    
    print(f"📊 Evaluation results: {len(results)} metrics calculated")
    print(f"   - Precision@10: {results.get('precision_at_10', 'N/A')}")
    print(f"   - Diversity: {results.get('intra_list_diversity', 'N/A'):.2f}")
    print(f"   - Avg Sustainability: {results.get('avg_sustainability_score', 'N/A'):.1f}")
    
    print("✅ Evaluation Metrics test completed!")

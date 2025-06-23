#!/usr/bin/env python3
"""
Heroku-optimalizált user_study/routes.py
Memória-alapú adatbázis + egyszerűsített logika
"""

import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify

# Blueprint és paths
user_study_bp = Blueprint('user_study', __name__, url_prefix='')

# Heroku-kompatibilis data directory
if os.environ.get('DYNO'):
    # Heroku-n: munkakönytár használata
    project_root = Path.cwd()
else:
    # Helyi fejlesztéshez
    project_root = Path(__file__).parent.parent

data_dir = project_root / "data"

print(f"🔧 Data directory: {data_dir}")
print(f"🔧 Project root: {project_root}")

# =============================================================================
# EGYSZERŰSÍTETT ADATBÁZIS - MEMÓRIÁBAN
# =============================================================================

class MemoryDatabase:
    """Egyszerű memória-alapú adatbázis Heroku-hoz"""
    
    def __init__(self):
        self.participants = []
        self.interactions = []
        self.questionnaires = []
        self.next_user_id = 1
        print("✅ Memory database initialized")
    
    def create_user(self, age_group, education, cooking_frequency, sustainability_awareness, version):
        user_id = self.next_user_id
        self.next_user_id += 1
        
        user = {
            'user_id': user_id,
            'age_group': age_group,
            'education': education,
            'cooking_frequency': cooking_frequency,
            'sustainability_awareness': sustainability_awareness,
            'version': version,
            'is_completed': False
        }
        self.participants.append(user)
        return user_id
    
    def log_interaction(self, user_id, recipe_id, rating, explanation_helpful, view_time):
        interaction = {
            'user_id': user_id,
            'recipe_id': recipe_id,
            'rating': rating,
            'explanation_helpful': explanation_helpful,
            'view_time_seconds': view_time
        }
        self.interactions.append(interaction)
    
    def save_questionnaire(self, user_id, responses):
        questionnaire = {'user_id': user_id, **responses}
        self.questionnaires.append(questionnaire)
        
        # User completed jelölése
        for user in self.participants:
            if user['user_id'] == user_id:
                user['is_completed'] = True
                break
    
    def get_stats(self):
        total = len(self.participants)
        completed = sum(1 for u in self.participants if u['is_completed'])
        
        return {
            'total_participants': total,
            'completed_participants': completed,
            'completion_rate': completed / total if total > 0 else 0,
            'avg_interactions_per_user': len(self.interactions) / total if total > 0 else 0,
            'version_distribution': []
        }

# =============================================================================
# EGYSZERŰSÍTETT AJÁNLÓRENDSZER
# =============================================================================

class SimpleRecommender:
    """Egyszerű, Heroku-kompatibilis ajánlórendszer"""
    
    def __init__(self):
        self.recipes = self._create_sample_recipes()
        print(f"✅ Recommender initialized with {len(self.recipes)} recipes")
    
    def _create_sample_recipes(self):
        """Hardcoded sample receptek Heroku-hoz"""
        recipes = [
            {
                'recipeid': 1, 'title': 'Gulyásleves',
                'ingredients': 'marhahús, hagyma, paprika, paradicsom, burgonya',
                'instructions': 'Pirítsd meg a hagymát, add hozzá a húst, majd a zöldségeket. Főzd 1 órán át.',
                'images': 'https://images.unsplash.com/photo-1547592180-85f173990554?w=400',
                'ESI': 45, 'HSI': 75, 'PPI': 85, 'composite_score': 68
            },
            {
                'recipeid': 2, 'title': 'Vegetáriánus Lecsó',
                'ingredients': 'paprika, paradicsom, hagyma, tojás',
                'instructions': 'Pirítsd meg a hagymát, add hozzá a paprikát és paradicsomot. Keverj bele tojást.',
                'images': 'https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400',
                'ESI': 85, 'HSI': 80, 'PPI': 70, 'composite_score': 79
            },
            {
                'recipeid': 3, 'title': 'Halászlé',
                'ingredients': 'ponty, csuka, hagyma, paradicsom, paprika',
                'instructions': 'Főzd ki a halból a levest, szűrd le, majd add hozzá a fűszereket.',
                'images': 'https://images.unsplash.com/photo-1544943910-4c1dc44aab44?w=400',
                'ESI': 60, 'HSI': 85, 'PPI': 75, 'composite_score': 73
            },
            {
                'recipeid': 4, 'title': 'Túrós Csusza',
                'ingredients': 'széles metélt, túró, tejföl, szalonna',
                'instructions': 'Főzd meg a tésztát, keverd össze túróval és tejföllel.',
                'images': 'https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400',
                'ESI': 55, 'HSI': 65, 'PPI': 80, 'composite_score': 67
            },
            {
                'recipeid': 5, 'title': 'Gombapaprikás',
                'ingredients': 'gomba, hagyma, paprika, tejföl',
                'instructions': 'Pirítsd meg a gombát hagymával, add hozzá a paprikát és tejfölt.',
                'images': 'https://images.unsplash.com/photo-1565299507177-b0ac66763828?w=400',
                'ESI': 90, 'HSI': 70, 'PPI': 65, 'composite_score': 75
            }
        ]
        return recipes
    
    def search_recipes(self, query, max_results=5):
        """Egyszerű keresés"""
        if not query.strip():
            return list(range(len(self.recipes)))
        
        search_terms = [term.strip().lower() for term in query.split(',')]
        matching_indices = []
        
        for idx, recipe in enumerate(self.recipes):
            ingredients = recipe['ingredients'].lower()
            if any(term in ingredients for term in search_terms if term):
                matching_indices.append(idx)
        
        return matching_indices[:max_results]
    
    def get_recommendations(self, version='v1', search_query="", n_recommendations=5):
        """Fő ajánlási algoritmus"""
        
        # Keresés vagy véletlenszerű
        if search_query.strip():
            indices = self.search_recipes(search_query, n_recommendations)
        else:
            indices = list(range(min(n_recommendations, len(self.recipes))))
        
        recommendations = [self.recipes[i].copy() for i in indices]
        
        # Verzió-specifikus információ
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
                rec['explanation'] = f"Ezt a receptet {rec['composite_score']}/100 pontszám alapján ajánljuk. 🌱 Környezetbarát és egészséges választás!"
        
        return recommendations

# =============================================================================
# GLOBÁLIS OBJEKTUMOK
# =============================================================================

db = MemoryDatabase()
recommender = SimpleRecommender()

def get_user_version():
    """A/B/C verzió kiválasztása"""
    if 'version' not in session:
        session['version'] = random.choice(['v1', 'v2', 'v3'])
    return session['version']

# =============================================================================
# ROUTE-OK
# =============================================================================

@user_study_bp.route('/')
@user_study_bp.route('/welcome')
def welcome():
    return render_template('welcome.html')

@user_study_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            age_group = request.form.get('age_group')
            education = request.form.get('education')
            cooking_frequency = request.form.get('cooking_frequency')
            sustainability_awareness = int(request.form.get('sustainability_awareness', 3))
            
            version = get_user_version()
            user_id = db.create_user(age_group, education, cooking_frequency, 
                                   sustainability_awareness, version)
            
            session['user_id'] = user_id
            session['version'] = version
            
            return redirect(url_for('user_study.instructions'))
            
        except Exception as e:
            print(f"Registration error: {e}")
            return render_template('register.html', error='Regisztráció sikertelen')
    
    return render_template('register.html')

@user_study_bp.route('/instructions')
def instructions():
    if 'user_id' not in session:
        return redirect(url_for('user_study.register'))
    
    version = session.get('version', 'v1')
    return render_template('instructions.html', version=version)

@user_study_bp.route('/study')
def study():
    if 'user_id' not in session:
        return redirect(url_for('user_study.register'))
    
    version = session.get('version', 'v1')
    search_query = request.args.get('search', '').strip()
    
    recommendations = recommender.get_recommendations(
        version=version, 
        search_query=search_query,
        n_recommendations=5
    )
    
    return render_template('study.html', 
                         recommendations=recommendations, 
                         version=version,
                         search_term=search_query)

@user_study_bp.route('/rate_recipe', methods=['POST'])
def rate_recipe():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    data = request.get_json()
    
    recipe_id = int(data.get('recipe_id'))
    rating = int(data.get('rating'))
    explanation_helpful = data.get('explanation_helpful')
    view_time = data.get('view_time_seconds', 0)
    
    db.log_interaction(user_id, recipe_id, rating, explanation_helpful, view_time)
    
    return jsonify({'status': 'success'})

@user_study_bp.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'user_id' not in session:
        return redirect(url_for('user_study.register'))
    
    if request.method == 'POST':
        user_id = session['user_id']
        
        responses = {
            'system_usability': request.form.get('system_usability'),
            'recommendation_quality': request.form.get('recommendation_quality'),
            'trust_level': request.form.get('trust_level'),
            'explanation_clarity': request.form.get('explanation_clarity'),
            'sustainability_importance': request.form.get('sustainability_importance'),
            'overall_satisfaction': request.form.get('overall_satisfaction'),
            'additional_comments': request.form.get('additional_comments', '')
        }
        
        db.save_questionnaire(user_id, responses)
        return redirect(url_for('user_study.thank_you'))
    
    version = session.get('version', 'v1')
    return render_template('questionnaire.html', version=version)

@user_study_bp.route('/thank_you')
def thank_you():
    version = session.get('version', 'v1')
    return render_template('thank_you.html', version=version)

@user_study_bp.route('/admin/stats')
def admin_stats():
    try:
        stats = db.get_stats()
        return render_template('admin_stats.html', stats=stats)
    except Exception as e:
        return f"Stats error: {e}", 500

# Export
__all__ = ['user_study_bp']

print("✅ User study routes loaded successfully")

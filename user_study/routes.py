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
import json

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

# CSAK EZT A RÉSZT CSERÉLD KI az eredeti routes.py-ban:

# ============================================================================= 
# RÉGI SimpleRecommender osztály törlése (kb. 76-180. sor)
# =============================================================================
# class SimpleRecommender:      <- EZ KERÜL KI
#     def __init__(self):       <- EGÉSZ OSZTÁLY TÖRLÉSE
#     ... (teljes osztály)
#     ... minden metódusával

# =============================================================================
# ÚJ HungarianCSVRecommender osztály berakása helyette
# =============================================================================

class HungarianJSONRecommender:
    """Magyar receptek JSON-ből - encoding problémák nélkül"""
    
    def __init__(self):
        self.recipes = []
        self.load_hungarian_json()
        print(f"✅ Hungarian JSON Recommender initialized with {len(self.recipes)} recipes")
    
    def load_hungarian_json(self):
        """JSON fájl betöltése - 100% megbízható"""
        json_paths = [
            "hungarian_recipes.json",
            project_root / "hungarian_recipes.json",
            data_dir / "hungarian_recipes.json"
        ]
        
        for json_path in json_paths:
            if Path(json_path).exists():
                try:
                    print(f"📊 Loading JSON from: {json_path}")
                    
                    with open(json_path, 'r', encoding='utf-8') as f:
                        recipes_data = json.load(f)
                    
                    print(f"✅ JSON loaded! {len(recipes_data)} recipes")
                    
                    # Adatok feldolgozása
                    self.process_json_recipes(recipes_data)
                    return
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {json_path}: {e}")
                    continue
        
        # Fallback
        print("🔧 No JSON found, using sample recipes")
        self.create_sample_recipes()
    
    def process_json_recipes(self, recipes_data):
        """JSON receptek feldolgozása"""
        print(f"🔄 Processing {len(recipes_data)} recipes from JSON...")
        
        processed_recipes = []
        
        for recipe in recipes_data:
            try:
                # Alapvető mezők biztosítása
                processed_recipe = {
                    'recipeid': recipe.get('recipeid', 0),
                    'title': str(recipe.get('title', recipe.get('name', 'Névtelen recept'))),
                    'ingredients': str(recipe.get('ingredients', '')),
                    'instructions': str(recipe.get('instructions', '')),
                    'category': str(recipe.get('category', 'Egyéb')),
                    'images': self.fix_image_url(recipe.get('images', '')),
                    'ESI': float(recipe.get('ESI', recipe.get('env_score', 70))),
                    'HSI': float(recipe.get('HSI', recipe.get('nutri_score', 70))),
                    'PPI': float(recipe.get('PPI', recipe.get('meal_score', 70)))
                }
                
                # Composite score számítása
                processed_recipe['composite_score'] = round(
                    processed_recipe['ESI'] * 0.4 + 
                    processed_recipe['HSI'] * 0.4 + 
                    processed_recipe['PPI'] * 0.2, 1
                )
                
                # Csak érvényes receptek
                if processed_recipe['title'] and processed_recipe['ingredients']:
                    processed_recipes.append(processed_recipe)
                    
            except Exception as e:
                print(f"⚠️ Skipping invalid recipe: {e}")
                continue
        
        self.recipes = processed_recipes
        print(f"✅ Successfully processed {len(self.recipes)} recipes")
        
        # Debug: első recept megjelenítése
        if self.recipes:
            first = self.recipes[0]
            print(f"📝 Sample: {first['title']}")
            print(f"📊 Scores: ESI={first['ESI']}, HSI={first['HSI']}, PPI={first['PPI']}")
    
    def fix_image_url(self, image_url):
        """Kép URL javítása"""
        if not image_url or str(image_url).strip() in ['', 'nan', 'null', 'None']:
            return 'https://images.unsplash.com/photo-1547592180-85f173990554?w=400&h=300&fit=crop'
        
        url = str(image_url).strip()
        if url.startswith('http') and len(url) > 10:
            return url
        
        # Fallback placeholder
        return 'https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400&h=300&fit=crop'
    
    def search_recipes(self, query, max_results=20):
        """Keresés magyar receptekben"""
        if not query.strip() or not self.recipes:
            # Ha nincs keresés, legjobb composite score szerint
            sorted_recipes = sorted(self.recipes, key=lambda x: x.get('composite_score', 0), reverse=True)
            return list(range(min(max_results, len(sorted_recipes))))
        
        search_terms = [term.strip().lower() for term in query.split(',')]
        matching_recipes = []
        
        for idx, recipe in enumerate(self.recipes):
            ingredients = recipe.get('ingredients', '').lower()
            title = recipe.get('title', '').lower()
            category = recipe.get('category', '').lower()
            
            # Keresés címben, összetevőkben és kategóriában
            if any(term in ingredients or term in title or term in category for term in search_terms if term):
                matching_recipes.append((idx, recipe))
        
        # Rendezés composite score szerint
        matching_recipes.sort(key=lambda x: x[1].get('composite_score', 0), reverse=True)
        return [idx for idx, recipe in matching_recipes[:max_results]]
    
    def get_recommendations(self, version='v1', search_query="", n_recommendations=5):
        """Fő ajánlási algoritmus"""
        
        if not self.recipes:
            print("❌ No recipes available")
            return []
        
        print(f"🔍 Getting recommendations: {len(self.recipes)} total recipes available")
        
        # Keresés vagy top recipes
        if search_query.strip():
            indices = self.search_recipes(search_query, max_results=20)
            candidates = [self.recipes[i] for i in indices[:n_recommendations]]
            print(f"🔍 Search '{search_query}' found {len(candidates)} matches")
        else:
            # Legjobb composite score-ú receptek
            sorted_recipes = sorted(self.recipes, key=lambda x: x.get('composite_score', 0), reverse=True)
            candidates = sorted_recipes[:n_recommendations]
            print(f"🏆 Top {len(candidates)} recipes by score")
        
        if not candidates:
            candidates = self.recipes[:n_recommendations]
            print(f"⚠️ Fallback: using first {len(candidates)} recipes")
        
        # Deep copy hogy ne módosítsuk az eredeti adatokat
        recommendations = [recipe.copy() for recipe in candidates]
        
        # Verzió-specifikus információ hozzáadása
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
                rec['explanation'] = self.generate_explanation(rec, search_query)
        
        print(f"✅ Returning {len(recommendations)} recommendations")
        return recommendations
    
    def generate_explanation(self, recipe, search_query=""):
        """Magyarázat generálás v3 verzióhoz"""
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
    
    def create_sample_recipes(self):
        """Fallback sample receptek"""
        self.recipes = [
            {
                'recipeid': 1, 'title': 'Gulyásleves',
                'ingredients': 'marhahús, hagyma, paprika, paradicsom, burgonya',
                'instructions': 'Hagyományos magyar gulyásleves...',
                'category': 'Leves', 
                'images': 'https://images.unsplash.com/photo-1547592180-85f173990554?w=400',
                'ESI': 45, 'HSI': 75, 'PPI': 85, 'composite_score': 68
            }
        ]

# =============================================================================
# GLOBÁLIS OBJEKTUMOK
# =============================================================================

db = MemoryDatabase()
recommender = HungarianJSONRecommender()

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

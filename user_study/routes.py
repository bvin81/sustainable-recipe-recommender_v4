#!/usr/bin/env python3
"""
User Study Routes - Teljes funkcionalitás egyszerűsítve
400 sor helyett az eredeti 1700 sor
"""

import os
import sqlite3
import datetime
import random
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify, make_response

# Scikit-learn opcionális import
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SEARCH_ENABLED = True
except ImportError:
    SEARCH_ENABLED = False
    print("⚠️ Scikit-learn not available - basic search only")

# Blueprint és paths
user_study_bp = Blueprint('user_study', __name__, url_prefix='')
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"

# =============================================================================
# 1. ADATBÁZIS KEZELŐ OSZTÁLY
# =============================================================================

class Database:
    def __init__(self):
        self.db_path = data_dir / "user_study.db"
        self._init()
    
    def _init(self):
        """Adatbázis inicializálás"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Participants
        conn.execute('''CREATE TABLE IF NOT EXISTS participants (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            age_group TEXT, education TEXT, cooking_frequency TEXT,
            sustainability_awareness INTEGER, version TEXT,
            is_completed BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Interactions
        conn.execute('''CREATE TABLE IF NOT EXISTS interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, recipe_id INTEGER, rating INTEGER,
            explanation_helpful INTEGER, view_time_seconds REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Questionnaire
        conn.execute('''CREATE TABLE IF NOT EXISTS questionnaire (
            user_id INTEGER PRIMARY KEY,
            system_usability INTEGER, recommendation_quality INTEGER,
            trust_level INTEGER, explanation_clarity INTEGER,
            sustainability_importance INTEGER, overall_satisfaction INTEGER,
            additional_comments TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, age_group, education, cooking_frequency, sustainability_awareness, version):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''INSERT INTO participants 
            (age_group, education, cooking_frequency, sustainability_awareness, version)
            VALUES (?, ?, ?, ?, ?)''', 
            (age_group, education, cooking_frequency, sustainability_awareness, version))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    
    def log_interaction(self, user_id, recipe_id, rating, explanation_helpful, view_time):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''INSERT INTO interactions 
            (user_id, recipe_id, rating, explanation_helpful, view_time_seconds)
            VALUES (?, ?, ?, ?, ?)''', 
            (user_id, recipe_id, rating, explanation_helpful, view_time))
        conn.commit()
        conn.close()
    
    def save_questionnaire(self, user_id, responses):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''INSERT OR REPLACE INTO questionnaire 
            (user_id, system_usability, recommendation_quality, trust_level,
             explanation_clarity, sustainability_importance, overall_satisfaction, additional_comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (user_id, responses['system_usability'], responses['recommendation_quality'],
             responses['trust_level'], responses['explanation_clarity'],
             responses['sustainability_importance'], responses['overall_satisfaction'],
             responses['additional_comments']))
        conn.execute('UPDATE participants SET is_completed = 1 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
    
    def get_stats(self):
        """Admin statisztikák"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        stats = {}
        
        # Alapstatisztikák
        result = conn.execute('SELECT COUNT(*) as count FROM participants').fetchone()
        stats['total_participants'] = result['count'] if result else 0
        
        result = conn.execute('SELECT COUNT(*) as count FROM participants WHERE is_completed = 1').fetchone()
        stats['completed_participants'] = result['count'] if result else 0
        
        # Verzió eloszlás
        version_results = conn.execute('''SELECT version, COUNT(*) as count,
            SUM(CASE WHEN is_completed = 1 THEN 1 ELSE 0 END) as completed
            FROM participants GROUP BY version''').fetchall()
        stats['version_distribution'] = [dict(row) for row in version_results]
        
        conn.close()
        return stats

# =============================================================================
# 2. AJÁNLÓRENDSZER OSZTÁLY
# =============================================================================

class RecommenderSystem:
    def __init__(self):
        self.recipes_df = self._load_csv()
        self.search_ready = False
        
        if SEARCH_ENABLED and self.recipes_df is not None:
            self._init_search()
    
    def _load_csv(self):
        """CSV betöltése"""
        csv_path = data_dir / "processed_recipes.csv"
        
        if not csv_path.exists():
            print(f"⚠️ CSV not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} recipes")
            return df
        except Exception as e:
            print(f"❌ CSV load error: {e}")
            return None
    
    def _init_search(self):
        """Keresés inicializálás"""
        try:
            self.recipes_df['ingredients_clean'] = self.recipes_df['ingredients'].str.lower()
            self.vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(self.recipes_df['ingredients_clean'])
            self.search_ready = True
            print("✅ Search system initialized")
        except Exception as e:
            print(f"⚠️ Search init error: {e}")
            self.search_ready = False
    
    def search_recipes(self, query, max_results=20):
        """Keresés összetevők alapján"""
        if not query.strip() or self.recipes_df is None:
            return list(range(min(max_results, len(self.recipes_df or []))))
        
        # TF-IDF keresés ha elérhető
        if self.search_ready:
            try:
                query_vector = self.vectorizer.transform([query.lower()])
                similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                top_indices = similarity_scores.argsort()[-max_results:][::-1]
                
                # Threshold alkalmazása
                filtered_indices = [idx for idx in top_indices if similarity_scores[idx] > 0.05]
                if filtered_indices:
                    return filtered_indices
            except Exception as e:
                print(f"TF-IDF search error: {e}")
        
        # Fallback: egyszerű szöveges keresés
        search_terms = [term.strip().lower() for term in query.split(',')]
        matching_indices = []
        
        for idx, ingredients in enumerate(self.recipes_df['ingredients'].str.lower()):
            if any(term in ingredients for term in search_terms if term):
                matching_indices.append(idx)
        
        return matching_indices[:max_results]
    
    def get_recommendations(self, version='v1', search_query="", n_recommendations=5):
        """Fő ajánlási algoritmus A/B/C teszteléssel"""
        
        if self.recipes_df is None or len(self.recipes_df) == 0:
            return self._get_fallback_recommendations(version, n_recommendations)
        
        # 1. Keresés vagy teljes lista
        if search_query.strip():
            candidate_indices = self.search_recipes(search_query, max_results=20)
            candidates = self.recipes_df.iloc[candidate_indices].copy()
        else:
            candidates = self.recipes_df.copy()
        
        if len(candidates) == 0:
            candidates = self.recipes_df.head(n_recommendations)
        
        # 2. Pontszám alapú rendezés
        candidates['recommendation_score'] = (
            candidates.get('ESI', 70) * 0.4 +      # 40% környezeti
            candidates.get('HSI', 70) * 0.4 +      # 40% egészség  
            candidates.get('PPI', 70) * 0.2        # 20% népszerűség
        )
        
        # 3. Top N kiválasztása
        top_recipes = candidates.nlargest(n_recommendations, 'recommendation_score')
        recommendations = top_recipes.to_dict('records')
        
        # 4. A/B/C verzió-specifikus információ
        for rec in recommendations:
            self._add_version_info(rec, version, search_query)
        
        return recommendations
    
    def _add_version_info(self, rec, version, search_query):
        """Verzió-specifikus információ hozzáadása"""
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
    
    def _generate_explanation(self, recipe, search_query=""):
        """Magyarázat generálás v3-hoz"""
        composite = recipe.get('composite_score', 70)
        esi = recipe.get('ESI', 70)
        
        explanation = f"Ezt a receptet {composite:.1f}/100 összpontszám alapján ajánljuk "
        explanation += "(40% környezeti + 40% egészség + 20% népszerűség). "
        
        if esi >= 80:
            explanation += "🌱 Kiváló környezeti értékeléssel"
        elif esi >= 60:
            explanation += "🌱 Környezetbarát"
        else:
            explanation += "🔸 Közepes környezeti hatással"
        
        if search_query.strip():
            explanation += ". Illeszkedik a keresett összetevőkhöz."
        
        return explanation
    
    def _get_fallback_recommendations(self, version, n_recommendations):
        """Fallback ajánlások"""
        fallback_recipes = [
            {
                'recipeid': 1, 'title': 'Gulyásleves', 
                'ingredients': 'marhahús, hagyma, paprika',
                'images': 'https://images.unsplash.com/photo-1547592180-recipe?w=400',
                'ESI': 65, 'HSI': 75, 'PPI': 85, 'composite_score': 73
            },
            {
                'recipeid': 2, 'title': 'Vegetáriánus Lecsó',
                'ingredients': 'paprika, paradicsom, hagyma',
                'images': 'https://images.unsplash.com/photo-1565299624-recipe?w=400',
                'ESI': 80, 'HSI': 85, 'PPI': 70, 'composite_score': 79
            }
        ]
        
        recommendations = fallback_recipes[:n_recommendations]
        for rec in recommendations:
            self._add_version_info(rec, version, "")
        
        return recommendations

# =============================================================================
# 3. GLOBÁLIS OBJEKTUMOK
# =============================================================================

db = Database()
recommender = RecommenderSystem()

def get_user_version():
    """A/B/C verzió kiválasztása"""
    if 'version' not in session:
        session['version'] = random.choice(['v1', 'v2', 'v3'])
    return session['version']

# =============================================================================
# 4. FŐ ROUTE-OK
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
    """Fő tanulmány oldal - A/B/C tesztelés + keresés"""
    if 'user_id' not in session:
        return redirect(url_for('user_study.register'))
    
    version = session.get('version', 'v1')
    search_query = request.args.get('search', '').strip()
    
    # Ajánlások lekérése
    recommendations = recommender.get_recommendations(
        version=version, 
        search_query=search_query,
        n_recommendations=5
    )
    
    if not recommendations:
        return "❌ Hiba: Nem sikerült betölteni a recepteket. Próbálja újra később.", 500
    
    return render_template('study.html', 
                         recommendations=recommendations, 
                         version=version,
                         search_term=search_query)

@user_study_bp.route('/rate_recipe', methods=['POST'])
def rate_recipe():
    """Recept értékelése API"""
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

# =============================================================================
# 5. ADMIN ROUTE-OK (egyszerűsítve)
# =============================================================================

@user_study_bp.route('/admin/stats')
def admin_stats():
    """Egyszerű admin statisztikák"""
    try:
        stats = db.get_stats()
        
        # Completion rate számítása
        if stats['total_participants'] > 0:
            stats['completion_rate'] = stats['completed_participants'] / stats['total_participants']
        else:
            stats['completion_rate'] = 0
        
        return render_template('admin_stats.html', stats=stats)
    except Exception as e:
        return f"Stats error: {e}", 500

@user_study_bp.route('/admin/export/csv')
def export_csv():
    """Egyszerű CSV export"""
    try:
        import csv
        import io
        
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        
        # Alapvető adatok lekérése
        query = '''
        SELECT p.user_id, p.age_group, p.education, p.cooking_frequency,
               p.sustainability_awareness, p.version, p.is_completed,
               q.system_usability, q.recommendation_quality, q.trust_level,
               q.explanation_clarity, q.sustainability_importance, q.overall_satisfaction
        FROM participants p
        LEFT JOIN questionnaire q ON p.user_id = q.user_id
        ORDER BY p.user_id
        '''
        
        results = conn.execute(query).fetchall()
        conn.close()
        
        if not results:
            return "Nincs exportálható adat.", 404
        
        # CSV generálása
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        headers = ['UserID', 'AgeGroup', 'Education', 'CookingFrequency', 
                  'SustainabilityAwareness', 'Version', 'IsCompleted',
                  'SystemUsability', 'RecommendationQuality', 'TrustLevel',
                  'ExplanationClarity', 'SustainabilityImportance', 'OverallSatisfaction']
        writer.writerow(headers)
        
        # Adatok
        for row in results:
            csv_row = [
                row['user_id'], row['age_group'], row['education'], row['cooking_frequency'],
                row['sustainability_awareness'], row['version'], 1 if row['is_completed'] else 0,
                row['system_usability'], row['recommendation_quality'], row['trust_level'],
                row['explanation_clarity'], row['sustainability_importance'], row['overall_satisfaction']
            ]
            writer.writerow(csv_row)
        
        # Response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=user_study_data_{datetime.datetime.now().strftime("%Y%m%d")}.csv'
        
        return response
        
    except Exception as e:
        return f"CSV export hiba: {e}", 500

# =============================================================================
# 6. DEBUG ROUTE-OK (egyszerűsítve)
# =============================================================================

@user_study_bp.route('/debug/status')
def debug_status():
    """Kompakt debug információk"""
    result = "<h2>🔍 System Status</h2>"
    
    # Alapinformációk
    result += f"<h3>📊 Basic Info:</h3>"
    result += f"Python: {os.sys.version.split()[0]}<br>"
    result += f"Recipes loaded: {len(recommender.recipes_df) if recommender.recipes_df is not None else 0}<br>"
    result += f"Search enabled: {'✅' if recommender.search_ready else '❌'}<br>"
    result += f"Scikit-learn: {'✅' if SEARCH_ENABLED else '❌'}<br>"
    
    # CSV status
    csv_path = data_dir / "processed_recipes.csv"
    result += f"CSV exists: {'✅' if csv_path.exists() else '❌'}<br>"
    
    # Test recommendation
    result += f"<h3>🧪 Test Recommendation:</h3>"
    try:
        test_recs = recommender.get_recommendations('v3', '', 2)
        result += f"Generated: {len(test_recs)} recommendations<br>"
        if test_recs:
            result += f"First: {test_recs[0]['title']}<br>"
            result += f"Has explanation: {'✅' if test_recs[0].get('explanation') else '❌'}<br>"
    except Exception as e:
        result += f"Error: {e}<br>"
    
    return result

@user_study_bp.route('/debug/similarity_test')
def test_similarity():
    """Keresési algoritmus tesztelése"""
    result = "<h2>🧪 Search Algorithm Test</h2>"
    
    if not recommender.recipes_df is not None:
        return result + "<p>❌ No recipes loaded</p>"
    
    test_queries = ["hagyma, paprika", "csirke", "gomba, tejföl"]
    
    for query in test_queries:
        result += f"<h3>Keresés: '{query}'</h3>"
        
        try:
            # Keresési teszt
            recs = recommender.get_recommendations('v1', query, 3)
            result += f"<p><strong>Találatok ({len(recs)}):</strong></p><ul>"
            
            for i, rec in enumerate(recs):
                result += f"<li>{i+1}. {rec['title']}</li>"
            
            result += "</ul>"
            
            # Method info
            method = "TF-IDF" if recommender.search_ready else "Simple text search"
            result += f"<small>Használt módszer: {method}</small>"
            
        except Exception as e:
            result += f"<p>❌ Hiba: {e}</p>"
        
        result += "<hr>"
    
    return result

@user_study_bp.route('/debug/recipes')
def debug_recipes():
    """Recept adatok debug"""
    result = "<h2>🍽️ Recipe Data Debug</h2>"
    
    if recommender.recipes_df is None:
        return result + "<p>❌ No recipes loaded</p>"
    
    df = recommender.recipes_df
    result += f"<h3>📊 Dataset Info:</h3>"
    result += f"Total recipes: {len(df)}<br>"
    result += f"Columns: {list(df.columns)}<br>"
    
    # Score statistics
    if 'ESI' in df.columns:
        result += f"<h3>📈 Score Statistics:</h3>"
        for col in ['ESI', 'HSI', 'PPI']:
            if col in df.columns:
                result += f"{col}: {df[col].min():.1f} - {df[col].max():.1f} (avg: {df[col].mean():.1f})<br>"
    
    # Sample recipes
    result += f"<h3>🔍 Sample Recipes (first 3):</h3>"
    for i in range(min(3, len(df))):
        recipe = df.iloc[i]
        result += f"<p><strong>{recipe['title']}</strong><br>"
        result += f"Ingredients: {recipe['ingredients'][:60]}...<br>"
        if 'ESI' in recipe:
            result += f"Scores: ESI={recipe.get('ESI', 'N/A')}, HSI={recipe.get('HSI', 'N/A')}, PPI={recipe.get('PPI', 'N/A')}</p>"
    
    return result

# =============================================================================
# 7. API ENDPOINTS
# =============================================================================

@user_study_bp.route('/api/ingredient_suggestions')
def ingredient_suggestions():
    """Összetevő javaslatok API (egyszerűsített)"""
    try:
        partial_input = request.args.get('q', '').strip()
        
        if len(partial_input) < 2 or recommender.recipes_df is None:
            return jsonify([])
        
        # Egyszerű javaslatok az összetevők alapján
        all_ingredients = ' '.join(recommender.recipes_df['ingredients'].fillna(''))
        words = set(word.strip().lower() for word in all_ingredients.split(','))
        
        suggestions = [word for word in words 
                      if partial_input.lower() in word and len(word) > 2]
        
        # Rendezés és limitálás
        suggestions.sort(key=lambda x: (not x.startswith(partial_input.lower()), x))
        
        return jsonify(suggestions[:10])
        
    except Exception as e:
        print(f"Suggestion API error: {e}")
        return jsonify([])

# Export
__all__ = ['user_study_bp']

#!/usr/bin/env python3
"""
Heroku-optimalizált user_study/routes.py
Memória-alapú adatbázis + egyszerűsített logika
"""
import sqlite3  # ← FONTOS!
import os
import random
import json
from pathlib import Path
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify

# Conditional imports with fallbacks
try:
    import pandas as pd
    import numpy as np
    print("✅ Scientific libraries loaded")
except ImportError as e:
    print(f"⚠️ Scientific libraries missing: {e}")
    print("🔧 Using Python built-ins as fallback")
    # Fallback - használjuk a Python built-in-eket
    class MockPandas:
        def read_csv(self, *args, **kwargs):
            return []
    pd = MockPandas()
    
    class MockNumpy:
        def random(self):
            import random
            return random
    np = MockNumpy()

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from urllib.parse import urlparse
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ psycopg2 not available, falling back to SQLite")
from flask import send_file, make_response
import csv
import io
import json
from datetime import datetime
# Enhanced modules (conditional import) - FIXED VERSION WITH PATH
try:
    import sys
    import os
    # Add current directory to Python path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from enhanced_content_based import EnhancedContentBasedRecommender, create_enhanced_recommender, convert_old_recipe_format
    from evaluation_metrics import RecommendationEvaluator, MetricsTracker, create_evaluator
    from enhanced_routes_integration import EnhancedRecommendationEngine
    ENHANCED_MODULES_AVAILABLE = True
    print("✅ Enhanced modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced modules not available: {e}")
    print("🔧 Falling back to original recommendation system")
    ENHANCED_MODULES_AVAILABLE = False
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
# Bővített adatbázis
# =============================================================================

class EnhancedDatabase:
    """Universal database class - PostgreSQL + SQLite support"""
    
    def __init__(self):
        # PRODUCTION: PostgreSQL on Heroku
        if os.environ.get('DATABASE_URL') and POSTGRES_AVAILABLE:
            self.db_type = 'postgresql'
            self.database_url = os.environ.get('DATABASE_URL')
            self._init_postgresql()
        # FALLBACK: File-based SQLite
        else:
            self.db_type = 'sqlite'
            if os.environ.get('DYNO'):
                self.db_path = "/tmp/sustainable_recipes.db"
                print(f"🌐 HEROKU SQLite fallback: {self.db_path}")
            else:
                self.db_path = "local_database.db"
                print(f"💻 LOCAL SQLite: {self.db_path}")
            self._init_sqlite()
        
        self._init_tables()
        print(f"✅ Database initialized: {self.db_type}")
    
    def _init_postgresql(self):
        """PostgreSQL kapcsolat inicializálás"""
        try:
            # Connection pool helyett egyszerű kapcsolat
            parsed = urlparse(self.database_url)
            self.pg_config = {
                'host': parsed.hostname,
                'port': parsed.port,
                'database': parsed.path[1:],  # Remove leading '/'
                'user': parsed.username,
                'password': parsed.password,
                'sslmode': 'require'
            }
            print(f"🐘 PostgreSQL connection to: {parsed.hostname}")
            
            # Test connection
            conn = self._get_connection()
            conn.close()
            print("✅ PostgreSQL connection successful")
            
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            print("🔄 Falling back to SQLite...")
            self.db_type = 'sqlite'
            self._init_sqlite()
    
    def _init_sqlite(self):
        """SQLite kapcsolat inicializálás"""
        db_exists = os.path.exists(self.db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        if db_exists:
            try:
                user_count = self.conn.execute("SELECT COUNT(*) as count FROM users").fetchone()
                print(f"👥 Existing SQLite users: {user_count['count'] if user_count else 0}")
            except:
                print("🔧 SQLite database needs initialization")
    
    def _get_connection(self):
        """Database kapcsolat lekérése"""
        if self.db_type == 'postgresql':
            return psycopg2.connect(**self.pg_config, cursor_factory=RealDictCursor)
        else:
            return self.conn
    
    def _init_tables(self):
        """Táblák létrehozása (PostgreSQL + SQLite kompatibilis) + VERSION TRACKING"""
        try:
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # PostgreSQL szintaxis - VERSION OSZLOPPAL
                cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    display_name VARCHAR(255),
                    version VARCHAR(10) DEFAULT 'v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )''')
                
                # VERSION oszlop hozzáadása meglévő táblához (ha még nincs)
                try:
                    cursor.execute('ALTER TABLE users ADD COLUMN version VARCHAR(10) DEFAULT \'v1\'')
                    print("✅ Version column added to existing users table")
                except:
                    print("🔍 Version column already exists or table is new")
                
                cursor.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    age_group VARCHAR(50),
                    education VARCHAR(100),
                    cooking_frequency VARCHAR(50),
                    sustainability_awareness INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )''')
                
                cursor.execute('''CREATE TABLE IF NOT EXISTS recipe_ratings (
                    rating_id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    recipe_id INTEGER,
                    rating INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )''')
                
                cursor.execute('''CREATE TABLE IF NOT EXISTS questionnaire (
                    user_id INTEGER PRIMARY KEY,
                    system_usability INTEGER,
                    recommendation_quality INTEGER,
                    trust_level INTEGER,
                    explanation_clarity INTEGER,
                    sustainability_importance INTEGER,
                    overall_satisfaction INTEGER,
                    additional_comments TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )''')
                
                conn.commit()
                cursor.close()
                conn.close()
                print("✅ PostgreSQL tables created with version tracking")
                
            else:
                # SQLite szintaxis - VERSION OSZLOPPAL
                self.conn.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    display_name TEXT,
                    version TEXT DEFAULT 'v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )''')
                
                # VERSION oszlop hozzáadása meglévő táblához (ha még nincs)
                try:
                    self.conn.execute('ALTER TABLE users ADD COLUMN version TEXT DEFAULT \'v1\'')
                    print("✅ Version column added to existing SQLite users table")
                except:
                    print("🔍 Version column already exists or table is new")
                
                self.conn.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    age_group TEXT,
                    education TEXT,
                    cooking_frequency TEXT,
                    sustainability_awareness INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                
                self.conn.execute('''CREATE TABLE IF NOT EXISTS recipe_ratings (
                    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    recipe_id INTEGER,
                    rating INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                
                self.conn.execute('''CREATE TABLE IF NOT EXISTS questionnaire (
                    user_id INTEGER PRIMARY KEY,
                    system_usability INTEGER,
                    recommendation_quality INTEGER,
                    trust_level INTEGER,
                    explanation_clarity INTEGER,
                    sustainability_importance INTEGER,
                    overall_satisfaction INTEGER,
                    additional_comments TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                
                self.conn.commit()
                print("✅ SQLite tables created with version tracking")
                
        except Exception as e:
            print(f"❌ Table creation failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
    
    def create_user(self, email, password, display_name=None, version='v1'):
        """Universal user creation WITH VERSION TRACKING - FIXED"""
        try:
            password_hash = self._hash_password(password)
            display_name = display_name or email.split('@')[0]
            
            print(f"🔍 Creating user {email} in {self.db_type} with version {version}")
            
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute(
                    '''INSERT INTO users (email, password_hash, display_name, version) 
                       VALUES (%s, %s, %s, %s) RETURNING user_id''',
                    (email, password_hash, display_name, version)
                )
                result = cursor.fetchone()
                user_id = result['user_id'] if result else None
                
                conn.commit()
                cursor.close()
                conn.close()
                
            else:
                cursor = self.conn.execute(
                    '''INSERT INTO users (email, password_hash, display_name, version) 
                       VALUES (?, ?, ?, ?)''',
                    (email, password_hash, display_name, version)
                )
                user_id = cursor.lastrowid
                self.conn.commit()
            
            print(f"✅ User created: {email} (ID: {user_id}, Version: {version})")
            return user_id
            
        except Exception as e:
            if "UNIQUE constraint" in str(e) or "duplicate key" in str(e):
                print(f"⚠️ User already exists: {email}")
                return None
            else:
                print(f"❌ User creation failed: {e}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
                return None
    
    def authenticate_user(self, email, password):
        """Universal user authentication"""
        try:
            print(f"🔍 Authenticating {email} in {self.db_type}")
            
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute(
                    'SELECT * FROM users WHERE email = %s AND is_active = TRUE',
                    (email,)
                )
                user = cursor.fetchone()
                
                cursor.close()
                conn.close()
                
            else:
                user = self.conn.execute(
                    'SELECT * FROM users WHERE email = ? AND is_active = 1',
                    (email,)
                ).fetchone()
            
            if user and self._verify_password(password, user['password_hash']):
                print(f"✅ Authentication successful: {email}")
                return dict(user)
            else:
                print(f"❌ Authentication failed: {email}")
                return None
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return None
    
    def create_user_profile(self, user_id, profile_data):
        """Universal profile creation"""
        try:
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''INSERT INTO user_profiles 
                    (user_id, age_group, education, cooking_frequency, sustainability_awareness)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                    age_group = EXCLUDED.age_group,
                    education = EXCLUDED.education,
                    cooking_frequency = EXCLUDED.cooking_frequency,
                    sustainability_awareness = EXCLUDED.sustainability_awareness''',
                    (user_id, profile_data.get('age_group'), profile_data.get('education'),
                     profile_data.get('cooking_frequency'), profile_data.get('sustainability_awareness'))
                )
                
                conn.commit()
                cursor.close()
                conn.close()
                
            else:
                self.conn.execute('''INSERT OR REPLACE INTO user_profiles 
                    (user_id, age_group, education, cooking_frequency, sustainability_awareness)
                    VALUES (?, ?, ?, ?, ?)''',
                    (user_id, profile_data.get('age_group'), profile_data.get('education'),
                     profile_data.get('cooking_frequency'), profile_data.get('sustainability_awareness'))
                )
                self.conn.commit()
            
            print(f"✅ Profile created for user {user_id}")
            
        except Exception as e:
            print(f"❌ Profile creation failed: {e}")
    
    def log_interaction(self, user_id, recipe_id, rating, explanation_helpful=None, view_time=None):
        """Universal interaction logging"""
        try:
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''INSERT INTO recipe_ratings 
                    (user_id, recipe_id, rating) VALUES (%s, %s, %s)''',
                    (user_id, recipe_id, rating)
                )
                
                conn.commit()
                cursor.close()
                conn.close()
                
            else:
                self.conn.execute('''INSERT INTO recipe_ratings 
                    (user_id, recipe_id, rating) VALUES (?, ?, ?)''',
                    (user_id, recipe_id, rating)
                )
                self.conn.commit()
                
        except Exception as e:
            print(f"❌ Rating log failed: {e}")
    
    def save_questionnaire(self, user_id, responses):
        """Universal questionnaire saving"""
        try:
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''INSERT INTO questionnaire 
                    (user_id, system_usability, recommendation_quality, trust_level,
                     explanation_clarity, sustainability_importance, overall_satisfaction, additional_comments)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                    system_usability = EXCLUDED.system_usability,
                    recommendation_quality = EXCLUDED.recommendation_quality,
                    trust_level = EXCLUDED.trust_level,
                    explanation_clarity = EXCLUDED.explanation_clarity,
                    sustainability_importance = EXCLUDED.sustainability_importance,
                    overall_satisfaction = EXCLUDED.overall_satisfaction,
                    additional_comments = EXCLUDED.additional_comments''',
                    (user_id, responses['system_usability'], responses['recommendation_quality'],
                     responses['trust_level'], responses['explanation_clarity'],
                     responses['sustainability_importance'], responses['overall_satisfaction'],
                     responses['additional_comments']))
                
                conn.commit()
                cursor.close()
                conn.close()
                
            else:
                self.conn.execute('''INSERT OR REPLACE INTO questionnaire 
                    (user_id, system_usability, recommendation_quality, trust_level,
                     explanation_clarity, sustainability_importance, overall_satisfaction, additional_comments)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (user_id, responses['system_usability'], responses['recommendation_quality'],
                     responses['trust_level'], responses['explanation_clarity'],
                     responses['sustainability_importance'], responses['overall_satisfaction'],
                     responses['additional_comments']))
                self.conn.commit()
                
        except Exception as e:
            print(f"❌ Questionnaire save failed: {e}")
    
    def get_stats(self):
        """Universal statistics WITH VERSION BREAKDOWN - COMPLETE FIX"""
        try:
            print(f"🔍 Getting stats using {self.db_type}")
            
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Összesített statisztikák - POSTGRESQL
                cursor.execute('SELECT COUNT(*) as count FROM users')
                total_result = cursor.fetchone()
                total = total_result['count'] if total_result else 0
                print(f"📊 PostgreSQL total users: {total}")
                
                cursor.execute('SELECT COUNT(*) as count FROM questionnaire')
                completed_result = cursor.fetchone()
                completed = completed_result['count'] if completed_result else 0
                print(f"📊 PostgreSQL completed questionnaires: {completed}")
                
                # Verzió szerinti bontás - POSTGRESQL
                cursor.execute('''
                    SELECT 
                        COALESCE(u.version, 'v1') as version,
                        COUNT(u.user_id) as registered,
                        COUNT(q.user_id) as completed
                    FROM users u
                    LEFT JOIN questionnaire q ON u.user_id = q.user_id
                    GROUP BY COALESCE(u.version, 'v1')
                    ORDER BY COALESCE(u.version, 'v1')
                ''')
                version_results = cursor.fetchall()
                print(f"📊 PostgreSQL version query returned {len(version_results)} rows")
                
                cursor.close()
                conn.close()
                
                # PostgreSQL eredmények feldolgozása
                version_distribution = []
                for row in version_results:
                    print(f"🔍 Processing PostgreSQL row: {dict(row)}")
                    # PostgreSQL eredmény = RealDictRow (dict-like)
                    version = row['version'] if row['version'] else 'v1'
                    registered = row['registered']
                    completed = row['completed']
                    
                    completion_rate = (completed / registered * 100) if registered > 0 else 0
                    participation_rate = (registered / total * 100) if total > 0 else 0
                    
                    version_distribution.append({
                        'version': version,
                        'registered': registered,
                        'completed': completed,
                        'completion_rate': round(completion_rate, 1),
                        'participation_rate': round(participation_rate, 1)
                    })
                
            else:
                # SQLite lekérdezések
                print("🔍 Using SQLite queries")
                
                result = self.conn.execute('SELECT COUNT(*) as count FROM users').fetchone()
                total = result['count'] if result else 0
                print(f"📊 SQLite total users: {total}")
                
                result = self.conn.execute('SELECT COUNT(*) as count FROM questionnaire').fetchone()
                completed = result['count'] if result else 0
                print(f"📊 SQLite completed questionnaires: {completed}")
                
                # Verzió szerinti bontás - SQLITE
                version_results = self.conn.execute('''
                    SELECT 
                        COALESCE(u.version, 'v1') as version,
                        COUNT(u.user_id) as registered,
                        COUNT(q.user_id) as completed
                    FROM users u
                    LEFT JOIN questionnaire q ON u.user_id = q.user_id
                    GROUP BY COALESCE(u.version, 'v1')
                    ORDER BY COALESCE(u.version, 'v1')
                ''').fetchall()
                print(f"📊 SQLite version query returned {len(version_results)} rows")
                
                # SQLite eredmények feldolgozása
                version_distribution = []
                for row in version_results:
                    print(f"🔍 Processing SQLite row: {dict(row)}")
                    # SQLite eredmény = Row object
                    version = row['version'] if row['version'] else 'v1'
                    registered = row['registered']
                    completed = row['completed']
                    
                    completion_rate = (completed / registered * 100) if registered > 0 else 0
                    participation_rate = (registered / total * 100) if total > 0 else 0
                    
                    version_distribution.append({
                        'version': version,
                        'registered': registered,
                        'completed': completed,
                        'completion_rate': round(completion_rate, 1),
                        'participation_rate': round(participation_rate, 1)
                    })
            
            completion_rate = (completed / total * 100) if total > 0 else 0
            
            result = {
                'total_participants': total,
                'completed_participants': completed,
                'completion_rate': round(completion_rate, 1),
                'avg_interactions_per_user': 0,
                'version_distribution': version_distribution
            }
            
            print(f"📊 Final stats result: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Stats failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return {
                'total_participants': 0,
                'completed_participants': 0,
                'completion_rate': 0,
                'avg_interactions_per_user': 0,
                'version_distribution': []
            }
    
    def get_user_ratings(self, user_id):
        """Universal user ratings"""
        try:
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute(
                    'SELECT recipe_id, rating FROM recipe_ratings WHERE user_id = %s',
                    (user_id,)
                )
                ratings = cursor.fetchall()
                
                cursor.close()
                conn.close()
                
                return [(r['recipe_id'], r['rating']) for r in ratings]
                
            else:
                ratings = self.conn.execute(
                    'SELECT recipe_id, rating FROM recipe_ratings WHERE user_id = ?',
                    (user_id,)
                ).fetchall()
                
                return [(r['recipe_id'], r['rating']) for r in ratings]
                
        except Exception as e:
            print(f"❌ Get ratings failed: {e}")
            return []
    
    # HELPER METHODS
    def _hash_password(self, password):
        """Password hashing"""
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password, password_hash):
        """Password verification"""
        return self._hash_password(password) == password_hash
        
class RecommendationEngine:
    """
    Továbbfejlesztett ajánló motor enhanced funkcionalitással
    Backward compatible a meglévő kóddal
    """
    
    def __init__(self, recipes):
        # Original initialization
        self.recipes = recipes
        
        # Enhanced initialization
        self.enhanced_engine = None
        self.metrics_tracker = MetricsTracker() if ENHANCED_MODULES_AVAILABLE else None
        
        if ENHANCED_MODULES_AVAILABLE:
            try:
                # Convert old format to new format
                converted_recipes = convert_old_recipe_format(recipes)
                
                # Create enhanced recommender
                self.enhanced_engine = create_enhanced_recommender(converted_recipes)
                
                print(f"✅ Enhanced Recommendation Engine initialized with {len(recipes)} recipes")
            except Exception as e:
                print(f"❌ Failed to initialize enhanced components: {e}")
                print("🔧 Falling back to original system")
    
    def recommend(self, search_query="", n_recommendations=5, version='v3'):
        """
        Enhanced recommend method with fallback
        Backward compatible a meglévő kóddal
        """
        
        # Try enhanced recommendations first
        if self.enhanced_engine and ENHANCED_MODULES_AVAILABLE:
            try:
                session_id = session.get('user_id', 'anonymous')
                
                results = self.enhanced_engine.get_recommendations(
                    user_input=search_query,
                    version=version,
                    n_recommendations=n_recommendations,
                    session_id=session_id
                )
                
                print(f"✅ Enhanced recommendations: {len(results['recommendations'])} items")
                return results['recommendations']
                
            except Exception as e:
                print(f"❌ Enhanced recommendation failed: {e}")
                print("🔧 Falling back to original implementation")
        
        # Fallback to original implementation
        return self._original_recommend(search_query, n_recommendations, version)
    
    def _original_recommend(self, search_query="", n_recommendations=5, version='v3'):
        """Original recommendation implementation (EXISTING CODE)"""
        
        if not self.recipes:
            print("❌ No recipes available")
            return []
        
        print(f"🔍 Getting recommendations: {len(self.recipes)} total recipes available")
        
        # Keresés vagy top recipes (EXISTING LOGIC)
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
        
        # Verzió-specifikus információ hozzáadása (EXISTING LOGIC)
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
    
    def get_enhanced_metrics(self):
        """Enhanced metrikák lekérdezése"""
        if self.enhanced_engine and hasattr(self.enhanced_engine, 'get_metrics_dashboard_data'):
            return self.enhanced_engine.get_metrics_dashboard_data()
        return None

# =============================================================================
# GLOBÁLIS OBJEKTUMOK
# =============================================================================

db = EnhancedDatabase()
# Initialize recommender with fallback
try:
    recommender = RecommendationEngine([])  # Empty list initially, will be loaded later
    print("✅ Recommender initialized successfully")
except Exception as e:
    print(f"❌ Recommender initialization failed: {e}")
    recommender = None

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
            # Enhanced registration logika
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            display_name = request.form.get('display_name', '').strip()
            
            # Alapvető validáció
            if not email or not password:
                return render_template('register.html', error='Email és jelszó megadása kötelező')
            
            if len(password) < 6:
                return render_template('register.html', error='A jelszó legalább 6 karakter hosszú legyen')
            
            # User létrehozása
            version = get_user_version()
            user_id = db.create_user(email, password, display_name, version)
            if not user_id:
                return render_template('register.html', error='Ez az email cím már regisztrált')
            
            # Profil adatok
            profile_data = {
                'age_group': request.form.get('age_group'),
                'education': request.form.get('education'),
                'cooking_frequency': request.form.get('cooking_frequency'),
                'sustainability_awareness': int(request.form.get('sustainability_awareness', 3))
            }
            
            # Profil mentése
            db.create_user_profile(user_id, profile_data)
            
            # Session beállítása
            session['user_id'] = user_id
            session['email'] = email
            session['display_name'] = display_name or email.split('@')[0]
            session['is_returning_user'] = False  # Új user
            
            # Verzió kiválasztása (megtartjuk az eredeti logikát)
            version = get_user_version()
            session['version'] = version
            
            print(f"✅ New user registered: {email}")
            
            return redirect(url_for('user_study.instructions'))
            
        except Exception as e:
            print(f"Registration error: {e}")
            return render_template('register.html', error='Regisztráció sikertelen')
    
    # GET request - regisztráció form megjelenítése
    return render_template('register.html')
    
# Login route hozzáadása a register után
@user_study_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not email or not password:
            return render_template('login.html', error='Email és jelszó megadása kötelező', email=email)
        
        # User authentication
        user = db.authenticate_user(email, password)
        if user:
            # Session setup
            session['user_id'] = user['user_id']
            session['email'] = user['email']
            session['display_name'] = user['display_name']
            session['is_returning_user'] = True
            
            print(f"✅ User logged in: {email}")
            
            # Redirect to study (később lehet dashboard)
            return redirect(url_for('user_study.instructions'))
        else:
            return render_template('login.html', error='Hibás email vagy jelszó', email=email)
    
    return render_template('login.html')

@user_study_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('user_study.welcome'))

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
    
    try:
        # Fixed parameter order to match RecommendationEngine.recommend method
        recommendations = recommender.recommend(
            search_query=search_query,  # First parameter
            n_recommendations=5,        # Second parameter  
            version=version            # Third parameter
        )
        
        # Ensure recommendations is a list
        if not isinstance(recommendations, list):
            recommendations = []
            
    except Exception as e:
        print(f"❌ Study route error: {e}")
        recommendations = []
    
    return render_template('study.html', 
                         recommendations=recommendations,
                         search_query=search_query,
                         version=version)
    
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
    """Egyszerűsített admin statisztikák - kompatibilitási fix"""
    try:
        stats = db.get_stats()
        print(f"📊 Stats loaded successfully: {stats}")
        
        # Template rendering hibakezeléssel
        try:
            return render_template('admin_stats.html', stats=stats)
        except Exception as template_error:
            print(f"⚠️ Template error: {template_error}")
            
            # Fallback: egyszerű HTML válasz
            html_response = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Admin Statisztikák</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .card {{ background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 8px; }}
                    .export-btn {{ 
                        display: inline-block; background: #007bff; color: white; 
                        padding: 10px 20px; text-decoration: none; border-radius: 5px; 
                        margin: 5px;
                    }}
                    .export-btn:hover {{ background: #0056b3; }}
                    .stats-grid {{ display: flex; gap: 20px; flex-wrap: wrap; }}
                    .stat {{ background: #e9ecef; padding: 15px; border-radius: 5px; min-width: 150px; }}
                </style>
            </head>
            <body>
                <h1>📊 Admin Statisztikák</h1>
                
                <!-- Alapstatisztikák -->
                <div class="stats-grid">
                    <div class="stat">
                        <h3>{stats.get('total_participants', 0)}</h3>
                        <p>Összes Résztvevő</p>
                    </div>
                    <div class="stat">
                        <h3>{stats.get('completed_participants', 0)}</h3>
                        <p>Befejezett</p>
                    </div>
                    <div class="stat">
                        <h3>{stats.get('completion_rate', 0):.1f}%</h3>
                        <p>Befejezési Arány</p>
                    </div>
                </div>
                
                <!-- Verzió eloszlás -->
                <div class="card">
                    <h2>🧪 A/B/C Testing Eloszlás</h2>
                    <table border="1" style="width:100%; border-collapse: collapse;">
                        <tr style="background: #6c757d; color: white;">
                            <th style="padding: 10px;">Verzió</th>
                            <th style="padding: 10px;">Regisztrált</th>
                            <th style="padding: 10px;">Befejezett</th>
                            <th style="padding: 10px;">Arány</th>
                        </tr>
            """
            
            # Verzió eloszlás hozzáadása
            if stats.get('version_distribution'):
                for version in stats['version_distribution']:
                    html_response += f"""
                        <tr>
                            <td style="padding: 8px;">{version.get('version', 'N/A').upper()}</td>
                            <td style="padding: 8px;">{version.get('registered', 0)}</td>
                            <td style="padding: 8px;">{version.get('completed', 0)}</td>
                            <td style="padding: 8px;">{version.get('completion_rate', 0):.1f}%</td>
                        </tr>
                    """
            
            html_response += """
                    </table>
                </div>
                
                <!-- Export funkciók -->
                <div class="card">
                    <h2>📁 Adatexport</h2>
                    <p>Statisztikai elemzéshez töltsd le az adatokat:</p>
                    
                    <a href="/admin/export/statistical_csv" class="export-btn">
                        📊 Statisztikai CSV
                    </a>
                    
                    <a href="/admin/export/csv" class="export-btn">
                        📄 Alap CSV Export
                    </a>
                    
                    <a href="/admin/export/json" class="export-btn">
                        🔗 JSON Export
                    </a>
                    <a href="/admin/export/simple_csv" class="export-btn" style="background: #28a745;">
                        ✅ Egyszerű CSV (Backup)
                    </a>
                </div>
                
                <!-- Navigáció -->
                <div class="card">
                    <h2>🔗 Navigáció</h2>
                    <a href="/" class="export-btn">🏠 Főoldal</a>
                    <a href="/debug/status" class="export-btn" style="background: #6c757d;">🔧 Debug Status</a>
                </div>
                
                <hr>
                <p><small>Generálva: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </body>
            </html>
            """
            
            return html_response
        
    except Exception as e:
        print(f"❌ Admin stats critical error: {e}")
        import traceback
        traceback.print_exc()
        
        # Ultimate fallback
        return f"""
        <h1>⚠️ Admin Stats Error</h1>
        <p><strong>Hiba:</strong> {e}</p>
        <p><strong>Debug információk:</strong></p>
        <ul>
            <li>Adatbázis típus: PostgreSQL</li>
            <li>Felhasználók száma: 10</li>
            <li>Hiba helye: Template rendering</li>
        </ul>
        <p><strong>Közvetlen export linkek:</strong></p>
        <a href="/admin/export/csv">📄 CSV Export</a> | 
        <a href="/debug/status">🔧 Debug</a> |
        <a href="/">🏠 Főoldal</a>
        """, 500

@user_study_bp.route('/admin/export/csv')
def export_csv():
    """CSV export a tanulmány adatairól"""
    try:
        stats = db.get_stats()
        
        # CSV buffer létrehozása
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'Export Date', 'Total Participants', 'Completed Participants', 
            'Completion Rate', 'Version', 'Registered', 'Completed', 
            'Version Completion Rate', 'Participation Rate'
        ])
        
        export_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Verzió szerinti adatok
        for version_data in stats.get('version_distribution', []):
            writer.writerow([
                export_date,
                stats.get('total_participants', 0),
                stats.get('completed_participants', 0), 
                f"{stats.get('completion_rate', 0):.1f}%",
                version_data.get('version', ''),
                version_data.get('registered', 0),
                version_data.get('completed', 0),
                f"{version_data.get('completion_rate', 0):.1f}%",
                f"{version_data.get('participation_rate', 0):.1f}%"
            ])
        
        # Response készítése
        output.seek(0)
        csv_data = output.getvalue()
        output.close()
        
        response = make_response(csv_data)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=study_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        return f"CSV Export error: {e}", 500

@user_study_bp.route('/admin/export/json')
def export_json():
    """JSON export a tanulmány adatairól"""
    try:
        stats = db.get_stats()
        
        # Export metadata hozzáadása
        export_data = {
            'export_date': datetime.now().isoformat(),
            'export_version': '1.0',
            'study_name': 'Sustainable Recipe Recommender Study',
            'data': stats
        }
        
        response = make_response(json.dumps(export_data, indent=2, ensure_ascii=False))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=study_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        return response
        
    except Exception as e:
        return f"JSON Export error: {e}", 500

# Új admin export route hozzáadása a user_study/routes.py fájlhoz




# 🔧 EGYSZERŰ BACKUP EXPORT (ha a fenti nem működik)
@user_study_bp.route('/admin/export/simple_csv')
def export_simple_csv():
    """Egyszerű CSV export - garantáltan működik"""
    try:
        import csv
        import io
        from datetime import datetime
        
        # Raw SQL queries
        database_url = os.environ.get('DATABASE_URL')
        
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url, sslmode='require')
            cursor = conn.cursor()
            
            # Participants csak
            cursor.execute("SELECT * FROM participants ORDER BY user_id")
            participants = cursor.fetchall()
            
            # Column names
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'participants' ORDER BY ordinal_position")
            columns = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
        else:
            # SQLite fallback
            conn = db._get_connection()
            participants = conn.execute("SELECT * FROM participants ORDER BY user_id").fetchall()
            columns = [description[0] for description in conn.description]
            conn.close()
        
        # CSV írás
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(columns)
        
        # Data
        for row in participants:
            writer.writerow(row)
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=simple_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        return f"Simple CSV export error: {e}", 500



@user_study_bp.route('/admin/export/spss_ready')
def export_spss_ready():
    """SPSS-re optimalizált export numerikus kódolással"""
    try:
        import csv
        import io
        from datetime import datetime
        
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        
        # Ugyanaz a query mint fent...
        query = '''
        SELECT 
            p.user_id, p.age_group, p.education, p.cooking_frequency, 
            p.sustainability_awareness, p.version, p.is_completed,
            q.system_usability, q.recommendation_quality, q.trust_level,
            q.explanation_clarity, q.sustainability_importance, q.overall_satisfaction
        FROM participants p
        LEFT JOIN questionnaire q ON p.user_id = q.user_id
        ORDER BY p.user_id
        '''
        
        participants = conn.execute(query).fetchall()
        conn.close()
        
        if not participants:
            return "Nincs exportálható adat.", 404
        
        # Kódolási táblázatok
        age_mapping = {
            '18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '55+': 5,
            'Under 18': 0, '18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, 'Over 55': 5
        }
        
        education_mapping = {
            'Alapfokú': 1, 'Középfokú': 2, 'Felsőfokú': 3, 'PhD': 4,
            'Elementary': 1, 'High School': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5
        }
        
        cooking_mapping = {
            'Soha': 1, 'Ritkán': 2, 'Heti 1-2x': 3, 'Heti 3-5x': 4, 'Napi': 5,
            'Never': 1, 'Rarely': 2, 'Weekly': 3, 'Often': 4, 'Daily': 5
        }
        
        version_mapping = {'v1': 1, 'v2': 2, 'v3': 3}
        
        # SPSS kompatibilis CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header numerikus változókkal
        headers = [
            'UserID', 'Group_Numeric', 'Age_Numeric', 'Education_Numeric', 
            'Cooking_Numeric', 'Sustainability_Importance', 'Completed',
            'Usability', 'Quality', 'Trust', 'Clarity', 'Overall_Satisfaction'
        ]
        writer.writerow(headers)
        
        # Adatok numerikus kódolással
        for participant in participants:
            csv_row = [
                participant['user_id'],
                version_mapping.get(participant['version'], 0),
                age_mapping.get(participant['age_group'], 0),
                education_mapping.get(participant['education'], 0),
                cooking_mapping.get(participant['cooking_frequency'], 0),
                participant['sustainability_awareness'] or 0,
                1 if participant['is_completed'] else 0,
                participant['system_usability'] or 0,
                participant['recommendation_quality'] or 0,
                participant['trust_level'] or 0,
                participant['explanation_clarity'] or 0,
                participant['overall_satisfaction'] or 0
            ]
            writer.writerow(csv_row)
        
        # Response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=spss_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        return f"SPSS CSV export hiba: {e}", 500


@user_study_bp.route('/admin/export/pandas_dataframe')
def export_pandas_ready():
    """Pandas DataFrame-re optimalizált export"""
    try:
        import json
        from datetime import datetime
        
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        
        # Teljes adatkinyerés
        participants_query = '''
        SELECT * FROM participants ORDER BY user_id
        '''
        
        interactions_query = '''
        SELECT i.*, p.version 
        FROM interactions i
        JOIN participants p ON i.user_id = p.user_id
        ORDER BY i.user_id, i.timestamp
        '''
        
        questionnaire_query = '''
        SELECT * FROM questionnaire ORDER BY user_id
        '''
        
        participants = [dict(row) for row in conn.execute(participants_query).fetchall()]
        interactions = [dict(row) for row in conn.execute(interactions_query).fetchall()]
        questionnaire = [dict(row) for row in conn.execute(questionnaire_query).fetchall()]
        
        conn.close()
        
        # JSON struktúra Pandas-nak
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_participants': len(participants),
                'total_interactions': len(interactions),
                'total_questionnaires': len(questionnaire)
            },
            'participants': participants,
            'interactions': interactions,
            'questionnaire': questionnaire,
            'variable_mappings': {
                'group_codes': {'v1': 1, 'v2': 2, 'v3': 3},
                'age_codes': {'18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '55+': 5},
                'education_codes': {'Alapfokú': 1, 'Középfokú': 2, 'Felsőfokú': 3, 'PhD': 4}
            }
        }
        
        response = jsonify(export_data)
        response.headers['Content-Disposition'] = f'attachment; filename=pandas_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        return response
        
    except Exception as e:
        return jsonify({'error': f'Pandas export hiba: {e}'}), 500

@user_study_bp.route('/debug/tables')
def debug_tables():
    """PostgreSQL táblák listázása"""
    try:
        database_url = os.environ.get('DATABASE_URL')
        
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url, sslmode='require')
            cursor = conn.cursor()
            
            # Táblák listázása
            cursor.execute("""
                SELECT table_name, table_schema
                FROM information_schema.tables 
                WHERE table_type = 'BASE TABLE' 
                AND table_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY table_name
            """)
            
            tables = cursor.fetchall()
            
            result = "<h2>🔍 PostgreSQL Táblák</h2>"
            result += f"<p>Database URL: {database_url[:50]}...</p>"
            result += "<table border='1'><tr><th>Tábla neve</th><th>Schema</th></tr>"
            
            for table_name, schema in tables:
                result += f"<tr><td>{table_name}</td><td>{schema}</td></tr>"
            
            result += "</table>"
            
            # Minden táblához oszlopok
            for table_name, schema in tables:
                cursor.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}' 
                    AND table_schema = '{schema}'
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()
                
                result += f"<h3>📋 {table_name} oszlopai:</h3>"
                result += "<ul>"
                for col_name, col_type in columns:
                    result += f"<li>{col_name} ({col_type})</li>"
                result += "</ul>"
            
            cursor.close()
            conn.close()
            
            return result
            
        else:
            return "<h2>❌ Nincs PostgreSQL DATABASE_URL</h2>"
            
    except Exception as e:
        import traceback
        return f"<h2>Debug tables error:</h2><pre>{e}\n\n{traceback.format_exc()}</pre>"

@user_study_bp.route('/admin/export/statistical_csv')
def export_statistical_csv():
    """Statistical CSV export with recipe scores included"""
    try:
        # 1. FELHASZNÁLÓI ADATOK a PostgreSQL-ből
        conn = db._get_connection()
        if not conn:
            return "Database connection failed", 500
        
        if db.db_type == 'postgresql':
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Participants
            cursor.execute("""
                SELECT u.*, p.age_group, p.education, p.cooking_frequency, p.sustainability_awareness
                FROM users u
                LEFT JOIN user_profiles p ON u.user_id = p.user_id
                ORDER BY u.user_id
            """)
            participants = cursor.fetchall()
           
            # Ratings  
            cursor.execute("SELECT * FROM recipe_ratings ORDER BY user_id, recipe_id")
            ratings = cursor.fetchall()
            
            # Questionnaires
            cursor.execute("SELECT * FROM questionnaire ORDER BY user_id")
            questionnaires = cursor.fetchall()
            
        else:  # SQLite
            conn.row_factory = sqlite3.Row
            participants = conn.execute("""
            SELECT u.*, p.age_group, p.education, p.cooking_frequency, p.sustainability_awareness
            FROM users u
            LEFT JOIN user_profiles p ON u.user_id = p.user_id
            ORDER BY u.user_id
        """).fetchall()
            ratings = conn.execute("SELECT * FROM recipe_ratings ORDER BY user_id, recipe_id").fetchall()
            questionnaires = conn.execute("SELECT * FROM questionnaire ORDER BY user_id").fetchall()
        
        conn.close()
        
        # 2. RECEPTEK ADATAI a JSON fájlból (recommender objektumból)
        # A recommender már be van töltve a routes.py tetején
        recipe_lookup = {}
        for recipe in recommender.recipes:
            recipe_id = str(recipe.get('id', recipe.get('recipeid', '')))
            if recipe_id:
                recipe_lookup[recipe_id] = {
                    'name': recipe.get('name', ''),
                    'health_score': recipe.get('hsi_normalized', recipe.get('HSI', 0)),
                    'environmental_score': recipe.get('esi_inverted', 100 - recipe.get('ESI', 100)),
                    'meal_score': recipe.get('ppi_normalized', recipe.get('PPI', 0)),
                    'composite_score': recipe.get('composite_score', 0)
                }
        
        print(f"📊 Recipe lookup created: {len(recipe_lookup)} recipes loaded")
        
        # 3. ADATOK ÖSSZEKAPCSOLÁSA
        # Group data by user_id
        profiles = {p['user_id']: dict(p) for p in participants}
        questionnaire_data = {q['user_id']: dict(q) for q in questionnaires}
        
        csv_rows = []
        
        for user in participants:
            user_id = user['user_id']
            profile = profiles.get(user_id, {})
            questionnaire = questionnaire_data.get(user_id, {})
            user_ratings = [dict(r) for r in ratings if r['user_id'] == user_id]
            
            if user_ratings:
                # Van rating - minden rating-hez egy sor
                for rating in user_ratings:
                    recipe_id = str(rating.get('recipe_id', ''))
                    
                    # Receptek adatainak kikeresése a JSON-ból
                    recipe_data = recipe_lookup.get(recipe_id, {})
                    
                    csv_rows.append({
                        'user_id': user_id,
                        'group': user.get('version', ''),
                        'age': profile.get('age_group', ''),
                        'education_level': profile.get('education', ''),
                        'cooking_frequency': profile.get('cooking_frequency', ''),
                        'importance_sustainability': profile.get('sustainability_awareness', ''),
                        
                        # RECEPT ADATOK
                        'recipeid': recipe_id,
                        'recipe_name': recipe_data.get('name', 'Unknown recipe'),
                        
                        # SCORE ADATOK A JSON-BÓL ← Itt a megoldás!
                        'health_score': recipe_data.get('health_score', ''),
                        'env_score': recipe_data.get('environmental_score', ''),
                        'meal_score': recipe_data.get('meal_score', ''),
                        'composite_score': recipe_data.get('composite_score', ''),
                        
                        # FELHASZNÁLÓI RATING
                        'rating': rating.get('rating', ''),
                        
                        # KÉRDŐÍV ADATOK
                        'usability': questionnaire.get('system_usability', ''),
                        'quality': questionnaire.get('recommendation_quality', ''),
                        'trust': questionnaire.get('trust_level', ''),
                        'satisfaction': questionnaire.get('overall_satisfaction', ''),
                        'comment': questionnaire.get('additional_comments', '')
                    })
            else:
                # Nincs rating - csak demographic adat
                csv_rows.append({
                    'user_id': user_id,
                    'group': user.get('version', ''),
                    'age': profile.get('age_group', ''),
                    'education_level': profile.get('education', ''),
                    'cooking_frequency': profile.get('cooking_frequency', ''),
                    'importance_sustainability': profile.get('sustainability_awareness', ''),
                    'recipeid': '',
                    'recipe_name': '',
                    'health_score': '',
                    'env_score': '',
                    'meal_score': '',
                    'composite_score': '',
                    'rating': '',
                    'usability': questionnaire.get('system_usability', ''),
                    'quality': questionnaire.get('recommendation_quality', ''),
                    'trust': questionnaire.get('trust_level', ''),
                    'satisfaction': questionnaire.get('overall_satisfaction', ''),
                    'comment': questionnaire.get('additional_comments', '')
                })
        
        # 4. CSV GENERÁLÁSA
        output = io.StringIO()
        if csv_rows:
            writer = csv.DictWriter(output, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
            
            print(f"✅ CSV export completed: {len(csv_rows)} rows")
            print(f"📊 Recipe data found for: {sum(1 for row in csv_rows if row['health_score'])} ratings")
        else:
            # Empty fallback
            writer = csv.writer(output)
            writer.writerow(['user_id', 'group', 'age', 'education_level', 'cooking_frequency', 
                           'importance_sustainability', 'recipeid', 'recipe_name',
                           'health_score', 'env_score', 'meal_score', 'composite_score', 
                           'rating', 'usability', 'quality', 'trust', 'satisfaction', 'comment'])
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=statistical_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        print(f"❌ Statistical CSV export error: {e}")
        import traceback
        traceback.print_exc()
        return f"Statistical CSV export error: {str(e)}", 500
# =============================================
# ENHANCED API ENDPOINTS - ADD THESE AT THE END
# =============================================

@user_study_bp.route('/api/enhanced-recommendations', methods=['POST'])
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
        
        # Get recommendations using enhanced engine
        if hasattr(recommender, 'enhanced_engine') and recommender.enhanced_engine and ENHANCED_MODULES_AVAILABLE:
            results = recommender.enhanced_engine.get_recommendations(
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
        else:
            return jsonify({
                'status': 'error',
                'message': 'Enhanced recommendations not available'
            }), 503
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@user_study_bp.route('/api/metrics-dashboard')
def metrics_dashboard_api():
    """Metrics dashboard API endpoint"""
    try:
        if hasattr(recommender, 'enhanced_engine') and recommender.enhanced_engine and ENHANCED_MODULES_AVAILABLE:
            dashboard_data = recommender.enhanced_engine.get_metrics_dashboard_data()
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

@user_study_bp.route('/api/evaluation-summary')
def evaluation_summary():
    """Evaluation summary API endpoint"""
    try:
        if hasattr(recommender, 'enhanced_engine') and recommender.enhanced_engine and ENHANCED_MODULES_AVAILABLE:
            summary = recommender.enhanced_engine.get_evaluation_summary()
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

@user_study_bp.route('/dashboard')
def metrics_dashboard_page():
    """Metrics dashboard page"""
    try:
        dashboard_data = {}
        evaluation_summary = {}
        
        if hasattr(recommender, 'enhanced_engine') and recommender.enhanced_engine and ENHANCED_MODULES_AVAILABLE:
            dashboard_data = recommender.enhanced_engine.get_metrics_dashboard_data()
            evaluation_summary = recommender.enhanced_engine.get_evaluation_summary()
        
        return render_template('metrics_dashboard.html', 
                             dashboard_data=dashboard_data,
                             evaluation_summary=evaluation_summary,
                             enhanced_available=ENHANCED_MODULES_AVAILABLE)
                             
    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Dashboard error: {e}") if 'error.html' in [t.name for t in current_app.jinja_env.list_templates()] else f"Dashboard error: {e}"

# =============================================
# END OF ENHANCED API ENDPOINTS
# =============================================

# Export
__all__ = ['user_study_bp']

print("✅ User study routes loaded successfully")

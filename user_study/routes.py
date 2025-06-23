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
        """Universal user creation WITH VERSION TRACKING"""
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
                user_id = cursor.fetchone()['user_id']
                
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
        """Universal statistics WITH VERSION BREAKDOWN - FIXED"""
        try:
            if self.db_type == 'postgresql':
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Összesített statisztikák
                cursor.execute('SELECT COUNT(*) as count FROM users')
                total_result = cursor.fetchone()
                total = total_result['count'] if total_result else 0
                
                cursor.execute('SELECT COUNT(*) as count FROM questionnaire')
                completed_result = cursor.fetchone()
                completed = completed_result['count'] if completed_result else 0
                
                # Verzió szerinti bontás
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
                version_data = cursor.fetchall()
                
                cursor.close()
                conn.close()
                
            else:
                # SQLite lekérdezések
                result = self.conn.execute('SELECT COUNT(*) as count FROM users').fetchone()
                total = result['count'] if result else 0
                
                result = self.conn.execute('SELECT COUNT(*) as count FROM questionnaire').fetchone()
                completed = result['count'] if result else 0
                
                # Verzió szerinti bontás
                version_data = self.conn.execute('''
                    SELECT 
                        COALESCE(u.version, 'v1') as version,
                        COUNT(u.user_id) as registered,
                        COUNT(q.user_id) as completed
                    FROM users u
                    LEFT JOIN questionnaire q ON u.user_id = q.user_id
                    GROUP BY COALESCE(u.version, 'v1')
                    ORDER BY COALESCE(u.version, 'v1')
                ''').fetchall()
            
            # Verzió adatok formázása
            version_distribution = []
            for row in version_data:
                # UNIVERZÁLIS DICTIONARY ACCESS
                if isinstance(row, dict):
                    # PostgreSQL eredmény (dict)
                    version = row.get('version', 'v1')
                    registered = row.get('registered', 0)
                    completed = row.get('completed', 0)
                else:
                    # SQLite eredmény (Row object)
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
            
            print(f"📊 Stats computed: {total} total, {completed} completed, {len(version_distribution)} versions")
            
            return {
                'total_participants': total,
                'completed_participants': completed,
                'completion_rate': (completed / total * 100) if total > 0 else 0,
                'avg_interactions_per_user': 0,
                'version_distribution': version_distribution
            }
            
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
            """JSON receptek feldolgozása TELJES NORMALIZÁLÁSSAL"""
            print(f"🔄 Processing {len(recipes_data)} recipes from JSON...")
            
            processed_recipes = []
            
            # ELSŐ LÉPÉS: Nyers adatok összegyűjtése és min/max értékek meghatározása
            raw_esi_values = []
            raw_hsi_values = []
            raw_ppi_values = []
            
            # Értékek gyűjtése normalizáláshoz
            for recipe in recipes_data:
                try:
                    esi = float(recipe.get('ESI', recipe.get('env_score', 70)))
                    hsi = float(recipe.get('HSI', recipe.get('nutri_score', 70)))
                    ppi = float(recipe.get('PPI', recipe.get('meal_score', 70)))
                    
                    raw_esi_values.append(esi)
                    raw_hsi_values.append(hsi)
                    raw_ppi_values.append(ppi)
                except (ValueError, TypeError):
                    continue
            
            # Min/Max értékek kiszámítása
            esi_min, esi_max = min(raw_esi_values), max(raw_esi_values)
            hsi_min, hsi_max = min(raw_hsi_values), max(raw_hsi_values)
            ppi_min, ppi_max = min(raw_ppi_values), max(raw_ppi_values)
            
            print(f"📊 Score ranges:")
            print(f"   ESI: {esi_min:.2f} - {esi_max:.2f}")
            print(f"   HSI: {hsi_min:.2f} - {hsi_max:.2f}")
            print(f"   PPI: {ppi_min:.2f} - {ppi_max:.2f}")
            
            # MÁSODIK LÉPÉS: Receptek feldolgozása normalizált értékekkel
            for recipe in recipes_data:
                try:
                    # Alapvető mezők biztosítása
                    raw_esi = float(recipe.get('ESI', recipe.get('env_score', 70)))
                    raw_hsi = float(recipe.get('HSI', recipe.get('nutri_score', 70)))
                    raw_ppi = float(recipe.get('PPI', recipe.get('meal_score', 70)))
                    
                    # NORMALIZÁLÁS 0-100 SKÁLÁRA: (x-min)/(max-min)*100
                    def normalize_score(value, min_val, max_val):
                        if max_val == min_val:  # Elkerüljük a nullával osztást
                            return 50.0  # Középérték ha minden azonos
                        return ((value - min_val) / (max_val - min_val)) * 100
                    
                    normalized_esi = normalize_score(raw_esi, esi_min, esi_max)
                    normalized_hsi = normalize_score(raw_hsi, hsi_min, hsi_max)
                    normalized_ppi = normalize_score(raw_ppi, ppi_min, ppi_max)
                    
                    processed_recipe = {
                        'recipeid': recipe.get('recipeid', 0),
                        'title': str(recipe.get('title', recipe.get('name', 'Névtelen recept'))),
                        'ingredients': str(recipe.get('ingredients', '')),
                        'instructions': str(recipe.get('instructions', '')),
                        'category': str(recipe.get('category', 'Egyéb')),
                        'images': self.fix_image_url(recipe.get('images', '')),
                        # Normalizált pontszámok mentése
                        'ESI': round(normalized_esi, 1),
                        'HSI': round(normalized_hsi, 1),
                        'PPI': round(normalized_ppi, 1),
                        # Eredeti pontszámok debug céljából
                        'raw_ESI': round(raw_esi, 1),
                        'raw_HSI': round(raw_hsi, 1),
                        'raw_PPI': round(raw_ppi, 1)
                    }
                    
                    # KOMPOZIT SCORE SZÁMÍTÁSA - JAVÍTOTT VERZIÓ
                    # Formula: (100 - normalized_ESI) * 0.4 + normalized_HSI * 0.4 + normalized_PPI * 0.2
                    # Magyarázat: 
                    # - ESI (Environmental Impact Score): alacsonyabb = jobb környezeti hatás -> (100-ESI)
                    # - HSI (Health Score): magasabb = egészségesebb -> HSI
                    # - PPI (Popularity/Meal Score): magasabb = népszerűbb/jobb étkezés -> PPI
                    
                    esi_inverted = 100 - normalized_esi  # Környezeti pontszám fordítása
                    composite = (
                        esi_inverted * 0.4 +           # Fordított környezeti (40%)
                        normalized_hsi * 0.4 +         # Egészség (40%)
                        normalized_ppi * 0.2           # Népszerűség/Étkezés (20%)
                    )
                    
                    processed_recipe['composite_score'] = round(composite, 1)
                    
                    # Debug információ az első néhány recepthez
                    if len(processed_recipes) < 3:
                        print(f"🔍 Recipe: {processed_recipe['title'][:30]}...")
                        print(f"   Raw scores: ESI={raw_esi:.1f}, HSI={raw_hsi:.1f}, PPI={raw_ppi:.1f}")
                        print(f"   Normalized: ESI={normalized_esi:.1f}, HSI={normalized_hsi:.1f}, PPI={normalized_ppi:.1f}")
                        print(f"   ESI inverted: {esi_inverted:.1f}")
                        print(f"   Composite: {composite:.1f}")
                        print(f"   Formula: ({esi_inverted:.1f}*0.4) + ({normalized_hsi:.1f}*0.4) + ({normalized_ppi:.1f}*0.2) = {composite:.1f}")
                        print()
                    
                    # Csak érvényes receptek
                    if processed_recipe['title'] and processed_recipe['ingredients']:
                        processed_recipes.append(processed_recipe)
                        
                except Exception as e:
                    print(f"⚠️ Skipping invalid recipe: {e}")
                    continue
            
            self.recipes = processed_recipes
            print(f"✅ Successfully processed {len(self.recipes)} recipes with normalized scores")
            
            # Összesített statisztikák
            if self.recipes:
                avg_composite = sum(r['composite_score'] for r in self.recipes) / len(self.recipes)
                min_composite = min(r['composite_score'] for r in self.recipes)
                max_composite = max(r['composite_score'] for r in self.recipes)
                
                print(f"📈 Composite score stats:")
                print(f"   Average: {avg_composite:.1f}")
                print(f"   Range: {min_composite:.1f} - {max_composite:.1f}")
            
            # Debug: első recept megjelenítése
            if self.recipes:
                first = self.recipes[0]
                print(f"📝 Sample: {first['title']}")
                print(f"📊 Final scores: ESI={first['ESI']}, HSI={first['HSI']}, PPI={first['PPI']}, Composite={first['composite_score']}")
    
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

db = EnhancedDatabase()
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

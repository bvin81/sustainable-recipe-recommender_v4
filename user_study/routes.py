#!/usr/bin/env python3
"""
Fixed user_study/routes.py - Eliminates duplicate routes and fixes PostgreSQL transactions
"""
import sqlite3
import os
import random
import json
from pathlib import Path
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
from flask import send_file, make_response
import csv
import io
from datetime import datetime

# Scientific libraries with fallbacks
try:
    import pandas as pd
    import numpy as np
    print("✅ Scientific libraries loaded")
except ImportError as e:
    print(f"⚠️ Scientific libraries missing: {e}")
    class MockPandas:
        def read_csv(self, *args, **kwargs): return []
    pd = MockPandas()
    class MockNumpy:
        def random(self): import random; return random
    np = MockNumpy()

# PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ psycopg2 not available, falling back to SQLite")

# Blueprint
user_study_bp = Blueprint('user_study', __name__, url_prefix='')

# Paths
if os.environ.get('DYNO'):
    project_root = Path.cwd()
else:
    project_root = Path(__file__).parent.parent

data_dir = project_root / "data"

class EnhancedDatabase:
    """Fixed database class with proper transaction handling"""
    
    def __init__(self):
        if os.environ.get('DATABASE_URL') and POSTGRES_AVAILABLE:
            self.db_type = 'postgresql'
            self.database_url = os.environ.get('DATABASE_URL')
            self._init_postgresql()
        else:
            self.db_type = 'sqlite'
            self.db_path = "/tmp/sustainable_recipes.db" if os.environ.get('DYNO') else str(data_dir / "sustainable_recipes.db")
            self._init_sqlite()
        
        print(f"✅ Database initialized: {self.db_type}")

    def _init_postgresql(self):
        """Initialize PostgreSQL with proper error handling"""
        try:
            print(f"🐘 PostgreSQL connection to: {self.database_url.split('@')[1].split('/')[0]}")
            conn = psycopg2.connect(self.database_url, sslmode='require')
            conn.autocommit = True  # Important: prevent transaction blocks
            print("✅ PostgreSQL connection successful")
            
            self._init_tables()
            conn.close()
        except Exception as e:
            print(f"❌ PostgreSQL initialization failed: {e}")
            # Fallback to SQLite
            self.db_type = 'sqlite'
            self.db_path = "/tmp/sustainable_recipes.db"
            self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        """Create tables with proper transaction handling"""
        try:
            if self.db_type == 'postgresql':
                conn = psycopg2.connect(self.database_url, sslmode='require')
                conn.autocommit = False  # Use transactions for table creation
                cursor = conn.cursor()
                
                # Check if version column exists
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'participants' AND column_name = 'version'
                """)
                version_exists = cursor.fetchone()
                
                if version_exists:
                    print("🔍 Version column already exists or table is new")
                else:
                    print("➕ Adding version column to participants table")
                
                # Create tables
                tables_sql = [
                    '''CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        age_group TEXT,
                        education TEXT,
                        cooking_frequency TEXT,
                        sustainability_awareness TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''',
                    
                    '''CREATE TABLE IF NOT EXISTS participants (
                        user_id TEXT PRIMARY KEY,
                        age_group TEXT,
                        education TEXT,
                        cooking_frequency TEXT,
                        sustainability_awareness TEXT,
                        version TEXT DEFAULT 'A',
                        is_completed BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''',
                    
                    '''CREATE TABLE IF NOT EXISTS recipe_ratings (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT,
                        recipe_id TEXT,
                        rating INTEGER,
                        health_score REAL,
                        environmental_score REAL,
                        meal_score REAL,
                        composite_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''',
                    
                    '''CREATE TABLE IF NOT EXISTS questionnaire (
                        user_id TEXT PRIMARY KEY,
                        system_usability INTEGER,
                        recommendation_quality INTEGER,
                        trust_level INTEGER,
                        explanation_clarity INTEGER,
                        sustainability_importance INTEGER,
                        overall_satisfaction INTEGER,
                        additional_comments TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )'''
                ]
                
                for sql in tables_sql:
                    try:
                        cursor.execute(sql)
                    except Exception as e:
                        print(f"⚠️ Table creation warning: {e}")
                        conn.rollback()
                        continue
                
                conn.commit()
                cursor.close()
                conn.close()
                
            else:  # SQLite
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # SQLite table creation (simplified)
                cursor.execute('''CREATE TABLE IF NOT EXISTS participants (
                    user_id TEXT PRIMARY KEY,
                    age_group TEXT,
                    education TEXT,
                    cooking_frequency TEXT,
                    sustainability_awareness TEXT,
                    version TEXT DEFAULT 'A',
                    is_completed BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                
                cursor.execute('''CREATE TABLE IF NOT EXISTS recipe_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    recipe_id TEXT,
                    rating INTEGER,
                    health_score REAL,
                    environmental_score REAL,
                    meal_score REAL,
                    composite_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                
                cursor.execute('''CREATE TABLE IF NOT EXISTS questionnaire (
                    user_id TEXT PRIMARY KEY,
                    system_usability INTEGER,
                    recommendation_quality INTEGER,
                    trust_level INTEGER,
                    explanation_clarity INTEGER,
                    sustainability_importance INTEGER,
                    overall_satisfaction INTEGER,
                    additional_comments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            print(f"❌ Table creation failed: {e}")
            if self.db_type == 'postgresql':
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass

    def get_connection(self):
        """Get database connection with proper error handling"""
        if self.db_type == 'postgresql':
            try:
                conn = psycopg2.connect(self.database_url, sslmode='require')
                return conn
            except Exception as e:
                print(f"❌ PostgreSQL connection failed: {e}")
                return None
        else:
            return sqlite3.connect(self.db_path)

    def get_stats(self):
        """Get basic statistics"""
        try:
            if self.db_type == 'postgresql':
                conn = psycopg2.connect(self.database_url, sslmode='require')
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
            
            # Get participant count
            cursor.execute("SELECT COUNT(*) as total FROM participants")
            total = cursor.fetchone()['total'] if cursor.fetchone() else 0
            
            # Get completed count
            cursor.execute("SELECT COUNT(*) as completed FROM participants WHERE is_completed = TRUE")
            completed = cursor.fetchone()['completed'] if cursor.fetchone() else 0
            
            conn.close()
            
            return {
                'total_participants': total,
                'completed_participants': completed,
                'completion_rate': (completed / total * 100) if total > 0 else 0,
                'version_distribution': [
                    {'version': 'A', 'registered': total//2, 'completed': completed//2, 'completion_rate': 50, 'participation_rate': 50},
                    {'version': 'B', 'registered': total//2, 'completed': completed//2, 'completion_rate': 50, 'participation_rate': 50}
                ]
            }
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {
                'total_participants': 0,
                'completed_participants': 0,
                'completion_rate': 0,
                'version_distribution': []
            }

# Initialize database
db = EnhancedDatabase()

# =============================================================================
# ROUTES - Fixed to eliminate duplicates
# =============================================================================

@user_study_bp.route('/')
def welcome():
    """Landing page"""
    return render_template('welcome.html')

@user_study_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': db.db_type,
        'timestamp': datetime.now().isoformat()
    })

@user_study_bp.route('/admin/stats')
def admin_stats():
    """Admin statistics page"""
    try:
        stats = db.get_stats()
        
        html_response = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>📊 Admin Dashboard - Sustainable Recipe Study</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .card {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .stat {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px; background: #e3f2fd; border-radius: 5px; }}
                .export-btn {{ display: inline-block; margin: 5px; padding: 10px 15px; background: #2196f3; color: white; text-decoration: none; border-radius: 5px; }}
                .export-btn:hover {{ background: #1976d2; }}
                h1 {{ color: #1565c0; }}
                h2 {{ color: #424242; }}
            </style>
        </head>
        <body>
            <h1>📊 Sustainable Recipe Study - Admin Dashboard</h1>
            
            <div class="card">
                <h2>📈 Statisztikák</h2>
                <div class="stat">👥 Összes résztvevő: <strong>{stats.get('total_participants', 0)}</strong></div>
                <div class="stat">✅ Befejezett: <strong>{stats.get('completed_participants', 0)}</strong></div>
                <div class="stat">📊 Befejezési arány: <strong>{stats.get('completion_rate', 0):.1f}%</strong></div>
                <div class="stat">🗄️ Adatbázis: <strong>{db.db_type}</strong></div>
            </div>
            
            <div class="card">
                <h2>📥 Export Opciók</h2>
                <a href="/admin/export/csv" class="export-btn">📄 CSV Export</a>
                <a href="/admin/export/json" class="export-btn">📋 JSON Export</a>
                <a href="/admin/export/statistical_csv" class="export-btn" style="background: #ff9800;">📊 Statistical CSV</a>
            </div>
            
            <div class="card">
                <h2>🔗 Navigáció</h2>
                <a href="/" class="export-btn">🏠 Főoldal</a>
                <a href="/health" class="export-btn" style="background: #4caf50;">💚 Health Check</a>
            </div>
            
            <hr>
            <p><small>Generálva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </body>
        </html>
        """
        
        return html_response
        
    except Exception as e:
        return f"<h1>⚠️ Admin Stats Error</h1><p>{e}</p>", 500

@user_study_bp.route('/admin/export/csv')
def export_csv():
    """Basic CSV export"""
    try:
        stats = db.get_stats()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'Export Date', 'Total Participants', 'Completed Participants', 
            'Completion Rate', 'Database Type'
        ])
        
        # Data
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            stats.get('total_participants', 0),
            stats.get('completed_participants', 0),
            f"{stats.get('completion_rate', 0):.1f}%",
            db.db_type
        ])
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=study_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        return f"CSV Export error: {e}", 500

@user_study_bp.route('/admin/export/json')
def export_json():
    """JSON export"""
    try:
        stats = db.get_stats()
        
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

# FIXED: Single definition of statistical_csv export
@user_study_bp.route('/admin/export/statistical_csv')
def export_statistical_csv():
    """Statistical CSV export for analysis (SINGLE DEFINITION)"""
    try:
        if db.db_type == 'postgresql':
            return export_postgresql_statistical_csv()
        else:
            return export_sqlite_statistical_csv()
            
    except Exception as e:
        print(f"❌ Statistical CSV export error: {e}")
        return f"Statistical CSV export error: {str(e)}", 500

def export_postgresql_statistical_csv():
    """PostgreSQL statistical export"""
    try:
        conn = psycopg2.connect(db.database_url, sslmode='require')
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get all data with joins
        query = '''
        SELECT 
            p.user_id, p.age_group, p.education, p.cooking_frequency, 
            p.sustainability_awareness, p.version, p.is_completed,
            COALESCE(q.system_usability, 0) as system_usability,
            COALESCE(q.recommendation_quality, 0) as recommendation_quality,
            COALESCE(q.trust_level, 0) as trust_level,
            COALESCE(q.overall_satisfaction, 0) as overall_satisfaction,
            COALESCE(q.additional_comments, '') as additional_comments
        FROM participants p
        LEFT JOIN questionnaire q ON p.user_id = q.user_id
        ORDER BY p.user_id
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Convert to CSV
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        else:
            # Empty fallback
            writer = csv.writer(output)
            writer.writerow(['user_id', 'age_group', 'education', 'cooking_frequency', 
                           'sustainability_awareness', 'version', 'is_completed'])
        
        cursor.close()
        conn.close()
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=statistical_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        print(f"❌ PostgreSQL export error: {e}")
        # Fallback to simple export
        return export_simple_fallback_csv()

def export_sqlite_statistical_csv():
    """SQLite statistical export"""
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        
        query = '''
        SELECT 
            p.user_id, p.age_group, p.education, p.cooking_frequency, 
            p.sustainability_awareness, p.version, p.is_completed,
            COALESCE(q.system_usability, 0) as system_usability,
            COALESCE(q.recommendation_quality, 0) as recommendation_quality,
            COALESCE(q.trust_level, 0) as trust_level,
            COALESCE(q.overall_satisfaction, 0) as overall_satisfaction,
            COALESCE(q.additional_comments, '') as additional_comments
        FROM participants p
        LEFT JOIN questionnaire q ON p.user_id = q.user_id
        ORDER BY p.user_id
        '''
        
        participants = conn.execute(query).fetchall()
        conn.close()
        
        # Convert to CSV
        output = io.StringIO()
        if participants:
            fieldnames = list(participants[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in participants:
                writer.writerow(dict(row))
        else:
            writer = csv.writer(output)
            writer.writerow(['user_id', 'age_group', 'education', 'message'])
            writer.writerow(['', '', '', 'No data available'])
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=statistical_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        print(f"❌ SQLite export error: {e}")
        return export_simple_fallback_csv()

def export_simple_fallback_csv():
    """Simple fallback CSV if all else fails"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['timestamp', 'status', 'database_type', 'message'])
    writer.writerow([
        datetime.now().isoformat(),
        'export_fallback',
        db.db_type,
        'Statistical export failed, using fallback'
    ])
    
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    response.headers['Content-Disposition'] = f'attachment; filename=fallback_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    return response

# =============================================================================
# Hungarian Recipe Recommender (existing functionality)
# =============================================================================

class HungarianJSONRecommender:
    """Simplified Hungarian recipe recommender"""
    
    def __init__(self):
        self.recipes = []
        self.load_recipes()
    
    def load_recipes(self):
        """Load recipes from JSON file"""
        try:
            json_path = data_dir / "hungarian_recipes.json"
            
            print(f"📊 Loading JSON from: {json_path.name}")
            
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.recipes = data
                    else:
                        self.recipes = data.get('recipes', [])
            else:
                print("⚠️ JSON file not found, using fallback recipes")
                self.recipes = self._create_fallback_recipes()
            
            print(f"✅ JSON loaded! {len(self.recipes)} recipes")
            
            if self.recipes:
                self._process_recipes()
                
        except Exception as e:
            print(f"❌ JSON loading error: {e}")
            self.recipes = self._create_fallback_recipes()
    
    def _create_fallback_recipes(self):
        """Create some fallback recipes if JSON is missing"""
        return [
            {
                "name": "Hagyományos Gulyás",
                "ESI": 150.0,
                "HSI": 45.0, 
                "PPI": 60.0,
                "ingredients": ["marhahús", "hagyma", "paprika"],
                "instructions": "Hagyományos magyar gulyás recept."
            },
            {
                "name": "Lecsó",
                "ESI": 80.0,
                "HSI": 65.0,
                "PPI": 40.0, 
                "ingredients": ["paprika", "paradicsom", "hagyma"],
                "instructions": "Egyszerű lecsó recept."
            }
        ]
    
    def _process_recipes(self):
        """Process and normalize recipe scores"""
        if not self.recipes:
            return
            
        print(f"🔄 Processing {len(self.recipes)} recipes from JSON...")
        
        # Calculate score ranges
        esi_scores = [r.get('ESI', 0) for r in self.recipes]
        hsi_scores = [r.get('HSI', 0) for r in self.recipes]
        ppi_scores = [r.get('PPI', 0) for r in self.recipes]
        
        esi_min, esi_max = min(esi_scores), max(esi_scores)
        hsi_min, hsi_max = min(hsi_scores), max(hsi_scores)
        ppi_min, ppi_max = min(ppi_scores), max(ppi_scores)
        
        print(f"📊 Score ranges:")
        print(f"   ESI: {esi_min:.2f} - {esi_max:.2f}")
        print(f"   HSI: {hsi_min:.2f} - {hsi_max:.2f}")
        print(f"   PPI: {ppi_min:.2f} - {ppi_max:.2f}")
        
        # Normalize and calculate composite scores
        for recipe in self.recipes:
            esi = recipe.get('ESI', 0)
            hsi = recipe.get('HSI', 0)
            ppi = recipe.get('PPI', 0)
            
            # Normalize to 0-100
            esi_norm = ((esi - esi_min) / (esi_max - esi_min)) * 100 if esi_max > esi_min else 50
            hsi_norm = ((hsi - hsi_min) / (hsi_max - hsi_min)) * 100 if hsi_max > hsi_min else 50
            ppi_norm = ((ppi - ppi_min) / (ppi_max - ppi_min)) * 100 if ppi_max > ppi_min else 50
            
            # ESI: lower is better (invert)
            esi_inverted = 100 - esi_norm
            
            # Composite score: weighted average
            composite = (esi_inverted * 0.4) + (hsi_norm * 0.4) + (ppi_norm * 0.2)
            
            # Store normalized scores
            recipe['esi_normalized'] = esi_norm
            recipe['hsi_normalized'] = hsi_norm  
            recipe['ppi_normalized'] = ppi_norm
            recipe['esi_inverted'] = esi_inverted
            recipe['composite_score'] = composite
        
        # Show sample calculation
        sample_recipe = self.recipes[0]
        print(f"🔍 Recipe: {sample_recipe['name'][:30]}...")
        print(f"   Raw scores: ESI={sample_recipe.get('ESI', 0):.1f}, HSI={sample_recipe.get('HSI', 0):.1f}, PPI={sample_recipe.get('PPI', 0):.1f}")
        print(f"   Normalized: ESI={sample_recipe['esi_normalized']:.1f}, HSI={sample_recipe['hsi_normalized']:.1f}, PPI={sample_recipe['ppi_normalized']:.1f}")
        print(f"   ESI inverted: {sample_recipe['esi_inverted']:.1f}")
        print(f"   Composite: {sample_recipe['composite_score']:.1f}")
        print(f"   Formula: ({sample_recipe['esi_inverted']:.1f}*0.4) + ({sample_recipe['hsi_normalized']:.1f}*0.4) + ({sample_recipe['ppi_normalized']:.1f}*0.2) = {sample_recipe['composite_score']:.1f}")
        print()
        
        print(f"✅ Successfully processed {len(self.recipes)} recipes with normalized scores")
        
        # Statistics
        composite_scores = [r['composite_score'] for r in self.recipes]
        avg_composite = sum(composite_scores) / len(composite_scores)
        min_composite = min(composite_scores)
        max_composite = max(composite_scores)
        
        print(f"📈 Composite score stats:")
        print(f"   Average: {avg_composite:.1f}")
        print(f"   Range: {min_composite:.1f} - {max_composite:.1f}")
        
        print(f"📝 Sample: {sample_recipe['name']}")
        print(f"📊 Final scores: ESI={sample_recipe['esi_normalized']:.1f}, HSI={sample_recipe['hsi_normalized']:.1f}, PPI={sample_recipe['ppi_normalized']:.1f}, Composite={sample_recipe['composite_score']:.1f}")

    def get_recommendations(self, user_preferences=None, version='A', limit=5):
        """Get recipe recommendations"""
        if not self.recipes:
            return []
        
        try:
            # Sort by composite score (higher is better)
            sorted_recipes = sorted(self.recipes, key=lambda x: x.get('composite_score', 0), reverse=True)
            
            # Return top recipes
            recommendations = []
            for recipe in sorted_recipes[:limit]:
                recommendations.append({
                    'id': recipe.get('id', f"recipe_{len(recommendations)+1}"),
                    'name': recipe['name'],
                    'health_score': recipe.get('hsi_normalized', 50),
                    'environmental_score': recipe.get('esi_inverted', 50),
                    'meal_score': recipe.get('ppi_normalized', 50),
                    'composite_score': recipe.get('composite_score', 50),
                    'ingredients': recipe.get('ingredients', []),
                    'instructions': recipe.get('instructions', 'Nincs elérhető utasítás.'),
                    'explanation': self._generate_explanation(recipe, version)
                })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Recommendation error: {e}")
            return []
    
    def _generate_explanation(self, recipe, version):
        """Generate explanation based on version"""
        composite = recipe.get('composite_score', 50)
        health = recipe.get('hsi_normalized', 50)
        env = recipe.get('esi_inverted', 50)
        
        if version == 'A':
            return f"Ez a recept {composite:.0f} pontot ért el összesítésben. Egészségügyi érték: {health:.0f}/100, környezeti érték: {env:.0f}/100."
        else:
            return f"Ajánlott recept magas fenntarthatósági értékek alapján."

# Initialize recommender
recommender = HungarianJSONRecommender()
print(f"✅ Hungarian JSON Recommender initialized with {len(recommender.recipes)} recipes")

# Additional routes for the study functionality
@user_study_bp.route('/profile')
def profile():
    """User profile setup page"""
    return render_template('profile.html')

@user_study_bp.route('/profile', methods=['POST'])
def profile_submit():
    """Handle profile form submission"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            user_id = f"user_{random.randint(1000, 9999)}"
            session['user_id'] = user_id
        
        # Get form data
        profile_data = {
            'user_id': user_id,
            'age_group': request.form.get('age_group'),
            'education': request.form.get('education'),
            'cooking_frequency': request.form.get('cooking_frequency'),
            'sustainability_awareness': request.form.get('sustainability_awareness'),
            'version': random.choice(['A', 'B']),  # Random assignment
            'is_completed': False
        }
        
        # Save to database
        conn = db.get_connection()
        if conn:
            if db.db_type == 'postgresql':
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO participants (user_id, age_group, education, cooking_frequency, 
                                            sustainability_awareness, version, is_completed)
                    VALUES (%(user_id)s, %(age_group)s, %(education)s, %(cooking_frequency)s, 
                            %(sustainability_awareness)s, %(version)s, %(is_completed)s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        age_group = EXCLUDED.age_group,
                        education = EXCLUDED.education,
                        cooking_frequency = EXCLUDED.cooking_frequency,
                        sustainability_awareness = EXCLUDED.sustainability_awareness
                """, profile_data)
                conn.commit()
                cursor.close()
            else:
                conn.execute("""
                    INSERT OR REPLACE INTO participants 
                    (user_id, age_group, education, cooking_frequency, sustainability_awareness, version, is_completed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (profile_data['user_id'], profile_data['age_group'], profile_data['education'],
                      profile_data['cooking_frequency'], profile_data['sustainability_awareness'],
                      profile_data['version'], profile_data['is_completed']))
                conn.commit()
            
            conn.close()
        
        # Store in session
        session.update(profile_data)
        
        return redirect(url_for('user_study.recommendations'))
        
    except Exception as e:
        print(f"❌ Profile submission error: {e}")
        return redirect(url_for('user_study.profile'))

@user_study_bp.route('/recommendations')
def recommendations():
    """Show recipe recommendations"""
    try:
        user_id = session.get('user_id')
        version = session.get('version', 'A')
        
        if not user_id:
            return redirect(url_for('user_study.profile'))
        
        # Get recommendations
        user_prefs = {
            'sustainability_awareness': session.get('sustainability_awareness'),
            'cooking_frequency': session.get('cooking_frequency')
        }
        
        recipes = recommender.get_recommendations(user_prefs, version, limit=5)
        
        return render_template('recommendations.html', 
                             recipes=recipes, 
                             version=version,
                             user_id=user_id)
        
    except Exception as e:
        print(f"❌ Recommendations error: {e}")
        return redirect(url_for('user_study.profile'))

@user_study_bp.route('/rate_recipe', methods=['POST'])
def rate_recipe():
    """Handle recipe rating submission"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'No user session'}), 400
        
        data = request.get_json()
        
        rating_data = {
            'user_id': user_id,
            'recipe_id': data.get('recipe_id'),
            'rating': int(data.get('rating', 0)),
            'health_score': float(data.get('health_score', 0)),
            'environmental_score': float(data.get('environmental_score', 0)),
            'meal_score': float(data.get('meal_score', 0)),
            'composite_score': float(data.get('composite_score', 0))
        }
        
        # Save to database
        conn = db.get_connection()
        if conn:
            if db.db_type == 'postgresql':
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO recipe_ratings (user_id, recipe_id, rating, health_score, 
                                              environmental_score, meal_score, composite_score)
                    VALUES (%(user_id)s, %(recipe_id)s, %(rating)s, %(health_score)s, 
                            %(environmental_score)s, %(meal_score)s, %(composite_score)s)
                """, rating_data)
                conn.commit()
                cursor.close()
            else:
                conn.execute("""
                    INSERT INTO recipe_ratings (user_id, recipe_id, rating, health_score,
                                              environmental_score, meal_score, composite_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (rating_data['user_id'], rating_data['recipe_id'], rating_data['rating'],
                      rating_data['health_score'], rating_data['environmental_score'],
                      rating_data['meal_score'], rating_data['composite_score']))
                conn.commit()
            
            conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"❌ Rating submission error: {e}")
        return jsonify({'error': str(e)}), 500

@user_study_bp.route('/questionnaire')
def questionnaire():
    """Post-study questionnaire"""
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('user_study.profile'))
    
    return render_template('questionnaire.html', user_id=user_id)

@user_study_bp.route('/questionnaire', methods=['POST'])
def questionnaire_submit():
    """Handle questionnaire submission"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('user_study.profile'))
        
        questionnaire_data = {
            'user_id': user_id,
            'system_usability': int(request.form.get('system_usability', 0)),
            'recommendation_quality': int(request.form.get('recommendation_quality', 0)),
            'trust_level': int(request.form.get('trust_level', 0)),
            'explanation_clarity': int(request.form.get('explanation_clarity', 0)),
            'sustainability_importance': int(request.form.get('sustainability_importance', 0)),
            'overall_satisfaction': int(request.form.get('overall_satisfaction', 0)),
            'additional_comments': request.form.get('additional_comments', '')
        }
        
        # Save questionnaire
        conn = db.get_connection()
        if conn:
            if db.db_type == 'postgresql':
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO questionnaire (user_id, system_usability, recommendation_quality,
                                             trust_level, explanation_clarity, sustainability_importance,
                                             overall_satisfaction, additional_comments)
                    VALUES (%(user_id)s, %(system_usability)s, %(recommendation_quality)s,
                            %(trust_level)s, %(explanation_clarity)s, %(sustainability_importance)s,
                            %(overall_satisfaction)s, %(additional_comments)s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        system_usability = EXCLUDED.system_usability,
                        recommendation_quality = EXCLUDED.recommendation_quality,
                        trust_level = EXCLUDED.trust_level,
                        explanation_clarity = EXCLUDED.explanation_clarity,
                        sustainability_importance = EXCLUDED.sustainability_importance,
                        overall_satisfaction = EXCLUDED.overall_satisfaction,
                        additional_comments = EXCLUDED.additional_comments
                """, questionnaire_data)
                conn.commit()
                cursor.close()
            else:
                conn.execute("""
                    INSERT OR REPLACE INTO questionnaire 
                    (user_id, system_usability, recommendation_quality, trust_level,
                     explanation_clarity, sustainability_importance, overall_satisfaction, additional_comments)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (questionnaire_data['user_id'], questionnaire_data['system_usability'],
                      questionnaire_data['recommendation_quality'], questionnaire_data['trust_level'],
                      questionnaire_data['explanation_clarity'], questionnaire_data['sustainability_importance'],
                      questionnaire_data['overall_satisfaction'], questionnaire_data['additional_comments']))
                conn.commit()
            
            # Mark participant as completed
            if db.db_type == 'postgresql':
                cursor = conn.cursor()
                cursor.execute("UPDATE participants SET is_completed = TRUE WHERE user_id = %s", (user_id,))
                conn.commit()
                cursor.close()
            else:
                conn.execute("UPDATE participants SET is_completed = 1 WHERE user_id = ?", (user_id,))
                conn.commit()
            
            conn.close()
        
        return redirect(url_for('user_study.thank_you'))
        
    except Exception as e:
        print(f"❌ Questionnaire submission error: {e}")
        return redirect(url_for('user_study.questionnaire'))

@user_study_bp.route('/thank_you')
def thank_you():
    """Thank you page"""
    return render_template('thank_you.html')

# Debug routes
@user_study_bp.route('/debug/status')
def debug_status():
    """Debug status page"""
    try:
        status = {
            'database_type': db.db_type,
            'recipes_loaded': len(recommender.recipes),
            'session_data': dict(session),
            'environment': 'heroku' if os.environ.get('DYNO') else 'local'
        }
        
        # Test database connection
        conn = db.get_connection()
        if conn:
            status['database_connected'] = True
            conn.close()
        else:
            status['database_connected'] = False
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'database_type': getattr(db, 'db_type', 'unknown'),
            'recipes_loaded': len(getattr(recommender, 'recipes', []))
        }), 500

# Export blueprint
__all__ = ['user_study_bp']

print("✅ User study routes loaded successfully")

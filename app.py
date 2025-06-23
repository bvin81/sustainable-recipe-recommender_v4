#!/usr/bin/env python3
"""
Egyszerűsített Flask alkalmazás
Minimális kód, maximum funkcionalitás
"""

import os
from flask import Flask, redirect, url_for, jsonify
from pathlib import Path

# Project setup
project_root = Path(__file__).parent

def create_app():
    """Egyszerű Flask app létrehozása intelligens fallback-kel"""
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # User study blueprint regisztráció (hibakezeléssel)
    try:
        from user_study.routes import user_study_bp
        app.register_blueprint(user_study_bp)
        print("✅ User study routes registered")
    except ImportError as e:
        print(f"⚠️ User study import failed: {e}")
        # Fallback route-ok regisztrálása
        register_fallback_routes(app)
    
    # Alapvető route-ok
    @app.route('/')
    def index():
        """Főoldal -> tanulmány"""
        try:
            return redirect(url_for('user_study.welcome'))
        except:
            return "<h1>Welcome!</h1><p>App is starting up... <a href='/health'>Check health</a></p>"
    
    @app.route('/health')
    def health():
        """Heroku health check"""
        return jsonify({
            "status": "healthy",
            "service": "sustainable-recipe-recommender",
            "version": "3.0-simplified"
        })
    
    return app

def register_fallback_routes(app):
    """Fallback route-ok ha a user_study modul nem tölthető be"""
    
    @app.route('/welcome')
    @app.route('/user_study/')
    def fallback_welcome():
        return """
        <h1>🚧 System Starting Up</h1>
        <p>The application is initializing. Please try again in a moment.</p>
        <p><a href="/health">Check system health</a></p>
        """
    
    @app.route('/debug/status')
    def fallback_debug():
        return """
        <h2>🔍 Fallback Debug Status</h2>
        <p><strong>Status:</strong> User study module not loaded</p>
        <p><strong>Possible issues:</strong></p>
        <ul>
            <li>Missing dependencies (pandas, sklearn)</li>
            <li>Database initialization pending</li>
            <li>CSV files not processed yet</li>
        </ul>
        <p><strong>Next steps:</strong> Run setup_database.py or check logs</p>
        """

# App inicializálás
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)

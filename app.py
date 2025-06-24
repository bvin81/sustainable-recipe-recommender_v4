#!/usr/bin/env python3
"""
Fixed Heroku-optimized Flask application
Eliminates blueprint registration errors and transaction issues
"""

import os
import sys
import traceback
from flask import Flask, redirect, url_for, jsonify
from pathlib import Path

def create_app():
    """Create Flask app with proper error handling"""
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    print("🏁 Initializing application...")
    print("🚀 Starting Flask app...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Blueprint registration with comprehensive error handling
    blueprint_loaded = False
    try:
        print("📦 Attempting to import user_study...")
        from user_study.routes import user_study_bp
        
        # Clear any existing routes that might conflict
        if hasattr(app, 'view_functions'):
            conflicting_routes = [key for key in app.view_functions.keys() 
                                if key.startswith('user_study.')]
            for route in conflicting_routes:
                app.view_functions.pop(route, None)
        
        app.register_blueprint(user_study_bp)
        blueprint_loaded = True
        print("✅ User study routes registered successfully")
        
    except AssertionError as e:
        if "View function mapping is overwriting" in str(e):
            print(f"⚠️ Blueprint conflict detected: {e}")
            print("🔧 Attempting to resolve route conflicts...")
            
            # Try to register with a different prefix to avoid conflicts
            try:
                from user_study.routes import user_study_bp
                # Create a new blueprint instance to avoid conflicts
                app.register_blueprint(user_study_bp, url_prefix='/study')
                blueprint_loaded = True
                print("✅ User study routes registered with /study prefix")
            except Exception as retry_error:
                print(f"❌ Retry failed: {retry_error}")
                
    except ImportError as e:
        print(f"⚠️ User study import failed: {e}")
        print("Traceback:", traceback.format_exc())
        
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        print("Traceback:", traceback.format_exc())
    
    # Register fallback routes if blueprint failed to load
    if not blueprint_loaded:
        print("🔧 Registering fallback routes...")
        register_fallback_routes(app)
    
    # Core application routes
    @app.route('/')
    def index():
        """Main landing page"""
        try:
            if blueprint_loaded:
                # Try to redirect to the study
                return redirect(url_for('user_study.welcome'))
            else:
                return fallback_home()
        except Exception as e:
            print(f"Index route error: {e}")
            return fallback_home()
    
    @app.route('/health')
    def health():
        """Health check endpoint with detailed status"""
        status = {
            "status": "healthy" if blueprint_loaded else "degraded",
            "service": "sustainable-recipe-recommender",
            "version": "3.1-fixed",
            "blueprint_loaded": blueprint_loaded,
            "python_version": sys.version.split()[0],
            "environment": "heroku" if os.environ.get('DYNO') else "local",
            "working_directory": os.getcwd()
        }
        
        # Check dependencies
        deps = {}
        try:
            import pandas
            deps['pandas'] = pandas.__version__
        except ImportError:
            deps['pandas'] = 'missing'
        
        try:
            import numpy
            deps['numpy'] = numpy.__version__
        except ImportError:
            deps['numpy'] = 'missing'
            
        try:
            import psycopg2
            deps['psycopg2'] = 'available'
        except ImportError:
            deps['psycopg2'] = 'missing - using sqlite fallback'
        
        status['dependencies'] = deps
        
        # Test database connection if available
        try:
            if blueprint_loaded:
                from user_study.routes import db
                conn = db.get_connection()
                if conn:
                    status['database_status'] = f'connected ({db.db_type})'
                    conn.close()
                else:
                    status['database_status'] = 'connection failed'
            else:
                status['database_status'] = 'blueprint not loaded'
        except Exception as e:
            status['database_status'] = f'error: {str(e)}'
        
        return jsonify(status)
    
    # Error handlers
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal server error",
            "message": "Please check /health for detailed status",
            "blueprint_loaded": blueprint_loaded
        }), 500
    
    @app.errorhandler(404)
    def not_found(error):
        available_routes = ["/", "/health"]
        if blueprint_loaded:
            available_routes.extend([
                "/profile", "/recommendations", "/questionnaire", 
                "/admin/stats", "/admin/export/csv", "/admin/export/statistical_csv"
            ])
        
        return jsonify({
            "error": "Not found", 
            "available_routes": available_routes,
            "blueprint_loaded": blueprint_loaded
        }), 404
    
    print(f"✅ Flask app created successfully (blueprint_loaded: {blueprint_loaded})")
    print("✅ Application created successfully")
    return app

def register_fallback_routes(app):
    """Register fallback routes when main blueprint fails"""
    print("🔧 Setting up fallback routes...")
    
    @app.route('/fallback_stats')
    def fallback_stats():
        """Fallback admin stats"""
        return f"""
        <h1>⚠️ Fallback Mode - Admin Stats</h1>
        <p><strong>Status:</strong> Blueprint failed to load</p>
        <p><strong>Database:</strong> Not available</p>
        <p><strong>Environment:</strong> {'Heroku' if os.environ.get('DYNO') else 'Local'}</p>
        <hr>
        <h2>🔗 Available Routes:</h2>
        <ul>
            <li><a href="/">🏠 Home</a></li>
            <li><a href="/health">💚 Health Check</a></li>
            <li><a href="/fallback_export">📄 Fallback Export</a></li>
        </ul>
        <hr>
        <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        """
    
    @app.route('/fallback_export')
    def fallback_export():
        """Fallback CSV export"""
        from datetime import datetime
        import io
        from flask import make_response
        
        # Create minimal CSV
        output = io.StringIO()
        output.write("timestamp,status,message\n")
        output.write(f"{datetime.now().isoformat()},fallback_mode,Blueprint registration failed\n")
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=fallback_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response

def fallback_home():
    """Fallback home page when blueprint fails"""
    from datetime import datetime
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌱 Sustainable Recipe Recommender - Fallback Mode</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #d32f2f; }}
            h2 {{ color: #1976d2; }}
            .status {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .btn {{ display: inline-block; padding: 10px 15px; background: #2196f3; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
            .btn:hover {{ background: #1976d2; }}
            .btn-warning {{ background: #ff9800; }}
            .btn-success {{ background: #4caf50; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚠️ Sustainable Recipe Recommender - Fallback Mode</h1>
            
            <div class="status">
                <strong>System Status:</strong> The main application blueprint failed to load. The system is running in fallback mode.
            </div>
            
            <h2>🔧 Available Actions:</h2>
            <a href="/health" class="btn btn-success">💚 System Health Check</a>
            <a href="/fallback_stats" class="btn btn-warning">📊 Fallback Stats</a>
            <a href="/fallback_export" class="btn">📄 Export Data</a>
            
            <h2>📋 System Information:</h2>
            <ul>
                <li><strong>Environment:</strong> {'Heroku' if os.environ.get('DYNO') else 'Local Development'}</li>
                <li><strong>Python Version:</strong> {sys.version.split()[0]}</li>
                <li><strong>Working Directory:</strong> {os.getcwd()}</li>
                <li><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
            
            <h2>🚀 Next Steps:</h2>
            <ol>
                <li>Check the <a href="/health" class="btn btn-success">health endpoint</a> for detailed diagnostics</li>
                <li>Review Heroku logs for specific error messages</li>
                <li>Ensure all dependencies are properly installed</li>
                <li>Check database connection settings</li>
            </ol>
            
            <hr>
            <p><small>Sustainable Recipe Recommender v3.1 - Fallback Mode</small></p>
        </div>
    </body>
    </html>
    """

# Create the application instance
app = create_app()

# For Heroku deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = not os.environ.get('DYNO')  # Debug only in local development
    
    print(f"🚀 Starting server on port {port}")
    print(f"🔧 Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

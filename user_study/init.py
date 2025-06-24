"""
User Study Module - Enhanced Version with Content-Based Filtering
"""
from .routes import user_study_bp

# Enhanced modules (conditional import)
try:
    from .enhanced_content_based import EnhancedContentBasedRecommender
    from .evaluation_metrics import RecommendationEvaluator
    from .enhanced_routes_integration import EnhancedRecommendationEngine
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

__all__ = [
    'user_study_bp',
    'ENHANCED_AVAILABLE'
]

# Export enhanced classes if available
if ENHANCED_AVAILABLE:
    __all__.extend([
        'EnhancedContentBasedRecommender',
        'RecommendationEvaluator', 
        'EnhancedRecommendationEngine'
    ])

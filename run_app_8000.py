"""
Run Flask app with Waitress on port 8000 (to bypass potential port 5000 issues)
"""
import sys
from waitress import serve
from app import app

if __name__ == '__main__':
    print("=" * 80)
    print("Stock Analysis API - Starting with Waitress on port 8000")
    print("=" * 80)
    print()
    print("✓ Listening on http://0.0.0.0:8000")
    print("✓ Accessible at http://127.0.0.1:8000")
    print()
    print("Available endpoints:")
    print("  GET  http://127.0.0.1:8000/health")
    print("  GET  http://127.0.0.1:8000/status")
    print("  GET  http://127.0.0.1:8000/api/analysis/AAPL")
    print("  GET  http://127.0.0.1:8000/api/summary/AAPL")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    try:
        serve(app, host='0.0.0.0', port=8000, threads=4)
    except KeyboardInterrupt:
        print("\nShutdown requested")
        sys.exit(0)

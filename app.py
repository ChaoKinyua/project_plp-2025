"""
Flask API for serving stock analysis predictions and triggering analysis runs.
Deploy with: gunicorn -w 4 -b 0.0.0.0:5000 app:app
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

import pandas as pd
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
DATA_DIR = Path("data/processed")
LOGS_DIR = Path("logs")

# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "stock-analysis-api"
    }), 200


@app.route('/status', methods=['GET'])
def status():
    """Get pipeline status and last run info"""
    log_file = LOGS_DIR / "project.log"
    
    last_run = None
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-100:]):  # Check last 100 lines
                if "Pipeline" in line or "Processing" in line:
                    last_run = line.strip()
                    break
    
    available_tickers = set()
    for f in DATA_DIR.glob("*_metrics.csv"):
        ticker = f.name.split('_')[0]
        available_tickers.add(ticker)
    
    return jsonify({
        "status": "operational",
        "last_run": last_run or "Never run",
        "available_tickers": sorted(list(available_tickers)),
        "data_directory": str(DATA_DIR),
        "timestamp": datetime.now().isoformat()
    }), 200


# ============================================================================
# Analysis Results Endpoints
# ============================================================================

@app.route('/api/analysis/<ticker>', methods=['GET'])
def get_analysis(ticker: str):
    """
    Get model comparison metrics for a ticker.
    Example: GET /api/analysis/AAPL
    """
    ticker = ticker.upper()
    metrics_file = DATA_DIR / f"{ticker}_metrics.csv"
    
    if not metrics_file.exists():
        return jsonify({
            "error": f"No analysis found for {ticker}",
            "available": sorted([f.name.split('_')[0] for f in DATA_DIR.glob("*_metrics.csv")])
        }), 404
    
    try:
        metrics_df = pd.read_csv(metrics_file)
        return jsonify({
            "ticker": ticker,
            "models": metrics_df.to_dict(orient="records"),
            "best_model": metrics_df.loc[metrics_df['RMSE'].idxmin()].to_dict(),
            "timestamp": datetime.fromtimestamp(metrics_file.stat().st_mtime).isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/summary/<ticker>', methods=['GET'])
def get_summary(ticker: str):
    """
    Get summary statistics for a ticker.
    Example: GET /api/summary/AAPL
    """
    ticker = ticker.upper()
    summary_file = DATA_DIR / f"{ticker}_summary.csv"
    
    if not summary_file.exists():
        return jsonify({"error": f"No summary for {ticker}"}), 404
    
    try:
        summary_df = pd.read_csv(summary_file)
        return jsonify({
            "ticker": ticker,
            "statistics": summary_df.to_dict(orient="records")
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions/<ticker>', methods=['GET'])
def get_predictions(ticker: str):
    """
    Get processed predictions and features.
    Example: GET /api/predictions/AAPL
    """
    ticker = ticker.upper()
    processed_files = sorted(DATA_DIR.glob(f"{ticker}_processed_*.json"), reverse=True)
    
    if not processed_files:
        return jsonify({"error": f"No predictions for {ticker}"}), 404
    
    try:
        latest = processed_files[0]
        with open(latest) as f:
            data = json.load(f)
        return jsonify({
            "ticker": ticker,
            "file": latest.name,
            "metadata": data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Comparison & Analysis Endpoints
# ============================================================================

@app.route('/api/compare', methods=['POST'])
def compare_tickers():
    """
    Compare metrics across multiple tickers.
    POST body: {"tickers": ["AAPL", "MSFT", "GOOGL"]}
    """
    try:
        data = request.get_json() or {}
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({"error": "No tickers provided"}), 400
        
        comparison = {}
        for ticker in tickers:
            metrics_file = DATA_DIR / f"{ticker.upper()}_metrics.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                comparison[ticker] = {
                    "models": len(df),
                    "best_rmse": float(df['RMSE'].min()),
                    "avg_accuracy": float(df['Directional Accuracy'].mean() * 100) if 'Directional Accuracy' in df.columns else None
                }
        
        return jsonify({
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/best-model', methods=['GET'])
def get_best_model():
    """
    Find the best performing model across all tickers.
    Example: GET /api/best-model?metric=RMSE
    """
    metric = request.args.get('metric', 'RMSE')
    best_score = float('inf') if metric == 'RMSE' else float('-inf')
    best_entry = None
    
    for metrics_file in DATA_DIR.glob("*_metrics.csv"):
        try:
            df = pd.read_csv(metrics_file)
            if metric not in df.columns:
                continue
            
            if metric == 'RMSE':
                idx = df[metric].idxmin()
                if df[metric].min() < best_score:
                    best_score = df[metric].min()
                    best_entry = {
                        "ticker": metrics_file.name.split('_')[0],
                        "model": df.iloc[idx]['Model'],
                        "score": float(best_score)
                    }
            else:
                idx = df[metric].idxmax()
                if df[metric].max() > best_score:
                    best_score = df[metric].max()
                    best_entry = {
                        "ticker": metrics_file.name.split('_')[0],
                        "model": df.iloc[idx]['Model'],
                        "score": float(best_score)
                    }
        except Exception as e:
            logger.warning(f"Error processing {metrics_file}: {e}")
            continue
    
    if not best_entry:
        return jsonify({"error": f"No data found for metric: {metric}"}), 404
    
    return jsonify({
        "metric": metric,
        "best": best_entry,
        "timestamp": datetime.now().isoformat()
    }), 200


# ============================================================================
# Control Endpoints
# ============================================================================

@app.route('/api/run-analysis', methods=['POST'])
def trigger_analysis():
    """
    Trigger a manual analysis run.
    POST body: {"force_refresh": false, "tickers": ["AAPL"]}
    
    WARNING: This blocks until analysis completes (15-40 minutes)
    """
    try:
        from main import run_pipeline
        
        logger.info("Manual analysis trigger received")
        run_pipeline()
        
        return jsonify({
            "status": "success",
            "message": "Analysis completed",
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "GET /status",
            "GET /api/analysis/<ticker>",
            "GET /api/summary/<ticker>",
            "GET /api/predictions/<ticker>",
            "POST /api/compare",
            "GET /api/best-model",
            "POST /api/run-analysis"
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting Stock Analysis API")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Development (without debugger to avoid reloader issues)
    logger.info("API listening on http://0.0.0.0:5000")
    logger.info("Available endpoints:")
    logger.info("  GET  /health")
    logger.info("  GET  /status")
    logger.info("  GET  /api/analysis/<ticker>")
    logger.info("  GET  /api/summary/<ticker>")
    logger.info("  GET  /api/predictions/<ticker>")
    logger.info("  POST /api/compare")
    logger.info("  GET  /api/best-model")
    logger.info("  POST /api/run-analysis")
    
    # Removed app.run() - use run_app.py or run_app_8000.py with Waitress instead
    # app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    
    # Production: use gunicorn
    # gunicorn -w 4 -b 0.0.0.0:5000 --timeout 3600 app:app

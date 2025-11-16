"""
Scheduled runner for live deployment.
Runs analysis pipeline on a schedule and logs results.

Usage:
    python scheduled_runner.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import schedule
import time

from main import run_pipeline

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("scheduler")


def run_scheduled_analysis():
    """Execute analysis pipeline with error handling"""
    logger.info("=" * 80)
    logger.info("Starting scheduled analysis run")
    logger.info("=" * 80)
    
    try:
        run_pipeline()
        logger.info("✓ Analysis completed successfully")
    except Exception as e:
        logger.error(f"✗ Analysis failed: {e}", exc_info=True)
        # TODO: Add alert mechanism (email, Slack, etc.)


def schedule_jobs():
    """Setup recurring scheduled jobs"""
    
    # Run analysis daily at 5 PM ET (market close + 1 hour)
    schedule.every().day.at("17:00").do(run_scheduled_analysis)
    logger.info("Scheduled daily analysis at 17:00 ET")
    
    # Optional: Run again at 6 PM for backup
    # schedule.every().day.at("18:00").do(run_scheduled_analysis)
    
    # Optional: Run on market open at 9:30 AM ET for intraday updates
    # schedule.every().weekday.at("09:30").do(run_scheduled_analysis)


def main():
    """Main scheduler loop"""
    logger.info("Stock Analysis Scheduler started")
    logger.info(f"Current time: {datetime.now()}")
    
    schedule_jobs()
    
    logger.info("Scheduler is ready. Waiting for scheduled jobs...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

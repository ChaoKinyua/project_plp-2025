"""
Quick test to verify the pipeline setup is working
"""
import sys
from pathlib import Path

print("=" * 80)
print("DEPENDENCY CHECK")
print("=" * 80)

# Check critical imports
dependencies = {
    "pandas": "data processing",
    "numpy": "numerical computing",
    "tensorflow": "deep learning",
    "keras": "neural networks",
    "sklearn": "machine learning",
    "statsmodels": "time series",
    "yfinance": "data download",
    "plotly": "visualization",
    "flask": "REST API",
    "pyarrow": "parquet files"
}

all_ok = True
for package, purpose in dependencies.items():
    try:
        __import__(package)
        print(f"✓ {package:20} {purpose}")
    except ImportError as e:
        print(f"✗ {package:20} MISSING - {purpose}")
        all_ok = False

print()
print("=" * 80)
print("DATA FILES CHECK")
print("=" * 80)

data_raw = Path("data/raw")
data_processed = Path("data/processed")

print(f"Raw data dir exists: {'✓' if data_raw.exists() else '✗'}")
print(f"Processed data dir exists: {'✓' if data_processed.exists() else '✗'}")

raw_files = list(data_raw.glob("*.parquet")) if data_raw.exists() else []
processed_files = list(data_processed.glob("*")) if data_processed.exists() else []

print(f"Raw files: {len(raw_files)}")
print(f"Processed files: {len(processed_files)}")

if processed_files:
    print("\nProcessed data found:")
    for f in sorted(processed_files)[:5]:
        print(f"  - {f.name}")

print()
print("=" * 80)
print("CONFIG CHECK")
print("=" * 80)

try:
    from config import CONFIG
    print(f"✓ Config loaded")
    print(f"  Tickers: {CONFIG.data.tickers}")
    print(f"  Data period: {CONFIG.data.start_date} to {CONFIG.data.end_date}")
    print(f"  LSTM epochs: {CONFIG.lstm.epochs}")
except Exception as e:
    print(f"✗ Config error: {e}")
    all_ok = False

print()
print("=" * 80)
if all_ok:
    print("✅ ALL CHECKS PASSED - Ready to run pipeline!")
    print("\nRun: python main.py")
else:
    print("❌ SOME CHECKS FAILED")
    print("\nRun: pip install -r requirements.txt")

print("=" * 80)

"""
Simple test script for Flask API endpoints
Run this in a new terminal while app.py is running
"""

import time
import requests
import json

BASE_URL = "http://localhost:5000"

print("=" * 80)
print("FLASK API TEST")
print("=" * 80)
print()

# Give server time to start if just launched
print("Waiting for server to start...")
for i in range(10):
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✓ Server is ready!\n")
            break
    except:
        if i < 9:
            print(f"  Attempt {i+1}/10 - retrying in 1 second...")
            time.sleep(1)
        else:
            print("✗ Server not responding after 10 attempts")
            print("Make sure app.py is running: python app.py")
            exit(1)

# Test endpoints
endpoints = [
    ("GET", "/health", None),
    ("GET", "/status", None),
    ("GET", "/api/analysis/AAPL", None),
    ("GET", "/api/analysis/MSFT", None),
    ("GET", "/api/summary/AAPL", None),
    ("GET", "/api/best-model", None),
]

for method, endpoint, body in endpoints:
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"Testing: {method} {endpoint}")
        
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=body, timeout=5)
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                keys = list(data.keys())[:3]
                print(f"  Response keys: {keys}")
            else:
                print(f"  Response type: {type(data)}")
            print("  ✓ PASS\n")
        else:
            print(f"  ✗ FAIL - {response.status_code}\n")
    
    except Exception as e:
        print(f"  ✗ ERROR - {e}\n")

print("=" * 80)
print("Test complete!")
print("=" * 80)

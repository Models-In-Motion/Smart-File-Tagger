import sys
import os

# Add the app/ folder to Python's path so tests can import main, extractor, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
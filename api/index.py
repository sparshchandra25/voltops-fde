import sys
import os

# Ensure the root directory is in the Python path for local module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


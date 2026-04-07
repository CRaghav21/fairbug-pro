"""
Main entry point for FairBug application
Author: Your Name
Date: 2024
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiments import main

if __name__ == "__main__":
    # Run all experiments
    main()
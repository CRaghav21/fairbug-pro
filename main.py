"""
Main entry point for FairBug application
Author: Raghavendra J Chigarahalli
Date: 5th Apr 2026
"""

import sys
import os

# Adding src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiments import main

if __name__ == "__main__":
    # Running all experiments
    main()
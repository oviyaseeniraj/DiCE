"""
Run All Analysis Scripts

This script runs all the analysis scripts in sequence.
"""

import os
import subprocess
import time

def main():
    """Run all analysis scripts"""
    print("Starting adversarial debiasing analysis...")
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Run train_adversarial_debiasing.py
    print("\n" + "="*50)
    print("Running train_adversarial_debiasing.py...")
    print("="*50)
    subprocess.run(["python", "train_adversarial_debiasing.py"])
    
    # Run evaluate_fairness.py
    print("\n" + "="*50)
    print("Running evaluate_fairness.py...")
    print("="*50)
    subprocess.run(["python", "evaluate_fairness.py"])
    
    # Run compare_performance.py
    print("\n" + "="*50)
    print("Running compare_performance.py...")
    print("="*50)
    subprocess.run(["python", "compare_performance.py"])
    
    # Run analyze_recourse.py
    print("\n" + "="*50)
    print("Running analyze_recourse.py...")
    print("="*50)
    subprocess.run(["python", "analyze_recourse.py"])
    
    print("\n" + "="*50)
    print("All analysis complete!")
    print("Results and plots are available in the 'plots' directory.")
    print("="*50)

if __name__ == "__main__":
    main()

"""
Run Fairness Comparisons

This script runs both the adversarial debiasing comparison and the improved FRL counterfactual generation.
"""

import os
import sys
import subprocess

def main():
    """Main function to run the fairness comparisons"""
    # Create output directories
    os.makedirs('plots/adversarial_debiasing', exist_ok=True)
    os.makedirs('plots/frl_counterfactuals', exist_ok=True)
    
    # Run the adversarial debiasing comparison
    print("Running adversarial debiasing comparison...")
    try:
        subprocess.run([sys.executable, 'compare_adversarial_debiasing.py'], check=True)
        print("Adversarial debiasing comparison completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running adversarial debiasing comparison: {str(e)}")
    
    # Run the improved FRL counterfactual generation
    print("\nRunning improved FRL counterfactual generation...")
    try:
        subprocess.run([sys.executable, 'fix_frl_counterfactuals.py'], check=True)
        print("Improved FRL counterfactual generation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running improved FRL counterfactual generation: {str(e)}")
    
    print("\nAll fairness comparisons have been completed.")
    print("Check the following directories for the results:")
    print("- plots/adversarial_debiasing: F1 score and recall by gender for adversarial debiasing")
    print("- plots/frl_counterfactuals: Improved FRL counterfactual visualizations")

if __name__ == "__main__":
    main()

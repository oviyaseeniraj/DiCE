"""
Run the comparison between Fair Representation Learning (FRL) and standard DiCE

This script runs the comparison implemented in compare_frl_dice.py
"""

import os
import sys
import compare_frl_dice

def main():
    """Main function to run the comparison"""
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Run the comparison
    compare_frl_dice.main()

if __name__ == "__main__":
    main()

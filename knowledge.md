# DiCE Project Knowledge

## Environment Setup

### macOS System Python and urllib3 Compatibility Issue

When running the project on macOS with the system Python (which uses LibreSSL), you may encounter warnings or errors related to urllib3 v2, which only supports OpenSSL 1.1.1+ but the system Python is compiled with LibreSSL.

Error message example:
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
```

Solutions:
1. Pin urllib3 to version 1.26.x in requirements.txt
2. Use a non-system Python installation (e.g., from Homebrew, pyenv, or conda)
3. Create a virtual environment with a non-system Python

For more details, see: https://github.com/urllib3/urllib3/issues/3020

## Project Structure

- `dice_ml/`: Main package directory
  - `constants.py`: Contains constants like backend types and sampling strategies
  - `dice.py`: Main interface for different DiCE implementations
  - `data_interfaces/`: Interfaces for handling data
  - `explainer_interfaces/`: Different explainer implementations
  - `model_interfaces/`: Interfaces for different ML frameworks

## Fair Representation Learning (FRL)

The project includes a Fair Representation Learning (FRL) implementation for generating counterfactual explanations that are fair with respect to protected attributes like gender or race.

Key files:
- `dice_ml/explainer_interfaces/dice_frl.py`: FRL-based DiCE explainer
- `compare_frl_dice.py`: Script to compare FRL with standard DiCE
- `run_frl_comparison.py`: Runner script for the comparison

## Running Comparisons

To run the FRL vs. standard DiCE comparison:
```bash
python run_frl_comparison.py
```

This will generate plots in the `plots/` directory comparing fairness metrics and counterfactual examples.

## Counterfactual Generation

### Troubleshooting Counterfactual Generation

When counterfactuals fail to generate, try these approaches:

1. **Try multiple methods**: Different methods work better for different datasets and models
   - Random: Simple but may not find optimal counterfactuals
   - Genetic: Good for complex decision boundaries but slower
   - KDTree: Fast for finding nearest neighbors but may not work for all models

2. **Vary parameters**:
   - Increase `total_CFs` (e.g., 3, 5, 10)
   - Try different `desired_class` values ("opposite" or specific class index)
   - Adjust weights: `proximity_weight`, `diversity_weight`, `sparsity_weight`

3. **Visualize counterfactuals as HTML**:
   - Use HTML for rich visualizations with highlighting of changes
   - Compare side-by-side to see differences between methods
   - Analyze impact by counting changes per feature

Example code for generating HTML visualizations:
```python
# Save visualization to HTML file
with open('counterfactual.html', 'w') as f:
    f.write('<html><head><title>Counterfactuals</title>')
    f.write('<style>body{font-family:Arial;} .highlight{background-color:#ffffcc;}</style></head><body>')
    f.write('<h1>Counterfactual Examples</h1>')
    f.write(cf_example.test_instance_df.to_html())
    f.write(cf_example.final_cfs_df.to_html())
    f.write('</body></html>')
```

## Error Handling

### Counterfactual Generation Failures

When generating counterfactuals, the process may fail with errors like:
```
No Counterfactuals found for the given configuration, perhaps try with different parameters...
```

This can happen due to:
1. The model's decision boundary being too complex
2. The query instances being too far from the decision boundary
3. Constraints making it difficult to find valid counterfactuals

Solutions:
- Increase the number of counterfactuals to generate (`total_CFs` parameter)
- Try different sampling methods (random, genetic, kdtree)
- Adjust feature ranges or constraints
- Use a simpler model or different dataset

The code should handle these failures gracefully by:
- Catching exceptions from counterfactual generation
- Providing fallback visualizations
- Continuing with other analyses even if counterfactual generation fails

### Index Mismatches in Data Processing

When working with pandas DataFrames from different sources (e.g., train/test splits), be careful with index handling:

1. Don't assume that indices in one DataFrame will match positions in another DataFrame
2. When accessing values across DataFrames, use direct value access (e.g., `df.values`) instead of index-based lookups when possible
3. If you must map between DataFrames with different indices, create explicit mappings rather than assuming index correspondence

Common errors to watch for:
- "single positional indexer is out-of-bounds" - Often indicates an index mismatch when using `.iloc`
- "'NoneType' object has no attribute 'values'" - Check for None values before accessing attributes

### Handling NaN and Infinite Values in Metrics

When calculating and visualizing fairness metrics:

1. Always check for NaN and infinite values before plotting or performing calculations
2. Use `np.isfinite()` to check if values are valid before using them
3. Provide fallback visualizations or informative messages when metrics are invalid
4. Only calculate combined metrics when there are enough valid individual metrics
5. Handle division by zero carefully, especially when calculating improvement percentages

Common errors to watch for:
- "All arrays must be of the same length" - Often indicates that some metrics are missing or invalid
- "posx and posy should be finite values" - Indicates NaN or infinite values in plotting data

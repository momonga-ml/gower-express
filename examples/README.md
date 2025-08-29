# Gower Distance Examples

This directory contains comprehensive examples demonstrating the capabilities of the gower-express package.

## 📓 Available Notebooks

### `bank_marketing_similarity.ipynb` - **Comprehensive Analysis**

A complete tutorial using the UCI Bank Marketing dataset that demonstrates:

- **Data Loading & Exploration**: Understanding the dataset structure and characteristics
- **Basic Gower Distance**: Computing distance matrices and understanding results
- **Similarity Analysis**: Finding k-nearest neighbors and analyzing relationships
- **Practical Applications**:
  - Anomaly detection for quality control
  - Golden standard matching
  - Missing data robustness
- **Advanced Features**:
  - Custom feature weighting
  - Performance benchmarking
  - GPU acceleration (if available)
- **Clustering Integration**: Using Gower distances with clustering algorithms
- **Comparative Analysis**: Gower vs. Euclidean distance performance
- **Real-world Applications**: Business use cases and implementation guidance

## 🚀 Getting Started

### Prerequisites

Install required dependencies:

```bash
# Core requirements
pip install gower ucimlrepo pandas numpy matplotlib seaborn

# Optional for advanced features
pip install plotly scikit-learn umap-learn

# Or install all at once
pip install gower ucimlrepo pandas numpy matplotlib seaborn plotly scikit-learn umap-learn
```

### Running the Examples

1. **Jupyter Notebook**:
   ```bash
   jupyter notebook examples/gower_raisin_similarity.ipynb
   ```

2. **JupyterLab**:
   ```bash
   jupyter lab examples/gower_raisin_similarity.ipynb
   ```

3. **VS Code**: Open the `.ipynb` file directly in VS Code with the Python extension


## 🔧 Customization

The notebook is designed to be easily adapted to your own datasets:

1. Replace the data loading section with your dataset
2. Adjust feature weights based on domain knowledge
3. Modify thresholds for quality control applications
4. Integrate with your existing ML pipelines

## 📚 Further Resources

- [Gower Distance Paper](https://www.jstor.org/stable/2528823) - Original 1971 publication
- [UCI ML Repository](https://archive.ics.uci.edu/ml) - More datasets to try
- [Package Documentation](../README.md) - Full gower-express documentation

## 🤝 Contributing

Have ideas for additional examples? Please:
1. Fork the repository
2. Create your example notebook
3. Submit a pull request
4. Include documentation and test data

## 📄 License

These examples are provided under the same license as the main package. See the root LICENSE file for details.

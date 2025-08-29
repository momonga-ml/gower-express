# Gower Distance Examples

This directory contains comprehensive examples demonstrating the capabilities of the gower-express package.

## 📓 Available Notebooks

### `bank_marketing_similarity.ipynb` - **Comprehensive Analysis**

A complete tutorial using the UCI Bank Marketing dataset (45,211 clients) that demonstrates:

- **Data Loading & Exploration**: Mixed categorical/numerical features (9 categorical, 7 numerical)
- **Basic Gower Distance**: Computing distance matrices with automatic feature detection
- **Customer Similarity Analysis**: Finding similar bank clients for targeted marketing
- **Practical Applications**:
  - Customer anomaly detection and risk assessment
  - Golden standard client profiling
  - Missing data robustness (handles incomplete records)
- **Advanced Features**:
  - Custom feature weighting (demographics, financial, contact-focused)
  - Performance benchmarking (~1.6M calculations/second)
  - Memory-efficient computation
- **Clustering Integration**: Customer segmentation with precomputed distances
- **Comparative Analysis**: Gower vs. Euclidean distance for mixed-type banking data
- **Real-world Applications**: Marketing, risk assessment, customer segmentation

**Key Results**: Superior k-NN accuracy, efficient processing of large datasets, robust handling of missing values, and easy integration with sklearn workflows.

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
   jupyter notebook examples/bank_marketing_similarity.ipynb
   ```

2. **JupyterLab**:
   ```bash
   jupyter lab examples/bank_marketing_similarity.ipynb
   ```

3. **Quick Validation** (test without running full notebook):
   ```bash
   python examples/validate_notebook.py
   ```

4. **VS Code**: Open the `.ipynb` file directly in VS Code with the Python extension


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

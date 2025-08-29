# Gower Distance Examples

This directory contains comprehensive examples demonstrating the capabilities of the gower-express package.

## 📓 Available Notebooks

### `gower_raisin_similarity.ipynb` - **Comprehensive Agricultural Data Analysis**

A complete tutorial using the UCI Raisin dataset that demonstrates:

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

## 📊 Dataset Information

The notebook uses the **UCI Raisin Dataset**:
- **900 samples** of raisin grains from Turkey
- **7 morphological features** extracted from computer vision
- **2 varieties**: Kecimen and Besni raisins
- **Perfect for demonstrating**: Quality control, similarity analysis, and classification

## 🎯 Learning Objectives

After working through the notebook, you'll understand:

1. **When to use Gower distance** vs. other distance metrics
2. **How to handle mixed-type data** (numerical + categorical)
3. **Practical applications** in agriculture and food industry
4. **Performance considerations** and optimization techniques
5. **Integration with machine learning** workflows
6. **Real-world implementation** strategies

## 💡 Use Cases Demonstrated

- **Quality Control**: Automated sorting and grading
- **Anomaly Detection**: Identifying unusual or damaged products
- **Similarity Search**: Finding products with similar characteristics
- **Clustering**: Unsupervised product categorization
- **Missing Data Handling**: Robust analysis without preprocessing

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

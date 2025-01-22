# Iris Flower Classification Project

This project focuses on classifying Iris flower species based on their sepal and petal dimensions using machine learning algorithms. The dataset used is the well-known Iris dataset, which consists of measurements for three species of Iris: *setosa*, *versicolor*, and *virginica*.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview
The main objectives of this project are:
1. Perform exploratory data analysis (EDA) on the Iris dataset to uncover insights.
2. Visualize feature correlations and distributions.
3. Implement machine learning models like K-Nearest Neighbors (KNN) and Decision Tree classifiers to predict Iris flower species.
4. Evaluate model performance using accuracy and other metrics.

## Dataset Description
The Iris dataset contains 150 samples with the following attributes:
- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**
- **Species**: Target variable (*setosa*, *versicolor*, *virginica*)

## Exploratory Data Analysis
- Pair plots were created to visualize feature distributions and relationships.
- A correlation heatmap was generated to highlight feature interdependencies.

## Machine Learning Models
1. **K-Nearest Neighbors (KNN)**:
   - Used to classify Iris species based on the nearest data points.
   - Achieved competitive accuracy with optimal parameter tuning.

2. **Decision Tree Classifier**:
   - A tree-based model that splits data based on feature thresholds.
   - Visualized the decision tree for interpretability.

## Dependencies
The following Python libraries are required:
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Install the dependencies using:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/iris-classification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd iris-classification
   ```

3. Run the Python script:
   ```bash
   python main.py
   ```

4. View the visualizations and model evaluation results in the output.

## Results
- **KNN Classifier**:
  - Accuracy: ~96%
  - Provided a comprehensive classification report.

- **Decision Tree Classifier**:
  - Accuracy: ~96%
  - Visualized the tree for better understanding of decision boundaries.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


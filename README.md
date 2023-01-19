
# Best Split Dataset Prediciton

![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/dlamotto/decision-tree-split-prediciton)

This project is a Python script that uses decision tree algorithms to predict the best split for a given test dataset. 



## Features

- Uses the CART, Gini index, and Information gain classifications to analyze the training data
- Predicts the optimal split for the test data using the decision tree algorithm


## Installation

Install best_split_decision_tree with pip

```bash
  pip install best_split_decision_tree
```
    
## Examples

```python
import best_split_decision_tree 

training_set = load('training_data.txt')

test_set = load('test_data.txt')

# returns a list of predicted classifiers for observations in test data using Information Gain classification
best_split_decision_tree.classifyIG(training_set, test_set)

# returns a list of predicted classifiers for observations in test data using Gini Index classification
best_split_decision_tree.classifyG(training_set, test_set)

# returns a list of predicted classifiers for observations in test data using CART classification
best_split_decision_tree.classifyCART(training_set, test_set)

```


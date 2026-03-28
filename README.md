# Book Recommendation System

## Overview
This project implements an **item-based collaborative filtering recommender system** for books using Python.  
It predicts user preferences and suggests books based on similar users’ ratings, and is evaluated using **Precision@K** and **Recall@K**.

---

## Features
- **Preprocessing**
  - Load and clean book, user, and rating data
  - Handle missing values and filter active users
- **Data Preparation**
  - Train/test split for evaluation
  - Construct user-book pivot matrix
  - Convert to sparse matrix for memory efficiency
- **Recommendation Model**
  - Item-based K-Nearest Neighbors (KNN) using user ratings
  - Compute similarity between books
  - Recommend top-N books based on a seed book
- **Evaluation**
  - Compute Precision@K and Recall@K per user
  - Handles sparse datasets and thresholds for liked books

---

## Technologies Used
- Python
- Pandas, NumPy — data manipulation
- Scikit-learn — KNN modeling
- SciPy — sparse matrix computation

---

## Sample Usage
```python
# Convert pivot table to sparse matrix
book_sparse = csr_matrix(book_pivot)

# Recommend books similar to a seed book
seed_book = "1984"
recommendations = recommend_books(seed_book, n=5)
print(recommendations)

# Evaluate model
precision, recall = precision_recall_at_k(k=5, threshold=8)
print("Precision@5:", precision)
print("Recall@5:", recall)

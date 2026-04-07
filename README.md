# Book Recommendation System

## Overview
This project implements a **hybrid book recommender system** using Python.  
It combines **popularity-based recommendations**, **item-based collaborative filtering (KNN)**, and **matrix factorization (SVD)** to suggest books to users.  
The system predicts user preferences and is evaluated using **Precision@K** and **Recall@K**.

---

## Features
- **Preprocessing**
  - Load and clean book, user, and rating data
  - Handle missing values and filter active users
- **Data Preparation**
  - Train/test split for evaluation
  - Construct user-book pivot matrix
  - Convert to sparse matrix for memory efficiency
- **Recommendation Models**
  - **Popularity-Based**: Recommend most popular books
  - **Item-Based KNN**: Compute similarity between books
  - **SVD (Matrix Factorization)**: Predict ratings using latent features
  - **Hybrid**: Combine multiple models for improved recommendations
- **Evaluation**
  - Compute **Precision@K** and **Recall@K** per user
  - Handles sparse datasets and thresholds for liked books

---

## Technologies Used
- Python  
- Pandas, NumPy — data manipulation  
- Scikit-learn — KNN modeling  
- Surprise — SVD matrix factorization  
- SciPy — sparse matrix computation  

---

## Sample Usage
```python
# Convert pivot table to sparse matrix
book_sparse = csr_matrix(book_pivot)

# Recommend books using different models
seed_book = "1984"
print("KNN Recommendations:", recommend_books_knn(seed_book=seed_book, n_suggestions=5, ))
print("SVD Recommendations:", recommend_svd(user_id="276725", n_suggestions=5))
print("Popularity Recommendations:", recommend_books_popular(n_suggestions=5))
print("Hybrid Recommendations:", rec_books = hybrid_recommend(user_id="276725", n_suggestions=5))
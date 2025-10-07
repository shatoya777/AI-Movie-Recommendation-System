# AI TMDB Movie Recommendation System

Built a personalized movie recommendation system that filters based on user input (genre, year, tone) using Natural Language Processing and clustering. Applied sentiment analysis and machine learning to match users with movies that fit their emotional preferences. Developed during the AI4ALL Ignite program, this project demonstrates applied skills in data cleaning, vectorization, clustering, and classification.

View Project: https://colab.research.google.com/drive/11RvBgsKIU1AAcUD7O9c4jBxrXL02O0Zv#scrollTo=KIeG-m2sFIB0

## Problem Statement <!--- do not change this line -->

With the rise of streaming platforms, users face overwhelming choices and decision fatigue when selecting a movie. Recommender systems often rely on general popularity or ratings, overlooking a user’s mood or tone preference. This project aims to address that gap by integrating sentiment analysis into a content-based filtering system to recommend movies that match a user’s desired emotional tone. This innovation enhances recommendation relevance and personalization.

## Key Results <!--- do not change this line -->

1. Implemented a tone-sensitive recommendation system using the TMDb 5000 Movie Dataset.
2. Vectorized movie overviews using TF-IDF and clustered them using KMeans (with optimal k via silhouette scoring).
3. Classified movie "tone" (positive/neutral/negative) using VADER sentiment analysis and evaluated using ROC-AUC (score: 0.598).
4. Enabled user input for genre, release year, and tone to generate personalized movie recommendations.
5. Visualized sentiment distribution, word clouds, and 3D PCA clusters to interpret patterns in movie data.


## Methodologies <!--- do not change this line -->

To accomplish this we: 
Cleaned and preprocessed movie metadata and text overview data from the TMDb dataset. Extracted tone using VADER sentiment analysis (compound polarity scoring). Applied TF-IDF vectorization to transform movie descriptions into numerical features. Used KMeans clustering and Principal Component Analysis (PCA) for dimensionality reduction and visualization. Classified tone using human-labeled training data and evaluated predictions using ROC-AUC. Visualized genre-tone distributions and sentiment maps using Matplotlib and Plotly.


## Data Sources <!--- do not change this line -->

TMDb 5000 Movie Dataset from Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Technologies Used <!--- do not change this line -->

- Python
- Pandas
- scikit-learn
- VADER Sentiment (NLTK)
- TF-IDF Vectorizer
- KMeans Clustering
- PCA
- Matplotlib
- Plotly
- WordCloud
- Jupyter Notebooks


## Authors <!--- do not change this line -->

This project was completed in collaboration with:

Shatoya Gardner (shatoyagg.6@gmail.com)
Simon Plotkin (simplot75@gmail.com)
Nathan Seife (nathanseife3@gmail.com)
Syeda Ayesha Busha (bushra.2@wright.edu)
AI4ALL Ignite Program (Summer 2025)

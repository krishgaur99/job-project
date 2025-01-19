# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import GridSearchCV

# Step 1: Load datasets
# movies.csv - columns: movieId, title, genres
# ratings.csv - columns: userId, movieId, rating, timestamp
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preview datasets
print(movies.head())
print(ratings.head())

# Step 2: Content-Based Filtering (Similarity Based on Genres)

# Step 2.1: Use TF-IDF Vectorizer to compute movie similarities based on genres
tfidf = TfidfVectorizer(stop_words='english')

# Fill missing genres with empty string
movies['genres'] = movies['genres'].fillna('')

# Compute the TF-IDF matrix for genres
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function to recommend movies based on content similarity
def recommend_movies_content_based(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get the pairwise similarity scores of all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Exclude itself
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

# Test content-based recommendation
print("Content-based recommendations for 'The Godfather':")
print(recommend_movies_content_based('The Godfather'))


# Step 3: Collaborative Filtering using Matrix Factorization (SVD)

# Step 3.1: Prepare the dataset for surprise (Collaborative Filtering)
reader = Reader()

# Load the ratings dataset into Surprise dataset format
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Step 3.2: Build and train the SVD model
svd = SVD()

# Train the model on the trainset
svd.fit(trainset)

# Test the model on the testset
predictions = svd.test(testset)

# Compute RMSE (Root Mean Squared Error)
accuracy.rmse(predictions)

# Step 3.3: Collaborative Filtering - Predict User Ratings for Movies
def recommend_movies_collaborative(userId, num_recommendations=10):
    # Get all movieIds
    movie_ids = ratings['movieId'].unique()
    
    # Predict ratings for all movies the user hasn't rated
    user_ratings = {movie_id: svd.predict(userId, movie_id).est for movie_id in movie_ids}
    
    # Sort the movies by predicted ratings
    sorted_ratings = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top 'num_recommendations' movieIds
    top_movie_ids = [x[0] for x in sorted_ratings[:num_recommendations]]
    
    # Return the movie titles of the recommended movies
    return movies[movies['movieId'].isin(top_movie_ids)]['title']

# Test collaborative filtering recommendation
print("Collaborative filtering recommendations for user 1:")
print(recommend_movies_collaborative(1))


# Step 4: Hybrid Recommendation System (Combine both methods)
def hybrid_recommendation(userId, movie_title, num_recommendations=10):
    # Get content-based recommendations
    content_based_recommendations = recommend_movies_content_based(movie_title)
    
    # Get collaborative filtering recommendations
    collaborative_recommendations = recommend_movies_collaborative(userId, num_recommendations)
    
    # Combine both recommendations and remove duplicates
    hybrid_recommendations = pd.concat([content_based_recommendations, collaborative_recommendations]).drop_duplicates().head(num_recommendations)
    
    return hybrid_recommendations

# Test hybrid recommendation system
print("Hybrid recommendations for user 1 with 'The Godfather':")
print(hybrid_recommendation(1, 'The Godfather'))

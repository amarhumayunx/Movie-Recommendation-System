!pip install scikit-surprise
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load MovieLens dataset (u.data from Colab's sample_data folder)
url = "/content/sample_data/u.data"
columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=columns, usecols=[0, 1, 2], encoding='latin-1')

# Load movie titles from a sample online source
movies_url = "/content/sample_data/u.item"
movies_columns = ['movie_id', 'movie_title']
movies = pd.read_csv(movies_url, sep='|', names=movies_columns, usecols=[0, 1], encoding='latin-1')

# Merge movies with ratings
df = pd.merge(df, movies, on='movie_id')

# Basic Data Exploration
print(f"Data Shape: {df.shape}")
print(f"First few rows of the dataset:\n{df.head()}")

# Visualize distribution of ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=df, palette="viridis")
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Prepare the data for Surprise library (Collaborative Filtering)
reader = Reader(rating_scale=(1, 5))  # Rating scale (1 to 5)
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

# Split the dataset into training and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD (Singular Value Decomposition) model for collaborative filtering
svd = SVD()

# Train the model on the training data
svd.fit(trainset)

# Evaluate the model on the test set
predictions = svd.test(testset)

# Calculate RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"RMSE of the model: {rmse:.4f}")

# Function to recommend top N movies for a user
def recommend_movies(user_id, n=10):
    # Get all movie ids
    all_movie_ids = df['movie_id'].unique()
    
    # Predict ratings for each movie for the user
    predictions = [svd.predict(user_id, movie_id) for movie_id in all_movie_ids]
    
    # Sort predictions by predicted rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top N movie recommendations
    top_n_recommendations = predictions[:n]
    recommended_movie_ids = [prediction.iid for prediction in top_n_recommendations]
    
    # Get movie titles corresponding to the recommended movie ids
    recommended_movies = movies[movies['movie_id'].isin(recommended_movie_ids)]
    
    return recommended_movies

# Example: Recommend top 10 movies for a specific user (user_id = 1)
user_id = 1
recommended_movies = recommend_movies(user_id, n=10)

print(f"\nTop 10 recommended movies for User {user_id}:")
print(recommended_movies)

# Visualize the top 10 recommended movies
plt.figure(figsize=(10, 6))
sns.barplot(x='movie_title', y='movie_id', data=recommended_movies, palette="coolwarm")
plt.title(f"Top 10 Recommended Movies for User {user_id}")
plt.xlabel("Movie Title")
plt.ylabel("Movie ID")
plt.xticks(rotation=90)
plt.show()
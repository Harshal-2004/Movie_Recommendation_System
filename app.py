from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import requests
from flask_cors import CORS
from fuzzywuzzy import process
from datetime import datetime
import random
import numpy as np
from functools import lru_cache
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the movies and similarity matrix
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Get all movie titles for fuzzy matching
all_titles = movies['title'].tolist()

# TMDB API configuration
TMDB_API_KEY = '3fd2be6f0c70a2a598f084ddfb75487c'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMG_URL = 'https://image.tmdb.org/t/p/w500'

# Cache for movie details
movie_details_cache = {}
CACHE_DURATION = 3600  # Cache duration in seconds (1 hour)

@lru_cache(maxsize=1000)
def get_movie_recommendations(movie_id, n_recommendations=8):
    try:
        idx = movies[movies['id'] == movie_id].index[0]
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = movies.iloc[movie_indices][['id', 'title']].to_dict('records')
        return recommended_movies
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

def get_mood_based_recommendations(mood):
    mood_genres = {
        'chill': ['Comedy', 'Animation', 'Family'],
        'intense': ['Action', 'Thriller', 'Horror'],
        'romantic': ['Romance', 'Drama'],
        'thoughtful': ['Drama', 'Documentary'],
        'adventurous': ['Adventure', 'Action', 'Fantasy']
    }
    return mood_genres.get(mood.lower(), [])

def get_time_based_recommendations():
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12:
        return ['Family', 'Comedy', 'Animation']  # Morning
    elif 12 <= current_hour < 18:
        return ['Action', 'Adventure', 'Thriller']  # Afternoon
    else:
        return ['Drama', 'Romance', 'Horror']  # Evening/Night

def fetch_poster(movie_id):
    current_time = time.time()
    
    # Check if movie details are in cache and not expired
    if movie_id in movie_details_cache:
        cached_data = movie_details_cache[movie_id]
        if current_time - cached_data['timestamp'] < CACHE_DURATION:
            return cached_data['data']
    
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            movie_data = {
                'poster': f"{TMDB_IMG_URL}{data.get('poster_path', '')}",
                'overview': data.get('overview', 'No overview available'),
                'vote_average': data.get('vote_average', 0),
                'genres': [genre['name'] for genre in data.get('genres', [])],
                'release_date': data.get('release_date', ''),
                'runtime': data.get('runtime', 0)
            }
            
            # Cache the movie data
            movie_details_cache[movie_id] = {
                'data': movie_data,
                'timestamp': current_time
            }
            
            return movie_data
    except Exception as e:
        print(f"Error fetching movie details: {e}")
    
        return {
            'poster': '',
            'overview': 'No overview available',
            'vote_average': 0,
            'genres': [],
            'release_date': '',
            'runtime': 0
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/categories')
def categories():
    # Get all unique genres from movies
    all_genres = []
    for movie_genres in movies['genres']:
        if isinstance(movie_genres, list):
            all_genres.extend(movie_genres)
    all_genres = sorted(list(set(all_genres)))  # Get unique genres and sort them
    
    # Get movies for each genre with posters (using cache)
    genre_movies = {}
    for genre in all_genres:
        # Get movies in this genre
        genre_movies_df = movies[movies['genres'].apply(lambda x: genre in x if isinstance(x, list) else False)]
        
        # Sort by vote average and get top movies
        top_movies = genre_movies_df.nlargest(8, 'vote_average')
        
        # Get movie details with posters
        movies_with_posters = []
        for _, movie in top_movies.iterrows():
            movie_data = fetch_poster(movie['id'])
            if movie_data and movie_data['poster']:  # Only include movies with posters
                movies_with_posters.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'vote_average': movie_data['vote_average'],
                    'poster': movie_data['poster'],
                    'overview': movie_data['overview']
                })
            if len(movies_with_posters) >= 4:  # Only keep top 4 movies with posters
                break
        
        if movies_with_posters:  # Only include genres that have movies with posters
            genre_movies[genre] = movies_with_posters
    
    return render_template('categories.html', genres=list(genre_movies.keys()), genre_movies=genre_movies)

@app.route('/category/<genre>')
def category_movies(genre):
    # Get all movies in this genre
    genre_movies_df = movies[movies['genres'].apply(lambda x: genre in x if isinstance(x, list) else False)]
    
    # Sort by vote average and get top movies
    genre_movies_df = genre_movies_df.nlargest(20, 'vote_average')
    
    # Get movie details with posters
    movies_list = []
    for _, movie in genre_movies_df.iterrows():
        movie_data = fetch_poster(movie['id'])
        if movie_data and movie_data['poster']:  # Only include movies with posters
            movies_list.append({
                'id': movie['id'],
                'title': movie['title'],
                'vote_average': movie_data['vote_average'],
                'poster': movie_data['poster'],
                'overview': movie_data['overview']
            })
    
    return render_template('category_movies.html', genre=genre, movies=movies_list)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if query:
        # Use fuzzy matching to find similar titles (limit to 5 for better performance)
        matches = process.extract(query, all_titles, limit=5)
        movie_ids = [movies[movies['title'] == match[0]]['id'].iloc[0] for match in matches]
        
        # Get search results with posters (using cache)
        results = []
        recommendations = []
        
        for movie_id in movie_ids:
            movie_data = fetch_poster(movie_id)
            if movie_data['poster']:
                results.append({
                    'id': movie_id,
                    'title': movies[movies['id'] == movie_id]['title'].iloc[0],
                    **movie_data
                })
                
                # Get recommendations for the first matching movie (increased to 8)
                if len(recommendations) == 0:
                    rec_movies = get_movie_recommendations(movie_id, n_recommendations=8)
                    for rec_movie in rec_movies:
                        rec_data = fetch_poster(rec_movie['id'])
                        if rec_data['poster']:
                            recommendations.append({
                                'id': rec_movie['id'],
                                'title': rec_movie['title'],
                                **rec_data
                            })
        
        return render_template('search_results.html', 
                             results=results, 
                             recommendations=recommendations, 
                             query=query)
    return render_template('search.html')

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    # Get movie details with caching
    movie_data = fetch_poster(movie_id)
    movie_title = movies[movies['id'] == movie_id]['title'].iloc[0]
    
    # Get similar movies with caching (increased to 8 recommendations)
    similar_movies = []
    rec_movies = get_movie_recommendations(movie_id, n_recommendations=8)
    for rec_movie in rec_movies:
        rec_data = fetch_poster(rec_movie['id'])
        if rec_data['poster']:
            similar_movies.append({
                'id': rec_movie['id'],
                'title': rec_movie['title'],
                **rec_data
            })
    
    return render_template('movie_details.html', 
                         movie={'id': movie_id, 'title': movie_title, **movie_data},
                         similar_movies=similar_movies)

if __name__ == '__main__':
    app.run(debug=True)

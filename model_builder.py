import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pickle
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import ast
from tqdm import tqdm
import time

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# TMDB API configuration
TMDB_API_KEY = '3fd2be6f0c70a2a598f084ddfb75487c'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_list_string(text):
    try:
        return [item['name'] for item in ast.literal_eval(text)]
    except:
        return []

def parse_cast(text):
    try:
        return [item['name'] for item in ast.literal_eval(text)[:5]]
    except:
        return []

def parse_crew(text):
    try:
        return [item['name'] for item in ast.literal_eval(text) if item['job'] == 'Director']
    except:
        return []

def get_language_from_title(title):
    hindi_chars = ['ा', 'ी', 'ू', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', 'ँ']
    if any(char in str(title) for char in hindi_chars):
        return 'hi'
    return 'en'

def create_tags(row):
    try:
        tags = []
        
        # Add language tag with high weight
        if 'language' in row:
            tags.extend([row['language']] * 5)
        
        # Add overview words with high weight
        if isinstance(row['overview'], str) and row['overview'].strip():
            overview_words = row['overview'].split()
            tags.extend(overview_words * 3)
        
        # Add keywords with medium weight
        if isinstance(row['keywords'], list):
            tags.extend(row['keywords'] * 2)
        
        # Add genres with medium weight
        if isinstance(row['genres'], list):
            tags.extend(row['genres'] * 2)
        
        # Add cast and crew with lower weight
        if isinstance(row['cast'], list):
            tags.extend(row['cast'])
        if isinstance(row['crew'], list):
            tags.extend(row['crew'])
        
        # Clean and normalize tags
        cleaned_tags = []
        for tag in tags:
            if isinstance(tag, str):
                words = ''.join(' ' + c if c.isupper() else c for c in tag).strip()
                cleaned_tags.extend(words.lower().split())
        
        return ' '.join(cleaned_tags)
    except Exception as e:
        print(f"Error processing row: {e}")
        return ""

def fetch_hindi_movies():
    print("Fetching Hindi movies...")
    movies_data = []
    
    # Fetch more Hindi movies to ensure 40% of total dataset
    for page in range(1, 11):  # Increased to 10 pages
        url = f"{TMDB_BASE_URL}/discover/movie?api_key={TMDB_API_KEY}&with_original_language=hi&sort_by=popularity.desc&page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for movie in data['results']:
                movie_url = f"{TMDB_BASE_URL}/movie/{movie['id']}?api_key={TMDB_API_KEY}&append_to_response=credits,keywords"
                movie_response = requests.get(movie_url)
                if movie_response.status_code == 200:
                    movie_data = movie_response.json()
                    movies_data.append({
                        'id': movie_data['id'],
                        'title': movie_data['title'],
                        'overview': movie_data['overview'],
                        'genres': [genre['name'] for genre in movie_data.get('genres', [])],
                        'keywords': [kw['name'] for kw in movie_data.get('keywords', {}).get('keywords', [])],
                        'cast': [cast['name'] for cast in movie_data.get('credits', {}).get('cast', [])[:5]],
                        'crew': [crew['name'] for crew in movie_data.get('credits', {}).get('crew', []) if crew['job'] == 'Director'],
                        'language': 'hi',
                        'popularity': movie_data.get('popularity', 0),
                        'vote_average': movie_data.get('vote_average', 0),
                        'vote_count': movie_data.get('vote_count', 0)
                    })
        time.sleep(1)  # Add delay to avoid rate limiting
    
    return pd.DataFrame(movies_data)

def build_model():
    print("Loading existing dataset...")
    # Load existing TMDB dataset
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    
    print("Processing existing dataset...")
    # Process existing dataset
    movies = movies.merge(credits, on='title')
    movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Parse the string columns
    print("Parsing string columns...")
    movies['genres'] = movies['genres'].apply(parse_list_string)
    movies['keywords'] = movies['keywords'].apply(parse_list_string)
    movies['cast'] = movies['cast'].apply(parse_cast)
    movies['crew'] = movies['crew'].apply(parse_crew)
    
    # Add language information
    print("Adding language information...")
    movies['language'] = movies['title'].apply(get_language_from_title)
    
    # Fetch Hindi movies
    print("Fetching Hindi movies...")
    hindi_movies = fetch_hindi_movies()
    
    # Combine datasets
    print("Combining datasets...")
    combined_movies = pd.concat([movies, hindi_movies], ignore_index=True)
    
    # Remove duplicates
    print("Removing duplicates...")
    combined_movies = combined_movies.drop_duplicates(subset=['title', 'language'])
    
    # Ensure 60-40 split
    print("Balancing dataset...")
    english_movies = combined_movies[combined_movies['language'] == 'en']
    hindi_movies = combined_movies[combined_movies['language'] == 'hi']
    
    # Calculate target sizes
    total_movies = len(combined_movies)
    target_hindi = int(total_movies * 0.4)
    target_english = total_movies - target_hindi
    
    # Sample to achieve desired ratio
    if len(hindi_movies) > target_hindi:
        hindi_movies = hindi_movies.sample(n=target_hindi, random_state=42)
    if len(english_movies) > target_english:
        english_movies = english_movies.sample(n=target_english, random_state=42)
    
    # Recombine datasets
    combined_movies = pd.concat([english_movies, hindi_movies], ignore_index=True)
    
    # Create tags in batches
    print("Creating content-based tags...")
    batch_size = 100
    total_rows = len(combined_movies)
    tags = []
    
    for i in tqdm(range(0, total_rows, batch_size)):
        batch = combined_movies.iloc[i:i + batch_size]
        batch_tags = batch.apply(create_tags, axis=1)
        tags.extend(batch_tags)
    
    combined_movies['tags'] = tags
    
    # Create TF-IDF vectors
    print("Creating TF-IDF vectors...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    vectors = tfidf.fit_transform(combined_movies['tags'].fillna(''))
    
    # Calculate similarity
    print("Calculating similarity...")
    similarity = cosine_similarity(vectors)
    
    # Save the processed data
    print("Saving model...")
    pickle.dump(combined_movies, open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    
    # Print dataset statistics
    print("\nModel building completed successfully!")
    print(f"Total movies in dataset: {len(combined_movies)}")
    print(f"English movies: {len(english_movies)} ({len(english_movies)/len(combined_movies)*100:.1f}%)")
    print(f"Hindi movies: {len(hindi_movies)} ({len(hindi_movies)/len(combined_movies)*100:.1f}%)")
    
    # Test recommendation accuracy
    test_recommendations(combined_movies, similarity)

def test_recommendations(movies, similarity):
    print("\nTesting recommendation accuracy...")
    
    # Test English movies
    english_movies = movies[movies['language'] == 'en'].sample(n=5, random_state=42)
    print("\nTesting English movie recommendations:")
    for _, movie in english_movies.iterrows():
        index = movie.name
        distances = similarity[index]
        movie_list = list(enumerate(distances))
        similar_movies = sorted(movie_list, key=lambda x: x[1], reverse=True)[1:6]
        
        # Count language matches
        english_count = 0
        hindi_count = 0
        for i, _ in similar_movies:
            if movies.iloc[i]['language'] == 'en':
                english_count += 1
            else:
                hindi_count += 1
        
        print(f"\nMovie: {movie['title']}")
        print(f"English recommendations: {english_count}/5")
        print(f"Hindi recommendations: {hindi_count}/5")
    
    # Test Hindi movies
    hindi_movies = movies[movies['language'] == 'hi'].sample(n=5, random_state=42)
    print("\nTesting Hindi movie recommendations:")
    for _, movie in hindi_movies.iterrows():
        index = movie.name
        distances = similarity[index]
        movie_list = list(enumerate(distances))
        similar_movies = sorted(movie_list, key=lambda x: x[1], reverse=True)[1:6]
        
        # Count language matches
        english_count = 0
        hindi_count = 0
        for i, _ in similar_movies:
            if movies.iloc[i]['language'] == 'en':
                english_count += 1
            else:
                hindi_count += 1
        
        print(f"\nMovie: {movie['title']}")
        print(f"English recommendations: {english_count}/5")
        print(f"Hindi recommendations: {hindi_count}/5")

if __name__ == '__main__':
    build_model()

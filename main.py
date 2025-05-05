import zipfile
import os

# Define folder structure and files to include
base_dir = "/mnt/data/movie_recommender_app"
os.makedirs(base_dir, exist_ok=True)

# Backend files (Flask)
app_py = """
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
movies['overview'] = movies['overview'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title):
    try:
        index = movies[movies['title'].str.lower() == title.lower()].index[0]
        distances = list(enumerate(similarity[index]))
        distances = sorted(distances, key=lambda x: x[1], reverse=True)
        recommended = [movies.iloc[i[0]].title for i in distances[1:6]]
        return recommended
    except IndexError:
        return []

@app.route("/recommend", methods=["GET"])
def recommend_movies():
    movie_name = request.args.get("movie")
    recommended = recommend(movie_name)
    return jsonify(recommended)

if __name__ == "__main__":
    app.run(debug=True)
"""

# Frontend files (HTML, CSS, JS)
index_html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="style.css" />
    <title>Movie App</title>
  </head>
  <body>
    <header>
      <form id="form">
        <input type="text" id="search" class="search" placeholder="Search">
      </form>
    </header>
    <main id="main"></main>
    <script src="script.js"></script>
  </body>
</html>
"""

style_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@200;400&display=swap');

:root {
  --primary-color: #22254b;
  --secondary-color: #373b69;
}

* {
  box-sizing: border-box;
}

body {
  background-color: var(--primary-color);
  font-family: 'Poppins', sans-serif;
  margin: 0;
}

header {
  padding: 1rem;
  display: flex;
  justify-content: flex-end;
  background-color: var(--secondary-color);
}

.search {
  background-color: transparent;
  border: 2px solid var(--primary-color);
  border-radius: 50px;
  font-family: inherit;
  font-size: 1rem;
  padding: 0.5rem 1rem;
  color: #fff;
}

.search::placeholder {
  color: #7378c5;
}

.search:focus {
  outline: none;
  background-color: var(--primary-color);
}

main {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
}

.movie {
  width: 300px;
  margin: 1rem;
  background-color: var(--secondary-color);
  box-shadow: 0 4px 5px rgba(0, 0, 0, 0.2);
  position: relative;
  overflow: hidden;
  border-radius: 3px;
}

.movie img {
  width: 100%;
}

.movie-info {
  color: #eee;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap:0.2rem;
  padding: 0.5rem 1rem 1rem;
  letter-spacing: 0.5px;
}

.movie-info h3 {
  margin-top: 0;
}

.movie-info span {
  background-color: var(--primary-color);
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  font-weight: bold;
}

.movie-info span.green {
  color: lightgreen;
}

.movie-info span.orange {
  color: orange;
}

.movie-info span.red {
  color: red;
}

.overview {
  background-color: #fff;
  padding: 2rem;
  position: absolute;
  left: 0;
  bottom: 0;
  right: 0;
  max-height: 100%;
  transform: translateY(101%);
  overflow-y: auto;
  transition: transform 0.3s ease-in;
}

.movie:hover .overview {
  transform: translateY(0);
}
"""

script_js = """
const API_URL = 'https://api.themoviedb.org/3/discover/movie?sort_by=popularity.desc&api_key=3fd2be6f0c70a2a598f084ddfb75487c&page=1';
const IMG_PATH = 'https://image.tmdb.org/t/p/w1280';
const SEARCH_API = 'https://api.themoviedb.org/3/search/movie?api_key=3fd2be6f0c70a2a598f084ddfb75487c&query=';
const BACKEND_URL = 'http://127.0.0.1:5000/recommend?movie=';

const main = document.getElementById('main');
const form = document.getElementById('form');
const search = document.getElementById('search');

getMovies(API_URL);

async function getMovies(url) {
    const res = await fetch(url);
    const data = await res.json();
    showMovies(data.results);
}

function showMovies(movies) {
    main.innerHTML = '';
    movies.forEach((movie) => {
        const { title, poster_path, vote_average, overview } = movie;
        const movieEl = document.createElement('div');
        movieEl.classList.add('movie');
        movieEl.innerHTML = `
            <img src="${IMG_PATH + poster_path}" alt="${title}">
            <div class="movie-info">
                <h3>${title}</h3>
                <span class="${getClassByRate(vote_average)}">${vote_average}</span>
            </div>
            <div class="overview">
                <h3>Overview</h3>
                ${overview}
            </div>
        `;
        movieEl.addEventListener('click', () => {
            getRecommendations(title);
        });
        main.appendChild(movieEl);
    });
}

function getClassByRate(vote) {
    if (vote >= 8) {
        return 'green';
    } else if (vote >= 5) {
        return 'orange';
    } else {
        return 'red';
    }
}

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const searchTerm = search.value;
    if (searchTerm && searchTerm !== '') {
        getMovies(SEARCH_API + searchTerm);
        getRecommendations(searchTerm);
        search.value = '';
    } else {
        window.location.reload();
    }
});

async function getRecommendations(movie) {
    const res = await fetch(BACKEND_URL + movie);
    const data = await res.json();
    alert('Recommended: ' + data.join(', '));
}
"""

# Write files
with open(os.path.join(base_dir, "app.py"), "w") as f:
    f.write(app_py)
with open(os.path.join(base_dir, "index.html"), "w") as f:
    f.write(index_html)
with open(os.path.join(base_dir, "style.css"), "w") as f:
    f.write(style_css)
with open(os.path.join(base_dir, "script.js"), "w") as f:
    f.write(script_js)

# Copy dataset files
os.system(f"cp /mnt/data/tmdb_5000_movies.csv {base_dir}/")
os.system(f"cp /mnt/data/tmdb_5000_credits.csv {base_dir}/")

# Create zip
zip_path = "/mnt/data/movie_recommender_app.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for foldername, subfolders, filenames in os.walk(base_dir):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            arcname = os.path.relpath(filepath, base_dir)
            zipf.write(filepath, arcname)

zip_path

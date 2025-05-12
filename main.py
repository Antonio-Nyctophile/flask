
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load data
movies = pd.read_csv("cleaned_dataset.csv")
movies['Description'] = movies['Description'].fillna('')
movies['genres'] = movies['genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# TF-IDF and similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['Description'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=movies['movie_name'], columns=movies['movie_name'])

# Routes
@app.route('/')
def index():
    return "Movie Recommendation API is running!"

@app.route('/movies', methods=['GET'])
def get_all_movies():
    data = movies[['movie_name', 'genres']].to_dict(orient='records')
    return jsonify(data)

@app.route('/movie/<title>', methods=['GET'])
def get_movie(title):
    movie = movies[movies['movie_name'].str.lower() == title.lower()]
    if movie.empty:
        return jsonify({'error': 'Movie not found'}), 404
    return jsonify(movie.to_dict(orient='records')[0])

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if title not in similarity_df.index:
        return jsonify({'error': 'Movie not found'}), 404

    sim_scores = similarity_df[title].sort_values(ascending=False)[1:6]
    recommendations = sim_scores.index.tolist()
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

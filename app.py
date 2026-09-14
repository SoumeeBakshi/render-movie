from flask import Flask, render_template, request
import pickle
import difflib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open('movie_recommender_light.pkl', 'rb') as file:
    data = pickle.load(file)

movies_data = data['movies_data']
feature_vectors = data['feature_vectors']

list_of_all_titles = movies_data['title'].tolist()


def get_recommendations(movie_name):
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return None

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    similarity_scores = cosine_similarity(feature_vectors[index_of_the_movie], feature_vectors).flatten()
    similarity_score = list(enumerate(similarity_scores))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i < 11:
            recommendations.append(title_from_index)
            i += 1

    return recommendations


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    movie_name = request.form.get('movie_name')
    recommendations = get_recommendations(movie_name)

    if recommendations is None:
        return render_template('index.html', error="Movie not found. Try another title.", movie_name=movie_name)

    return render_template('index.html', recommendations=recommendations, movie_name=movie_name)


if __name__ == '__main__':
    app.run(debug=True)
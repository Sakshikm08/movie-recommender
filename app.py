from flask import Flask, render_template, request, jsonify, redirect, url_for
from recommendation_engine import MovieRecommender, GroqMovieEnhancer
import json

app = Flask(__name__)

# Initialize recommender systems (ONLY ONCE)
recommender = MovieRecommender()
groq_enhancer = None  # Initialize after checking if API key exists

try:
    recommender.load_data(
        movies_path='tmdb_5000_movies.csv',
        credits_path='tmdb_5000_credits.csv'
    )
    recommender.preprocess_data()
    recommender.build_content_based_model()
    print("✓ Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")

# Try to initialize Groq (optional - won't break app if no API key)
try:
    groq_enhancer = GroqMovieEnhancer()
    print("✓ Groq AI initialized successfully")
except Exception as e:
    print(f"⚠ Groq AI not available: {e}")
    print("AI features will be disabled. Add GROQ_API_KEY to .env to enable.")

# ============= BASIC ROUTES =============

@app.route('/')
def index():
    """Home page with featured movies"""
    top_movies = recommender.build_popularity_model().head(6)
    
    stats = {
        'total_movies': len(recommender.movies_df),
        'avg_rating': round(recommender.movies_df['vote_average'].mean(), 1),
        'total_genres': len(set([g for genres in recommender.movies_df['genres'] 
                                  for g in genres.split() if g]))
    }
    
    return render_template('index.html', 
                          movies=top_movies.to_dict('records'),
                          stats=stats)

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search for movies"""
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            results = recommender.search_movies(query)
            return render_template('search.html', 
                                  results=results.to_dict('records'),
                                  query=query)
    return render_template('search.html', results=[], query='')

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    """Get movie recommendations"""
    movie_titles = sorted(recommender.movies_df['title_x'].unique())
    
    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        num_recs = int(request.form.get('num_recommendations', 10))
        
        recs = recommender.get_content_recommendations(selected_movie, top_n=num_recs)
        
        if recs is not None:
            return render_template('recommendations.html',
                                  movie_titles=movie_titles,
                                  recommendations=recs.to_dict('records'),
                                  selected_movie=selected_movie)
    
    return render_template('recommendations.html', 
                          movie_titles=movie_titles,
                          recommendations=None,
                          selected_movie=None)

@app.route('/top-rated')
def top_rated():
    """Show top rated movies"""
    num_movies = request.args.get('num', 20, type=int)
    top_movies = recommender.build_popularity_model().head(num_movies)
    
    return render_template('top_rated.html', 
                          movies=top_movies.to_dict('records'))

@app.route('/genre', methods=['GET', 'POST'])
def genre():
    """Browse movies by genre"""
    # Extract unique genres
    all_genres = set()
    for genres_str in recommender.movies_df['genres']:
        for g in genres_str.split():
            if g:
                all_genres.add(g)
    
    genre_list = sorted(list(all_genres))
    
    if request.method == 'POST':
        selected_genre = request.form.get('genre')
        num_movies = int(request.form.get('num_movies', 10))
        
        genre_movies = recommender.get_recommendations_by_genre(
            selected_genre, 
            top_n=num_movies
        )
        
        if genre_movies is not None:
            return render_template('genre.html',
                                  genres=genre_list,
                                  movies=genre_movies.to_dict('records'),
                                  selected_genre=selected_genre)
    
    return render_template('genre.html', 
                          genres=genre_list,
                          movies=None,
                          selected_genre=None)

@app.route('/statistics')
def statistics():
    """Show dataset statistics"""
    # Genre distribution
    genre_counts = {}
    for genres_str in recommender.movies_df['genres']:
        for genre in genres_str.split():
            if genre:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Rating distribution
    rating_dist = recommender.movies_df['vote_average'].value_counts().sort_index()
    
    stats = {
        'genres': top_genres,
        'ratings': rating_dist.to_dict(),
        'total_movies': len(recommender.movies_df),
        'avg_rating': round(recommender.movies_df['vote_average'].mean(), 2)
    }
    
    return render_template('statistics.html', stats=stats)

# ============= AI-POWERED ROUTES (Optional) =============

@app.route('/ai-recommendations', methods=['GET', 'POST'])
def ai_recommendations():
    """Get AI-powered movie recommendations"""
    
    # Check if Groq is available
    if groq_enhancer is None:
        return render_template('error.html', 
                             message="AI features are not available. Please add GROQ_API_KEY to .env file.")
    
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 
              'Sci-Fi', 'Thriller', 'Adventure', 'Animation', 'Fantasy']
    
    moods = ['Uplifting', 'Intense', 'Relaxing', 'Thrilling', 
             'Emotional', 'Fun', 'Mysterious', 'Inspiring']
    
    if request.method == 'POST':
        selected_genre = request.form.get('genre')
        selected_mood = request.form.get('mood')
        num_movies = int(request.form.get('num_movies', 5))
        
        ai_response = groq_enhancer.get_ai_recommendations(
            genre=selected_genre if selected_genre != 'Any' else None,
            mood=selected_mood if selected_mood != 'Any' else None,
            num_movies=num_movies
        )
        
        return render_template('ai_recommendations.html',
                             genres=genres,
                             moods=moods,
                             ai_response=ai_response,
                             selected_genre=selected_genre,
                             selected_mood=selected_mood)
    
    return render_template('ai_recommendations.html',
                         genres=genres,
                         moods=moods,
                         ai_response=None)

@app.route('/movie-chat', methods=['GET', 'POST'])
def movie_chat():
    """Interactive movie chat powered by Groq"""
    
    # Check if Groq is available
    if groq_enhancer is None:
        return render_template('error.html', 
                             message="AI Chat is not available. Please add GROQ_API_KEY to .env file.")
    
    if request.method == 'POST':
        user_message = request.form.get('message')
        
        # Get AI response
        ai_response = groq_enhancer.chat_about_movies(user_message)
        
        return render_template('movie_chat.html',
                             user_message=user_message,
                             ai_response=ai_response)
    
    return render_template('movie_chat.html',
                         user_message=None,
                         ai_response=None)

@app.route('/enhanced-details/<movie_title>')
def enhanced_details(movie_title):
    """Show enhanced movie details with AI insights"""
    
    # Check if Groq is available
    if groq_enhancer is None:
        return render_template('error.html', 
                             message="AI enhancement is not available. Please add GROQ_API_KEY to .env file.")
    
    # Get movie from dataset
    movie_data = recommender.movies_df[
        recommender.movies_df['title_x'] == movie_title
    ]
    
    if len(movie_data) == 0:
        return "Movie not found", 404
    
    movie = movie_data.iloc[0]
    
    # Get AI-enhanced description
    enhanced_desc = groq_enhancer.enhance_movie_description(
        movie_title,
        movie['overview']
    )
    
    return render_template('enhanced_details.html',
                         movie=movie.to_dict(),
                         enhanced_description=enhanced_desc)

# ============= API ENDPOINTS =============

@app.route('/api/search/<query>')
def api_search(query):
    """API endpoint for search"""
    results = recommender.search_movies(query)
    return jsonify(results.to_dict('records'))

@app.route('/api/recommend/<movie>')
def api_recommend(movie):
    """API endpoint for recommendations"""
    num = request.args.get('num', 10, type=int)
    recs = recommender.get_content_recommendations(movie, top_n=num)
    
    if recs is not None:
        return jsonify(recs.to_dict('records'))
    return jsonify({'error': 'Movie not found'}), 404

# ============= ERROR HANDLER =============

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

# ============= RUN APP =============

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

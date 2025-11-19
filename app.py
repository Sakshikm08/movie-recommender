from flask import Flask, render_template, request, jsonify
from recommendation_engine import GroqMovieEnhancer, TMDBHelper
import os

app = Flask(__name__)

# Initialize only API-based services (no CSV loading)
groq_enhancer = None
tmdb_helper = None

print("=" * 50)
print("Initializing API-based services...")
print("=" * 50)

# Initialize Groq AI
try:
    groq_enhancer = GroqMovieEnhancer()
    print("✓ Groq AI initialized")
except Exception as e:
    print(f"⚠ Groq AI not available: {e}")

# Initialize TMDB API
try:
    tmdb_helper = TMDBHelper()
    print("✓ TMDB API initialized")
except Exception as e:
    print(f"⚠ TMDB API not available: {e}")

print("=" * 50)
print("SUCCESS! App ready (API-based mode)")
print("=" * 50)


@app.route('/')
def index():
    """Home page with trending movies"""
    if tmdb_helper:
        try:
            trending = tmdb_helper.get_trending_movies()[:6]
            stats = {
                'total_movies': '500K+',
                'avg_rating': '7.2',
                'total_genres': '20+'
            }
            return render_template('index.html', movies=trending, stats=stats)
        except:
            pass
    
    return render_template('index.html', movies=[], stats={
        'total_movies': '500K+',
        'avg_rating': '7.2',
        'total_genres': '20+'
    })


@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search movies via TMDB API"""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query and tmdb_helper:
            results = tmdb_helper.search_movie_list(query)
            return render_template('search.html', results=results, query=query)
    
    return render_template('search.html', results=None, query=None)


@app.route('/trending')
def trending():
    """Show trending movies"""
    if tmdb_helper is None:
        return render_template('error.html', 
                             message="TMDB API not available"), 500
    
    trending_movies = tmdb_helper.get_trending_movies()
    return render_template('trending.html', movies=trending_movies)


@app.route('/discover')
def discover():
    """Discover popular movies"""
    if tmdb_helper is None:
        return render_template('error.html',
                             message="TMDB API not available"), 500
    
    popular_movies = tmdb_helper.get_popular_movies(page=1)
    return render_template('discover.html', 
                          movies=popular_movies,
                          page_title="Discover Movies")


@app.route('/ai-recommendations', methods=['GET', 'POST'])
def ai_recommendations():
    """Get AI-powered recommendations"""
    
    if groq_enhancer is None:
        return render_template('error.html',
            message="AI features not available"), 500
    
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
            genres=genres, moods=moods,
            ai_response=ai_response,
            selected_genre=selected_genre,
            selected_mood=selected_mood)
    
    return render_template('ai_recommendations.html',
        genres=genres, moods=moods)


@app.route('/movie-chat', methods=['GET', 'POST'])
def movie_chat():
    """Chat about movies"""
    if groq_enhancer is None:
        return render_template('error.html', 
                             message="AI Chat not available"), 500
    
    if request.method == 'POST':
        user_message = request.form.get('message')
        ai_response = groq_enhancer.chat_about_movies(user_message)
        
        return render_template('movie_chat.html',
                             user_message=user_message,
                             ai_response=ai_response)
    
    return render_template('movie_chat.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

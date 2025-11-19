from flask import Flask, render_template, request, jsonify, redirect, url_for
from recommendation_engine import MovieRecommender, GroqMovieEnhancer, TMDBHelper
import pandas as pd
import json
import re
import os

app = Flask(__name__)

# Initialize recommender systems
recommender = MovieRecommender()
groq_enhancer = None
tmdb_helper = None

# ============= MEMORY-OPTIMIZED DATA LOADING =============
try:
    print("=" * 50)
    print("Loading data with memory optimization (2000 movies)...")
    print("=" * 50)
    
    # Load ONLY 2000 rows to fit in 512MB RAM
    recommender.movies_df = pd.read_csv('tmdb_5000_movies.csv', nrows=2000)
    recommender.credits_df = pd.read_csv('tmdb_5000_credits.csv', nrows=2000)
    
    print(f"✓ Loaded {len(recommender.movies_df)} movies")
    
    # Preprocess data
    recommender.preprocess_data()
    print("✓ Data preprocessed")
    
    # Build recommendation model
    recommender.build_content_based_model()
    print("✓ Content-based model built")
    
    print("=" * 50)
    print("SUCCESS: Recommender ready!")
    print("=" * 50)
    
except Exception as e:
    print("=" * 50)
    print(f"ERROR loading data: {e}")
    print("=" * 50)
    import traceback
    traceback.print_exc()
    recommender = None

# Initialize Groq AI (optional)
try:
    groq_enhancer = GroqMovieEnhancer()
    print("✓ Groq AI initialized")
except Exception as e:
    print(f"⚠ Groq AI not available: {e}")

# Initialize TMDB API (optional)
try:
    tmdb_helper = TMDBHelper()
    print("✓ TMDB API initialized")
except Exception as e:
    print(f"⚠ TMDB API not available: {e}")


# ============= BASIC ROUTES =============

@app.route('/')
def index():
    """Home page with featured movies"""
    if recommender is None:
        return render_template('error.html',
                             message="Movie database not loaded."), 500
    
    try:
        # Get top rated movies
        top_movies_df = recommender.build_popularity_model().head(6)
        movies_list = []
        
        # Enhance each movie with TMDB poster
        for _, movie in top_movies_df.iterrows():
            movie_dict = movie.to_dict()
            
            # Try to get poster from TMDB
            if tmdb_helper:
                tmdb_data = tmdb_helper.search_movie(movie['title_x'])
                if tmdb_data:
                    movie_dict['poster_url'] = tmdb_data['poster_url']
                else:
                    movie_dict['poster_url'] = None
            else:
                movie_dict['poster_url'] = None
            
            movies_list.append(movie_dict)
        
        # Get statistics
        stats = {
            'total_movies': len(recommender.movies_df),
            'avg_rating': round(recommender.movies_df['vote_average'].mean(), 1),
            'total_genres': len(recommender.movies_df['genres'].unique())
        }
        
        return render_template('index.html', 
                             movies=movies_list,
                             stats=stats)
    
    except Exception as e:
        return render_template('error.html', 
                             message=f"Error: {str(e)}"), 500


@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search for movies"""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            results = recommender.search_movies(query)
            
            # Enhance with TMDB posters
            enhanced_results = []
            for _, movie in results.iterrows():
                movie_dict = movie.to_dict()
                if tmdb_helper:
                    tmdb_data = tmdb_helper.search_movie(movie['title_x'])
                    if tmdb_data:
                        movie_dict['poster_url'] = tmdb_data['poster_url']
                    else:
                        movie_dict['poster_url'] = None
                else:
                    movie_dict['poster_url'] = None
                enhanced_results.append(movie_dict)
            
            return render_template('search.html', 
                                 results=enhanced_results,
                                 query=query)
    
    return render_template('search.html', results=None, query=None)


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    """Get movie recommendations"""
    movie_titles = sorted(recommender.movies_df['title_x'].unique())
    
    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        num_recs = int(request.form.get('num_recommendations', 10))
        
        recs = recommender.get_content_recommendations(selected_movie, top_n=num_recs)
        
        if recs is not None:
            # Enhance with TMDB posters
            enhanced_recs = []
            
            for _, movie in recs.iterrows():
                movie_dict = movie.to_dict()
                
                if tmdb_helper:
                    tmdb_data = tmdb_helper.search_movie(movie['title_x'])
                    if tmdb_data:
                        movie_dict['poster_url'] = tmdb_data['poster_url']
                    else:
                        movie_dict['poster_url'] = None
                else:
                    movie_dict['poster_url'] = None
                
                enhanced_recs.append(movie_dict)
            
            return render_template('recommendations.html',
                                 movie_titles=movie_titles,
                                 recommendations=enhanced_recs,
                                 selected_movie=selected_movie)
    
    return render_template('recommendations.html', 
                          movie_titles=movie_titles,
                          recommendations=None,
                          selected_movie=None)


@app.route('/top-rated')
def top_rated():
    """Show top rated movies"""
    num_movies = request.args.get('num', 20, type=int)
    top_movies_df = recommender.build_popularity_model().head(num_movies)
    
    # Enhance with TMDB posters
    movies_list = []
    for _, movie in top_movies_df.iterrows():
        movie_dict = movie.to_dict()
        
        if tmdb_helper:
            tmdb_data = tmdb_helper.search_movie(movie['title_x'])
            if tmdb_data:
                movie_dict['poster_url'] = tmdb_data['poster_url']
            else:
                movie_dict['poster_url'] = None
        else:
            movie_dict['poster_url'] = None
        
        movies_list.append(movie_dict)
    
    return render_template('top_rated.html', movies=movies_list)


@app.route('/genre', methods=['GET', 'POST'])
def genre():
    """Browse movies by genre"""
    # Extract unique genres
    all_genres = set()
    for genres_str in recommender.movies_df['genres']:
        for g in str(genres_str).split():
            if g:
                all_genres.add(g)
    
    genre_list = sorted(list(all_genres))
    
    if request.method == 'POST':
        selected_genre = request.form.get('genre')
        num_movies = int(request.form.get('num_movies', 10))
        
        genre_movies_df = recommender.get_recommendations_by_genre(
            selected_genre, 
            top_n=num_movies
        )
        
        if genre_movies_df is not None:
            # Enhance with TMDB posters
            enhanced_movies = []
            
            for _, movie in genre_movies_df.iterrows():
                movie_dict = movie.to_dict()
                
                if tmdb_helper:
                    tmdb_data = tmdb_helper.search_movie(movie['title_x'])
                    if tmdb_data:
                        movie_dict['poster_url'] = tmdb_data['poster_url']
                    else:
                        movie_dict['poster_url'] = None
                else:
                    movie_dict['poster_url'] = None
                
                enhanced_movies.append(movie_dict)
            
            return render_template('genre.html',
                                 genres=genre_list,
                                 movies=enhanced_movies,
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
        for genre in str(genres_str).split():
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


# ============= AI-POWERED ROUTES =============

@app.route('/ai-recommendations', methods=['GET', 'POST'])
def ai_recommendations():
    """Get AI-powered movie recommendations"""
    
    if groq_enhancer is None:
        return render_template(
            'error.html',
            message="AI features are not available. Please add GROQ_API_KEY to environment variables."
        ), 500
    
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance',
              'Sci-Fi', 'Thriller', 'Adventure', 'Animation', 'Fantasy']
    
    moods = ['Uplifting', 'Intense', 'Relaxing', 'Thrilling',
             'Emotional', 'Fun', 'Mysterious', 'Inspiring']
    
    if request.method == 'POST':
        selected_genre = request.form.get('genre')
        selected_mood = request.form.get('mood')
        num_movies = int(request.form.get('num_movies', 5))
        
        # Get AI response
        ai_response = groq_enhancer.get_ai_recommendations(
            genre=selected_genre if selected_genre != 'Any' else None,
            mood=selected_mood if selected_mood != 'Any' else None,
            num_movies=num_movies
        )
        
        movies_data = None
        
        try:
            # Extract JSON from markdown code block
            json_match = re.search(r'``````', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                # Remove 'json' language identifier if present
                if json_str.startswith('json'):
                    json_str = json_str[4:].strip()
                movies_data = json.loads(json_str)
            else:
                # Try finding JSON array directly
                json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    movies_data = json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"⚠️ JSON parsing error: {e}")
            movies_data = None
        
        return render_template(
            'ai_recommendations.html',
            genres=genres,
            moods=moods,
            movies_data=movies_data,
            ai_response=ai_response if not movies_data else None,
            selected_genre=selected_genre,
            selected_mood=selected_mood
        )
    
    return render_template(
        'ai_recommendations.html',
        genres=genres,
        moods=moods,
        movies_data=None,
        ai_response=None
    )


@app.route('/movie-chat', methods=['GET', 'POST'])
def movie_chat():
    """Interactive movie chat powered by Groq"""
    
    if groq_enhancer is None:
        return render_template('error.html', 
                             message="AI Chat is not available. Please add GROQ_API_KEY to environment variables."), 500
    
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
    
    if groq_enhancer is None:
        return render_template('error.html', 
                             message="AI enhancement is not available."), 500
    
    # Get movie from dataset
    movie_data = recommender.movies_df[
        recommender.movies_df['title_x'] == movie_title
    ]
    
    if len(movie_data) == 0:
        return render_template('error.html', message="Movie not found"), 404
    
    movie = movie_data.iloc[0]
    
    # Get AI-enhanced description
    enhanced_desc = groq_enhancer.enhance_movie_description(
        movie_title,
        movie['overview']
    )
    
    return render_template('enhanced_details.html',
                         movie=movie.to_dict(),
                         enhanced_description=enhanced_desc)


@app.route('/browse-enhanced')
def browse_enhanced():
    """Browse movies to see enhanced details"""
    movie_titles = sorted(recommender.movies_df['title_x'].unique())
    return render_template('browse_enhanced.html', movie_titles=movie_titles)


@app.route('/trending')
def trending():
    """Show trending movies from TMDB"""
    if tmdb_helper is None:
        return render_template('error.html', 
                             message="TMDB API not available. Please add TMDB_API_KEY to environment variables."), 500
    
    trending_movies = tmdb_helper.get_trending_movies()
    
    return render_template('trending.html', 
                          movies=trending_movies)


@app.route('/discover')
def discover():
    """Discover new movies from TMDB"""
    if tmdb_helper is None:
        return render_template('error.html',
                             message="TMDB API not available."), 500
    
    popular_movies = tmdb_helper.get_popular_movies(page=1)
    
    return render_template('discover.html', 
                          movies=popular_movies,
                          page_title="Discover New Movies")


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


# ============= ERROR HANDLERS =============

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', message="Internal server error"), 500


# ============= RUN APP =============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # ← Changed from 5000 to 8080
    app.run(host='0.0.0.0', port=port, debug=False)


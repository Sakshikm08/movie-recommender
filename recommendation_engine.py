import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler
import pickle
import ast
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.credits_df = None
        self.cosine_sim = None
        self.indices = None
        
    def load_data(self, movies_path='tmdb_5000_movies.csv', credits_path='tmdb_5000_credits.csv'):
        """Load TMDB dataset"""
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path)
        
        # Merge datasets
        self.credits_df.columns = ['id', 'title', 'cast', 'crew']
        self.movies_df = self.movies_df.merge(self.credits_df, on='id')
        
        return self.movies_df
    
    def preprocess_data(self):
        """Clean and prepare data for recommendation"""
        # Remove duplicates
        self.movies_df = self.movies_df.drop_duplicates(subset=['title_x'])
        
        # Handle missing values
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        self.movies_df['tagline'] = self.movies_df['tagline'].fillna('')
        
        # Parse JSON columns
        self.movies_df['genres'] = self.movies_df['genres'].apply(self._parse_features)
        self.movies_df['keywords'] = self.movies_df['keywords'].apply(self._parse_features)
        self.movies_df['cast'] = self.movies_df['cast'].apply(self._parse_cast)
        self.movies_df['crew'] = self.movies_df['crew'].apply(self._get_director)
        
        # Create soup of features
        self.movies_df['soup'] = self._create_soup()
        
        return self.movies_df
    
    def _parse_features(self, x):
        """Parse JSON features"""
        try:
            data = ast.literal_eval(x)
            return ' '.join([i['name'].replace(" ", "") for i in data])
        except:
            return ''
    
    def _parse_cast(self, x):
        """Get top 3 cast members"""
        try:
            data = ast.literal_eval(x)
            return ' '.join([i['name'].replace(" ", "") for i in data[:3]])
        except:
            return ''
    
    def _get_director(self, x):
        """Extract director name"""
        try:
            data = ast.literal_eval(x)
            for i in data:
                if i['job'] == 'Director':
                    return i['name'].replace(" ", "")
            return ''
        except:
            return ''
    
    def _create_soup(self):
        """Combine all features into metadata soup"""
        return (self.movies_df['keywords'] + ' ' + 
                self.movies_df['cast'] + ' ' +
                self.movies_df['crew'] + ' ' + 
                self.movies_df['genres'] + ' ' +
                self.movies_df['overview'])
    
    def build_content_based_model(self):
        """Build content-based recommendation model using TF-IDF"""
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(self.movies_df['soup'])
        
        # Compute cosine similarity
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Create reverse mapping of indices
        self.indices = pd.Series(
            self.movies_df.index, 
            index=self.movies_df['title_x']
        ).drop_duplicates()
        
        return self.cosine_sim
    
    def get_content_recommendations(self, title, top_n=10):
        """Get content-based recommendations"""
        try:
            idx = self.indices[title]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            
            movie_indices = [i[0] for i in sim_scores]
            recommendations = self.movies_df.iloc[movie_indices][
                ['title_x', 'vote_average', 'genres', 'overview', 'release_date']
            ].copy()
            recommendations['similarity_score'] = [i[1] for i in sim_scores]
            
            return recommendations
        except KeyError:
            return None
    
    def build_popularity_model(self):
        """Build weighted rating model"""
        C = self.movies_df['vote_average'].mean()
        m = self.movies_df['vote_count'].quantile(0.75)
        
        qualified = self.movies_df[self.movies_df['vote_count'] >= m].copy()
        
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)
        
        qualified['score'] = qualified.apply(weighted_rating, axis=1)
        return qualified.sort_values('score', ascending=False)
    
    def get_recommendations_by_genre(self, genre, top_n=10):
        """Get top movies by genre"""
        genre_movies = self.movies_df[
            self.movies_df['genres'].str.contains(genre, case=False, na=False)
        ].copy()
        
        if len(genre_movies) == 0:
            return None
        
        C = genre_movies['vote_average'].mean()
        m = genre_movies['vote_count'].quantile(0.60)
        
        qualified = genre_movies[genre_movies['vote_count'] >= m].copy()
        
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)
        
        qualified['score'] = qualified.apply(weighted_rating, axis=1)
        top_movies = qualified.sort_values('score', ascending=False).head(top_n)
        
        return top_movies[['title_x', 'vote_average', 'genres', 'overview', 'release_date', 'score']]
    
    def search_movies(self, query):
        """Search movies by title"""
        mask = self.movies_df['title_x'].str.contains(query, case=False, na=False)
        return self.movies_df[mask][['title_x', 'vote_average', 'genres', 'overview', 'release_date']].head(20)
    
    def save_model(self, filepath='movie_recommender.pkl'):
        """Save the model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'cosine_sim': self.cosine_sim,
                'indices': self.indices,
                'movies_df': self.movies_df
            }, f)
    
    def load_model(self, filepath='movie_recommender.pkl'):
        """Load pre-trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.cosine_sim = data['cosine_sim']
            self.indices = data['indices']
            self.movies_df = data['movies_df']


class GroqMovieEnhancer:
    def __init__(self):
        """Initialize Groq client"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"  # Updated model (Nov 2025)

    
    def get_ai_recommendations(self, genre=None, mood=None, num_movies=5):
        """Get AI-powered movie recommendations from Groq"""
        
        prompt = f"""You are a movie expert. Recommend {num_movies} movies"""
        
        if genre:
            prompt += f" in the {genre} genre"
        if mood:
            prompt += f" that match a {mood} mood"
        
        prompt += """.
        
For each movie, provide:
1. Title
2. Year
3. Brief description (2-3 sentences)
4. Why it's recommended
5. Rating out of 10

Format as JSON array with keys: title, year, description, reason, rating"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert movie critic and recommendation system. Provide diverse, high-quality movie suggestions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=2000
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Error getting AI recommendations: {str(e)}"
    
    def enhance_movie_description(self, movie_title, original_overview):
        """Enhance movie description with AI-generated insights"""
        
        prompt = f"""Given this movie: "{movie_title}"
Original description: {original_overview}

Provide:
1. A more engaging 2-sentence description
2. Three key themes/elements
3. One similar movie recommendation

Keep it concise and engaging."""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.6,
                max_tokens=300
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return original_overview
    
    def explain_recommendation(self, source_movie, recommended_movie, similarity_score):
        """Explain why a movie was recommended"""
        
        prompt = f"""Explain in 2-3 sentences why someone who liked "{source_movie}" would enjoy "{recommended_movie}". 
The similarity score is {similarity_score:.2%}. Focus on themes, style, or emotional resonance."""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=150
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return "Similar themes and style."
    
    def chat_about_movies(self, user_message, conversation_history=None):
        """Interactive chat about movies"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a friendly movie expert. Help users discover movies they'll love based on their preferences. Be conversational and enthusiastic."
            }
        ]
        
        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=500
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
        
class TMDBHelper:
    def __init__(self):
        """Initialize TMDB API client"""
        api_key = os.getenv('TMDB_API_KEY')
        if not api_key:
            raise ValueError("TMDB_API_KEY not found in environment variables")
        
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p"
        
    def get_poster_url(self, poster_path, size="w500"):
        """Get full poster URL from poster path"""
        if not poster_path:
            return "/static/images/no-poster.jpg"  # Fallback image
        return f"{self.image_base_url}/{size}{poster_path}"
    
    def search_movie(self, title):
        """Search for a movie and get details including poster"""
        import requests
        
        url = f"{self.base_url}/search/movie"
        params = {
            'api_key': self.api_key,
            'query': title,
            'language': 'en-US'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['results']:
                movie = data['results'][0]  # Get first result
                return {
                    'title': movie.get('title'),
                    'poster_path': movie.get('poster_path'),
                    'poster_url': self.get_poster_url(movie.get('poster_path')),
                    'overview': movie.get('overview'),
                    'release_date': movie.get('release_date'),
                    'vote_average': movie.get('vote_average'),
                    'vote_count': movie.get('vote_count'),
                    'backdrop_path': movie.get('backdrop_path')
                }
            return None
        except Exception as e:
            print(f"Error fetching movie from TMDB: {e}")
            return None
    
    def get_popular_movies(self, page=1):
        """Get popular movies with posters"""
        import requests
        
        url = f"{self.base_url}/movie/popular"
        params = {
            'api_key': self.api_key,
            'language': 'en-US',
            'page': page
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            movies = []
            for movie in data['results']:
                movies.append({
                    'title': movie.get('title'),
                    'poster_url': self.get_poster_url(movie.get('poster_path')),
                    'overview': movie.get('overview'),
                    'release_date': movie.get('release_date'),
                    'vote_average': movie.get('vote_average'),
                    'vote_count': movie.get('vote_count')
                })
            
            return movies
        except Exception as e:
            print(f"Error fetching popular movies: {e}")
            return []
    
    def get_trending_movies(self):
        """Get trending movies this week"""
        import requests
        
        url = f"{self.base_url}/trending/movie/week"
        params = {'api_key': self.api_key}
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            movies = []
            for movie in data['results'][:10]:  # Top 10
                movies.append({
                    'title': movie.get('title'),
                    'poster_url': self.get_poster_url(movie.get('poster_path')),
                    'overview': movie.get('overview'),
                    'release_date': movie.get('release_date'),
                    'vote_average': movie.get('vote_average')
                })
            
            return movies
        except Exception as e:
            print(f"Error fetching trending movies: {e}")
            return []

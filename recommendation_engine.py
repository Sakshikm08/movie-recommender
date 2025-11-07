import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler
import pickle
import ast

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

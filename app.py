from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Movie Recommender</title>
        <style>
            body { font-family: Arial; margin: 50px; background: #1a1a2e; color: white; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #16c79a; }
            .card { background: #0f3460; padding: 20px; margin: 20px 0; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Movie Recommender - Demo</h1>
            <div class="card">
                <h2>Project Overview</h2>
                <p>AI-powered movie recommendation system with ML algorithms</p>
            </div>
            <div class="card">
                <h3>Tech Stack:</h3>
                <p>Python • Flask • Docker • Pandas • Scikit-learn • Groq API • TMDB API</p>
            </div>
            <div class="card">
                <h3>GitHub:</h3>
                <a href="https://github.com/Sakshikm08/movie-recommender" style="color: #16c79a;">View Source Code</a>
            </div>
            <div class="card">
                <p><strong>Status:</strong> Demo version deployed successfully ✅</p>
                <p>Full ML features require 2GB RAM (current free tier: 512MB)</p>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/health')
def health():
    return {'status': 'ok'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Server starting on port {port}")
    app.run(host='0.0.0.0', port=port)

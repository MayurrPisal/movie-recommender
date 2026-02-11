import pandas as pd
import ast
import os


def load_and_clean_data():
    """
    Load and clean the TMDB 5000 movies dataset
    """
    # Get the directory where this file is located (ml directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to movie_recommender directory
    movie_recommender_dir = os.path.dirname(current_dir)
    # Data is in movie_recommender/data
    data_dir = os.path.join(movie_recommender_dir, 'data')
    
    movies_path = os.path.join(data_dir, 'tmdb_5000_movies.csv')
    credits_path = os.path.join(data_dir, 'tmdb_5000_credits.csv')
    
    # Check if files exist
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Movies file not found at: {movies_path}")
    if not os.path.exists(credits_path):
        raise FileNotFoundError(f"Credits file not found at: {credits_path}")
    
    print(f"Loading data from: {data_dir}")
    
    # Load datasets
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    # Merge datasets
    df = movies.merge(credits, on='title')
    
    # Parse JSON-like strings
    df['genres'] = df['genres'].apply(safe_parse)
    df['cast'] = df['cast'].apply(safe_parse)
    
    # Extract names from parsed data
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x[:5]] if isinstance(x, list) else [])
    
    # Keep only necessary columns
    df = df[['title', 'genres', 'cast', 'vote_average', 'popularity']]
    
    # Remove duplicates
    df = df.drop_duplicates(subset='title')
    
    # Create combined column for similarity calculation
    # Combine genres and cast into a single string
    df['combined'] = df['genres'].apply(lambda x: ' '.join(x)) + ' ' + df['cast'].apply(lambda x: ' '.join(x))
    
    print(f"Loaded {len(df)} movies")
    
    return df


def safe_parse(x):
    """
    Safely parse JSON-like strings
    """
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []
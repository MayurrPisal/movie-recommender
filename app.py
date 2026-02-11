import streamlit as st
import pandas as pd
from ml.preprocessing import load_and_clean_data
from ml.similarity import build_similarity_matrix
from ml.association import build_association_rules
from ml.recommender import hybrid_recommend
from llm.agent import create_simple_agent
import time
import random


# Page configuration
st.set_page_config(
    page_title="🎬 CineMatch - AI Movie Theater",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for theater vibe
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Raleway:wght@300;400;700&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a0000 50%, #0a0a0a 100%);
        color: #FFD700;
    }
    
    /* Theater Curtain Effect */
    .main {
        background-image: 
            linear-gradient(90deg, rgba(139,0,0,0.3) 0%, transparent 10%, transparent 90%, rgba(139,0,0,0.3) 100%),
            radial-gradient(circle at 50% 0%, rgba(139,0,0,0.2) 0%, transparent 50%);
        position: relative;
    }
    
    /* Animated Movie Title */
    .cinema-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 5rem;
        text-align: center;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF6347, #FFD700);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite, flicker 0.5s infinite alternate;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        margin: 20px 0;
        letter-spacing: 8px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes flicker {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.95; }
    }
    
    .subtitle {
        font-family: 'Raleway', sans-serif;
        text-align: center;
        color: #B8860B;
        font-size: 1.3rem;
        margin-bottom: 40px;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    /* Marquee Lights */
    .marquee-container {
        background: linear-gradient(90deg, #8B0000, #DC143C, #8B0000);
        padding: 15px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(220, 20, 60, 0.6), inset 0 0 20px rgba(0,0,0,0.5);
        border: 3px solid #FFD700;
        position: relative;
        overflow: hidden;
    }
    
    .marquee-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Movie Card Styling */
    .movie-card {
        background: linear-gradient(135deg, rgba(139,0,0,0.4) 0%, rgba(0,0,0,0.8) 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid #FFD700;
        box-shadow: 
            0 10px 40px rgba(255, 215, 0, 0.3),
            inset 0 0 20px rgba(139, 0, 0, 0.3);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '🎬';
        position: absolute;
        top: -20px;
        right: -20px;
        font-size: 100px;
        opacity: 0.1;
        transform: rotate(15deg);
    }
    
    .movie-card:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: #FFA500;
        box-shadow: 
            0 15px 50px rgba(255, 165, 0, 0.5),
            inset 0 0 30px rgba(220, 20, 60, 0.3);
    }
    
    .movie-number {
        font-family: 'Bebas Neue', cursive;
        font-size: 3.5rem;
        color: #FFD700;
        display: inline-block;
        margin-right: 20px;
        text-shadow: 0 0 10px #FFA500, 0 0 20px #FF6347;
    }
    
    .movie-title {
        font-family: 'Raleway', sans-serif;
        font-size: 1.8rem;
        color: #FFFFFF;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    
    .movie-info {
        color: #DEB887;
        font-size: 1rem;
        margin-top: 10px;
        font-family: 'Raleway', sans-serif;
    }
    
    /* Popcorn Stats */
    .stats-container {
        background: linear-gradient(135deg, rgba(139,0,0,0.3), rgba(0,0,0,0.6));
        border-radius: 15px;
        padding: 20px;
        border: 2px solid #8B0000;
        box-shadow: 0 5px 20px rgba(139, 0, 0, 0.5);
        text-align: center;
    }
    
    .stats-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .stats-number {
        font-family: 'Bebas Neue', cursive;
        font-size: 3rem;
        color: #FFD700;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    .stats-label {
        color: #DEB887;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Search Box */
    .stTextInput input {
        background: rgba(0,0,0,0.7) !important;
        border: 2px solid #FFD700 !important;
        border-radius: 25px !important;
        color: #FFD700 !important;
        font-size: 1.2rem !important;
        padding: 15px 25px !important;
        font-family: 'Raleway', sans-serif !important;
    }
    
    .stTextInput input:focus {
        border-color: #FFA500 !important;
        box-shadow: 0 0 20px rgba(255, 165, 0, 0.5) !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #8B0000, #DC143C, #8B0000) !important;
        color: #FFD700 !important;
        border: 3px solid #FFD700 !important;
        border-radius: 30px !important;
        padding: 15px 50px !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        font-family: 'Bebas Neue', cursive !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(220, 20, 60, 0.5) !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 10px 40px rgba(255, 215, 0, 0.6) !important;
        border-color: #FFA500 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(139,0,0,0.3);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0,0,0,0.5);
        border: 2px solid #8B0000;
        border-radius: 15px;
        color: #FFD700;
        font-size: 1.2rem;
        font-family: 'Bebas Neue', cursive;
        padding: 15px 30px;
        letter-spacing: 2px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #8B0000, #DC143C);
        border-color: #FFD700;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(0,0,0,0.7) !important;
        border: 2px solid #FFD700 !important;
        border-radius: 15px !important;
        color: #FFD700 !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(0,100,0,0.3) !important;
        border: 2px solid #32CD32 !important;
        border-radius: 15px !important;
        color: #90EE90 !important;
    }
    
    .stError {
        background: rgba(139,0,0,0.4) !important;
        border: 2px solid #DC143C !important;
        border-radius: 15px !important;
        color: #FFB6C1 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(139,0,0,0.3) !important;
        border: 1px solid #FFD700 !important;
        border-radius: 10px !important;
        color: #FFD700 !important;
        font-family: 'Raleway', sans-serif !important;
    }
    
    /* Film Strip Effect */
    .film-strip {
        background: 
            repeating-linear-gradient(
                90deg,
                #1a1a1a 0px,
                #1a1a1a 10px,
                transparent 10px,
                transparent 20px
            ),
            linear-gradient(180deg, #2a2a2a, #1a1a1a);
        height: 20px;
        margin: 30px 0;
        border-top: 3px solid #FFD700;
        border-bottom: 3px solid #FFD700;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
    }
    
    /* Spotlight Effect */
    .spotlight {
        position: fixed;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,215,0,0.1) 0%, transparent 70%);
        pointer-events: none;
        transition: all 0.1s ease;
        z-index: 9999;
    }
    
    /* Now Showing Banner */
    .now-showing {
        background: linear-gradient(45deg, #8B0000, #DC143C);
        padding: 20px;
        border-radius: 15px;
        border: 3px solid #FFD700;
        margin: 30px 0;
        box-shadow: 0 0 30px rgba(220, 20, 60, 0.6);
        text-align: center;
    }
    
    .now-showing h2 {
        font-family: 'Bebas Neue', cursive;
        color: #FFD700;
        font-size: 2.5rem;
        letter-spacing: 5px;
        margin: 0;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
    }
    
    /* Ticket Stub Style */
    .ticket {
        background: linear-gradient(135deg, #2C1810, #3D2314);
        border: 2px dashed #FFD700;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        position: relative;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }
    
    .ticket::before {
        content: 'ADMIT ONE';
        position: absolute;
        top: 10px;
        right: 20px;
        color: rgba(255, 215, 0, 0.3);
        font-size: 0.8rem;
        letter-spacing: 2px;
        transform: rotate(90deg);
        transform-origin: right;
    }
    </style>
""", unsafe_allow_html=True)


# Cache the data loading
@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize the recommendation system"""
    df = load_and_clean_data()
    similarity_matrix = build_similarity_matrix(df)
    rules = build_association_rules(df)
    agent = create_simple_agent(df, similarity_matrix, rules, hybrid_recommend)
    return df, similarity_matrix, rules, agent


def get_movie_details(df, movie_title):
    """Get detailed info about a movie"""
    movie = df[df['title'] == movie_title]
    if len(movie) > 0:
        return {
            'genres': ', '.join(movie.iloc[0]['genres'][:3]),
            'cast': ', '.join(movie.iloc[0]['cast'][:3]),
            'rating': movie.iloc[0]['vote_average'],
            'popularity': movie.iloc[0]['popularity']
        }
    return None


def display_movie_card_with_details(movie_title, index, df):
    """Display movie card with full details"""
    details = get_movie_details(df, movie_title)
    
    if details:
        st.markdown(f"""
            <div class="movie-card">
                <span class="movie-number">#{index}</span>
                <span class="movie-title">{movie_title}</span>
                <div class="movie-info">
                    <p>🎭 <b>Genres:</b> {details['genres']}</p>
                    <p>⭐ <b>Rating:</b> {details['rating']}/10 | 🔥 <b>Popularity:</b> {details['popularity']:.0f}</p>
                    <p>🎬 <b>Cast:</b> {details['cast']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="movie-card">
                <span class="movie-number">#{index}</span>
                <span class="movie-title">{movie_title}</span>
            </div>
        """, unsafe_allow_html=True)


def main():
    # Header with theater marquee
    st.markdown('<div class="cinema-title">🎬 CINEMATCH 🍿</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your Personal AI Movie Theater</p>', unsafe_allow_html=True)
    
    # Film strip divider
    st.markdown('<div class="film-strip"></div>', unsafe_allow_html=True)
    
    # Initialize system
    try:
        with st.spinner('🎬 Rolling the film...'):
            df, similarity_matrix, rules, agent = initialize_system()
        
        # Stats Dashboard - Popcorn style
        st.markdown('<div class="now-showing"><h2>📊 BOX OFFICE STATS</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="stats-container">
                    <div class="stats-icon">🎥</div>
                    <div class="stats-number">{len(df):,}</div>
                    <div class="stats-label">Total Movies</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_genres = df['genres'].explode().nunique()
            st.markdown(f"""
                <div class="stats-container">
                    <div class="stats-icon">🎭</div>
                    <div class="stats-number">{unique_genres}</div>
                    <div class="stats-label">Genres</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_cast = df['cast'].explode().nunique()
            st.markdown(f"""
                <div class="stats-container">
                    <div class="stats-icon">⭐</div>
                    <div class="stats-number">{unique_cast:,}</div>
                    <div class="stats-label">Actors</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_rating = df['vote_average'].mean()
            st.markdown(f"""
                <div class="stats-container">
                    <div class="stats-icon">🍿</div>
                    <div class="stats-number">{avg_rating:.1f}/10</div>
                    <div class="stats-label">Avg Rating</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="film-strip"></div>', unsafe_allow_html=True)
        
        # Main content
        st.markdown('<div class="now-showing"><h2>🔍 FIND YOUR NEXT MOVIE</h2></div>', unsafe_allow_html=True)
        
        # Tabs for different search methods
        tab1, tab2, tab3 = st.tabs(["🤖 AI CONCIERGE", "🎯 QUICK SEARCH", "🎬 BROWSE CATALOG"])
        
        with tab1:
            st.markdown("### 🎭 Talk to Our AI Movie Expert")
            
            user_input = st.text_input(
                "",
                placeholder="e.g., 'Show me movies like Inception' or 'I want action thrillers'...",
                key="ai_chat",
                label_visibility="collapsed"
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 6])
            with col_btn1:
                search_button = st.button("🎬 GET TICKETS", key="ai_btn", use_container_width=True)
            
            if search_button and user_input:
                with st.spinner("🎬 AI is finding your perfect match..."):
                    response = agent(user_input)
                    time.sleep(0.5)  # Dramatic pause
                
                st.markdown('<div class="marquee-container">', unsafe_allow_html=True)
                st.success(f"🎭 **AI Concierge Says:**\n\n{response}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 🎯 Direct Movie Search")
            
            col_search1, col_search2 = st.columns([3, 1])
            
            with col_search1:
                movie_list = sorted(df['title'].tolist())
                selected_movie = st.selectbox(
                    "Pick a movie from our collection:",
                    options=[""] + movie_list,
                    key="direct_search"
                )
            
            with col_search2:
                st.markdown("<br>", unsafe_allow_html=True)
                direct_button = st.button("🎟️ FIND SIMILAR", key="direct_btn", use_container_width=True)
            
            if direct_button and selected_movie:
                with st.spinner("🎬 Searching the vault..."):
                    movies = hybrid_recommend(selected_movie, df, similarity_matrix, rules)
                    time.sleep(0.3)
                
                if movies and movies[0] != "Movie not found in dataset":
                    st.markdown(f'<div class="now-showing"><h2>🎟️ SIMILAR TO: {selected_movie.upper()}</h2></div>', unsafe_allow_html=True)
                    
                    for i, movie in enumerate(movies, 1):
                        display_movie_card_with_details(movie, i, df)
                else:
                    st.error(f"🎬 Sorry! '{selected_movie}' not found in our theater!")
        
        with tab3:
            st.markdown("### 🎬 Browse Our Movie Collection")
            
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                all_genres = sorted(set([genre for genres in df['genres'] for genre in genres]))
                selected_genre = st.selectbox("🎭 Filter by Genre:", ["All Genres"] + all_genres)
            
            with col_filter2:
                sort_by = st.selectbox("📊 Sort by:", ["Popularity", "Rating", "Title"])
            
            with col_filter3:
                min_rating = st.slider("⭐ Minimum Rating:", 0.0, 10.0, 6.0, 0.5)
            
            # Filter data
            filtered_df = df[df['vote_average'] >= min_rating].copy()
            
            if selected_genre != "All Genres":
                filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: selected_genre in x)]
            
            # Sort
            if sort_by == "Popularity":
                filtered_df = filtered_df.sort_values('popularity', ascending=False)
            elif sort_by == "Rating":
                filtered_df = filtered_df.sort_values('vote_average', ascending=False)
            else:
                filtered_df = filtered_df.sort_values('title')
            
            st.markdown(f'<div class="now-showing"><h2>🎟️ NOW SHOWING: {len(filtered_df)} MOVIES</h2></div>', unsafe_allow_html=True)
            
            # Display movies in expandable cards
            for idx, row in filtered_df.head(20).iterrows():
                with st.expander(f"⭐ {row['title']} ({row['vote_average']}/10)"):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.markdown(f"""
                            <div class="ticket">
                                <p><b>🎭 Genres:</b> {', '.join(row['genres'][:3])}</p>
                                <p><b>⭐ Rating:</b> {row['vote_average']}/10</p>
                                <p><b>🔥 Popularity:</b> {row['popularity']:.1f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_info2:
                        st.markdown(f"""
                            <div class="ticket">
                                <p><b>🎬 Cast:</b> {', '.join(row['cast'][:3])}</p>
                                <p><b>🎭 All Genres:</b> {', '.join(row['genres'])}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Quick recommend button
                    if st.button(f"🎟️ Find Similar Movies", key=f"rec_{idx}"):
                        with st.spinner("🎬 Finding matches..."):
                            similar = hybrid_recommend(row['title'], df, similarity_matrix, rules)
                        
                        st.markdown("**🎬 You might also like:**")
                        for i, movie in enumerate(similar[:3], 1):
                            details = get_movie_details(df, movie)
                            if details:
                                st.markdown(f"**{i}.** {movie} ⭐ {details['rating']}/10")
        
        # Footer
        st.markdown('<div class="film-strip"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; color: #B8860B; padding: 30px; font-family: "Raleway", sans-serif;'>
                <p style='font-size: 1.5rem; letter-spacing: 3px;'>🎬 ENJOY THE SHOW 🍿</p>
                <p>Powered by AI • Made with ❤️ • TMDB Dataset</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🎬 **Technical Difficulties!** {str(e)}")
        st.info("💡 Make sure Ollama is running and your model is downloaded!")


if __name__ == "__main__":
    main()
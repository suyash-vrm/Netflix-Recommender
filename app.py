import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

st.set_page_config(page_title="Netflix Recommender", layout="wide")

# ==============================================================================
# 1. Data Loading & Preprocessing
# ==============================================================================
@st.cache_data
def load_data():
    movies_path = os.path.join("data", "movie_titles.csv")
    ratings_path = os.path.join("data", "combined_data_1.txt")
    
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        return None, None

    movies_df = pd.read_csv(movies_path, header=None, names=['movie_id', 'year', 'title'], encoding='iso-8859-1', on_bad_lines='skip')
    movies_df['movie_id'] = pd.to_numeric(movies_df['movie_id'], errors='coerce')
    movies_df = movies_df.dropna(subset=['movie_id'])
    movies_df['movie_id'] = movies_df['movie_id'].astype(int)
    
    with open(ratings_path, 'r') as f:
        data = f.readlines()
        
    parsed_data = []
    current_movie = None
    for line in data[:500000]:  # Limit for performance
        line = line.strip().replace('\x00', '').replace('\xff\xfe', '').replace('\xef\xbb\xbf', '')
        if not line:
            continue
        if line.endswith(':'):
            current_movie = int(line[:-1])
        else:
            parts = line.split(',')
            if len(parts) >= 2:
                user_id = int(parts[0])
                rating = float(parts[1])
                parsed_data.append([user_id, current_movie, rating])
                
    ratings_df = pd.DataFrame(parsed_data, columns=['user_id', 'movie_id', 'rating'])
    return ratings_df, movies_df

# ==============================================================================
# 2. Model Training (SVD)
# ==============================================================================
@st.cache_resource
def train_model(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    return model, trainset

# ==============================================================================
# 3. Recommendations & Hybrid Engine
# ==============================================================================
def get_hybrid_recommendations(user_id, model, trainset, movies_df, ratings_df, top_n=10):
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    user_movies = set(user_ratings['movie_id'].values)
    
    all_movies = set(movies_df['movie_id'].values)
    unseen_movies = list(all_movies - user_movies)
    
    if len(user_ratings) == 0:
        return None, "Cold Start: This user has no rating history. Showing popular movies instead."
        
    predictions = []
    for movie_id in unseen_movies:
        pred = model.predict(user_id, movie_id).est
        predictions.append((movie_id, pred))
        
    # Hybrid Approach: Boost movies that match the release decade of the user's top rated movies
    top_user_movies = user_ratings[user_ratings['rating'] >= 4.0]['movie_id']
    favorite_decades = []
    if len(top_user_movies) > 0:
        fav_years = movies_df[movies_df['movie_id'].isin(top_user_movies)]['year'].dropna()
        favorite_decades = [int(y // 10 * 10) for y in fav_years if str(y).replace('.','',1).isdigit()]
        
    hybrid_predictions = []
    for m_id, svd_score in predictions:
        movie_row = movies_df[movies_df['movie_id'] == m_id]
        hybrid_score = svd_score
        
        # Apply hybrid boost
        if not movie_row.empty and len(favorite_decades) > 0:
            try:
                m_year = float(movie_row['year'].values[0])
                m_decade = int(m_year // 10 * 10)
                if m_decade in favorite_decades:
                    hybrid_score += 0.15  # Small boost for matching preferred decades
            except:
                pass
                
        hybrid_predictions.append((m_id, hybrid_score))
        
    hybrid_predictions.sort(key=lambda x: x[1], reverse=True)
    return hybrid_predictions[:top_n], None

def explain_recommendation(movie_id, user_id, ratings_df, movies_df):
    user_history = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4.0)]
    if len(user_history) == 0:
        return "Recommended based on general popularity."
        
    # Find overlapping users
    movie_raters = set(ratings_df[ratings_df['movie_id'] == movie_id]['user_id'])
    
    best_explanation = ""
    max_overlap = 0
    
    for _, row in user_history.iterrows():
        past_movie_id = row['movie_id']
        past_raters = set(ratings_df[ratings_df['movie_id'] == past_movie_id]['user_id'])
        overlap = len(movie_raters.intersection(past_raters))
        if overlap > max_overlap:
            max_overlap = overlap
            try:
                title = movies_df[movies_df['movie_id'] == past_movie_id]['title'].values[0]
                best_explanation = f"Recommended because you highly rated '{title}', and similar users who enjoyed it also liked this."
            except:
                pass
                
    if best_explanation:
        return best_explanation
    return "Recommended based on your latent taste profile."

# ==============================================================================
# 4. Streamlit Dashboard UI
# ==============================================================================
st.title("🎥 Netflix Recommendation Engine")
st.markdown("An interactive hybrid recommendation dashboard built with Matrix Factorization (SVD) and Content Metadata.")

ratings_df, movies_df = load_data()

if ratings_df is None:
    st.error("Dataset not found! Please ensure `movie_titles.csv` and `combined_data_1.txt` are in the `data/` folder.")
    st.stop()

with st.spinner("Training Model... (This takes a few seconds)"):
    model, trainset = train_model(ratings_df)

unique_users = ratings_df['user_id'].unique()

st.sidebar.header("User Selection")
selected_user = st.sidebar.selectbox("Select a User ID:", unique_users[:100])

st.subheader(f"Recommendations for User: `{selected_user}`")

user_history = ratings_df[ratings_df['user_id'] == selected_user].merge(movies_df, on='movie_id')
st.markdown(f"**Total Past Ratings:** {len(user_history)}")
with st.expander("View User's Rating History"):
    st.dataframe(user_history[['title', 'rating', 'year']].sort_values('rating', ascending=False))

recs, cold_start_msg = get_hybrid_recommendations(selected_user, model, trainset, movies_df, ratings_df)

if cold_start_msg:
    st.info(cold_start_msg)
    popular = movies_df.head(10)
    st.table(popular[['title', 'year']])
else:
    st.markdown("### Top 10 Hybrid Recommendations")
    
    for rank, (m_id, score) in enumerate(recs, 1):
        movie_info = movies_df[movies_df['movie_id'] == m_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            year = movie_info['year'].values[0]
            
            with st.container():
                st.markdown(f"#### {rank}. {title} ({int(year) if pd.notna(year) else 'N/A'})")
                st.markdown(f"**Predicted Score:** `{score:.2f} / 5.0`")
                
                explanation = explain_recommendation(m_id, selected_user, ratings_df, movies_df)
                st.caption(f"💡 *{explanation}*")
                st.divider()

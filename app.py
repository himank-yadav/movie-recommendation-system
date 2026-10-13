
import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Search for a movie and discover similar movies.")

# 1. Load Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Add GEMINI_API_KEY to your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# 2. Load your new dataset
@st.cache_data
def load_movies():
    df = pd.read_csv("movies.csv")

    required = [
        "movie_id",
        "movie_name",
        "year",
        "genre",
        "overview",
        "director",
        "cast"
    ]

    # Check whether all required columns exist
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in movies.csv: {missing}"
        )

    # Remove rows without essential movie information
    df = df.dropna(
        subset=["movie_id", "movie_name"]
    ).copy()

    # Fill optional text fields if they are empty
    text_columns = [
        "genre",
        "overview",
        "director",
        "cast"
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)

    df["year"] = df["year"].fillna("").astype(str)

    # Remove duplicate movies
    df = df.drop_duplicates(
        subset=["movie_id"]
    ).reset_index(drop=True)

    # Combine all useful information for embeddings
    df["text"] = (
        "Movie: " + df["movie_name"].astype(str)
        + ". Year: " + df["year"]
        + ". Genre: " + df["genre"]
        + ". Overview: " + df["overview"]
        + ". Director: " + df["director"]
        + ". Cast: " + df["cast"]
    )

    return df


# 3. Generate embeddings using Gemini Embedding 2
@st.cache_data(show_spinner="Generating movie embeddings...")
def generate_embeddings(movie_texts):
    contents = [
        types.Content(
            parts=[
                types.Part.from_text(text=text)
            ]
        )
        for text in movie_texts
    ]

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    vectors = [
        embedding.values
        for embedding in response.embeddings
    ]

    return np.asarray(vectors, dtype=np.float32)


# 4. Load data and create embeddings
try:
    movies = load_movies()

    if len(movies) < 2:
        st.error("Your dataset must contain at least two movies.")
        st.stop()

    embeddings = generate_embeddings(
        tuple(movies["text"].tolist())
    )

    if len(embeddings) != len(movies):
        st.error("Embedding count does not match movie count.")
        st.stop()

except Exception as e:
    st.error(f"Could not load movies or generate embeddings: {e}")
    st.stop()


# 5. Search and select a movie
st.subheader("Find a movie")

selected_movie = st.selectbox(
    "Search by movie name",
    options=movies["movie_name"].tolist(),
    index=None,
    placeholder="Type a movie name..."
)

number = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=min(10, len(movies) - 1),
    value=min(5, len(movies) - 1)
)


# 6. Recommend similar movies
if st.button("Find Recommendations", type="primary"):

    if selected_movie is None:
        st.warning("Please select a movie first.")

    else:
        # Find the selected movie's index
        movie_index = movies.index[
            movies["movie_name"] == selected_movie
        ][0]

        selected_vector = embeddings[movie_index].reshape(1, -1)

        # Calculate cosine similarity with every movie
        scores = cosine_similarity(
            selected_vector,
            embeddings
        )[0]

        # Sort from highest similarity to lowest
        ranked_indices = np.argsort(scores)[::-1]

        # Exclude the selected movie itself
        ranked_indices = [
            i for i in ranked_indices
            if i != movie_index
        ]

        recommendations = ranked_indices[:number]

        st.subheader(
            f"Movies similar to {selected_movie}"
        )

        for rank, idx in enumerate(recommendations, start=1):
            movie = movies.iloc[idx]

            with st.container(border=True):
                st.markdown(
                    f"### {rank}. {movie['movie_name']}"
                )

                st.write(f"**Year:** {movie['year']}")
                st.write(f"**Genre:** {movie['genre']}")
                st.write(f"**Director:** {movie['director']}")
                st.write(f"**Cast:** {movie['cast']}")
                st.write(f"**Overview:** {movie['overview']}")

                st.caption(
                    f"Cosine similarity: {scores[idx]:.3f}"
                )

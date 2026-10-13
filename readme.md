# 🎬 Movie Recommendation System

A content-based movie recommendation system built with **Python, Streamlit, Gemini Embedding 2, and Cosine Similarity**. Users can search for a movie in the catalog and discover similar movies based on their genres, plot descriptions, directors, cast, and release years.

## ✨ Features

- 🔍 Search and select movies from the catalog.
- 🤖 Generate semantic vector embeddings using Gemini Embedding 2.
- 🎯 Recommend similar movies using cosine similarity.
- 🎭 Use movie genres, overviews, directors, cast, and release years to calculate similarity.
- 🔢 Choose the number of recommendations to display.
- ⚡ Cache movie data and embeddings to reduce repeated API calls.
- 🖥️ Interactive web interface built with Streamlit.

## 🛠️ Tech Stack

- **Python** — Core programming language
- **Streamlit** — Web application interface
- **Google Gemini API** — Text embedding generation
- **Gemini Embedding 2** — Converts movie information into vector representations
- **NumPy** — Numerical array operations
- **Pandas** — Dataset processing
- **Scikit-learn** — Cosine similarity calculation
- **python-dotenv** — Environment variable management

## 📁 Project Structure

```text
movie-recommender/
│
├── app.py
├── movies.csv
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 📊 Dataset

The application uses a CSV file named `movies.csv`.

Required columns:

| Column | Description |
|---|---|
| `movie_id` | Unique movie identifier |
| `movie_name` | Movie title |
| `year` | Release year |
| `genre` | Movie genres |
| `overview` | Plot description |
| `director` | Movie director |
| `cast` | Main cast members |

The application combines these fields into a text description before generating embeddings.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd movie-recommender
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Create a `requirements.txt` file containing:

```text
streamlit
pandas
numpy
scikit-learn
google-genai
python-dotenv
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual API key.

**Important:** Never commit your API key or `.env` file to a public repository.

### 5. Add the dataset

Place your movie dataset in the project directory and name it `movies.csv`. Make sure it contains all the required columns.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 🧠 How It Works

1. **Load the dataset:** Pandas reads the movie catalog from `movies.csv`.
2. **Prepare text:** The movie title, year, genre, overview, director, and cast are combined into a single text description.
3. **Generate embeddings:** Gemini Embedding 2 converts each movie description into a numerical vector.
4. **Calculate similarity:** Cosine similarity compares the selected movie's vector with the vectors of all movies in the catalog.
5. **Rank recommendations:** The application sorts movies by similarity and excludes the selected movie.
6. **Display results:** Streamlit displays the recommended titles, years, genres, directors, cast, overviews, and similarity scores.

## 📐 Cosine Similarity

Cosine similarity measures the similarity between two vectors:

A higher score indicates that two movie descriptions are more similar in the embedding space. It is a similarity measure, not a probability that a user will enjoy a movie.

**Built with Python, Gemini Embedding 2, and Streamlit.**
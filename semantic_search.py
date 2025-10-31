import pandas as pd
import numpy as np
import re
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

# Load the GloVe model (you must have glove.6B.50d.word2vec.txt locally)
# You can download it from: https://nlp.stanford.edu/projects/glove/
# Then convert it to word2vec format using:
# gensim.scripts.glove2word2vec.glove2word2vec('glove.6B.50d.txt', 'glove.6B.50d.word2vec.txt')
glove_model_path = "glove.6B.50d.word2vec.txt"
glove_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=False)

# Load movies dataset
movies_df = pd.read_csv("movies.csv", encoding="utf-8")
movies_df = movies_df.dropna(subset=["title", "description"]).reset_index(drop=True)

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# Sentence embedding using GloVe
def sentence_vector(sentence):
    words = preprocess(sentence).split()
    word_vecs = [glove_model[word] for word in words if word in glove_model]
    if len(word_vecs) == 0:
        return np.zeros(glove_model.vector_size)
    return np.mean(word_vecs, axis=0)

# Precompute embeddings for all movie descriptions
movies_df["embedding"] = movies_df["description"].apply(sentence_vector)

def semantic_movie_search(query, top_k=5):
    query_vec = sentence_vector(query).reshape(1, -1)
    movie_vecs = np.vstack(movies_df["embedding"].values)
    similarities = cosine_similarity(query_vec, movie_vecs)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_movies = movies_df.iloc[top_indices][["title", "description"]]
    top_movies["similarity"] = similarities[top_indices]
    return top_movies

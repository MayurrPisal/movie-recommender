from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(df):
    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(df["combined"])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix
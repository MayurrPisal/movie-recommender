import numpy as np


def hybrid_recommend(movie_name, df, similarity_matrix, rules, top_n=4):
    if movie_name not in df["title"].values:
        return ["Movie not found in dataset"]

    idx = df[df["title"] == movie_name].index[0]

    # --- 1️⃣ Similarity Scores ---
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:20]

    final_scores = {}

    movie_features = set(df.iloc[idx]["combined"].split())

    # --- 2️⃣ Association Boost ---
    boost_dict = {}

    for _, row in rules.iterrows():
        if movie_features.issuperset(row["antecedents"]):
            for consequent in row["consequents"]:
                boost_dict[consequent] = row["confidence"]

    # --- 3️⃣ Combine Scores ---
    for i, sim_score in sim_scores:
        title = df.iloc[i]["title"]

        boost = 0
        movie_tokens = set(df.iloc[i]["combined"].split())

        # If recommended movie shares boosted token
        for token in movie_tokens:
            if token in boost_dict:
                boost = boost_dict[token]

        final_score = (0.7 * sim_score) + (0.3 * boost)

        final_scores[title] = final_score

    recommended = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [movie[0] for movie in recommended[:top_n]]
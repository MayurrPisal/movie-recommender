from ml.preprocessing import load_and_clean_data
from ml.similarity import build_similarity_matrix
from ml.association import build_association_rules
from ml.recommender import hybrid_recommend
from llm.agent import create_agent


print("Loading dataset...")
df = load_and_clean_data()

print("Building similarity matrix...")
similarity_matrix = build_similarity_matrix(df)

print("Building association rules...")
rules = build_association_rules(df)

print("Initializing AI agent...")
agent = create_agent(df, similarity_matrix, rules, hybrid_recommend)

print("\n🎬 Movie Recommendation System Ready!\n")

while True:
    query = input("Enter movie name (or type exit): ")

    if query.lower() == "exit":
        break

    response = agent(query)
    print(response)
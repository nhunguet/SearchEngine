from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 📝 Sample document corpus
documents = [
    "Machine learning is widely used in healthcare to analyze medical data.",
    "AI is transforming many industries, including healthcare and finance.",
    "Deep learning helps in image recognition and medical diagnosis.",
    "Natural language processing is used in AI chatbots and translation.",
    "AI and robotics are advancing in the field of surgery.",
]

# 🏷 Poorly formulated query vs. well-defined query
poor_query = "AI"
good_query = "Machine learning in healthcare"

# 🛠 Convert documents and queries into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents + [poor_query, good_query])

# 🔍 Compute cosine similarity between queries and documents
poor_query_vector = tfidf_matrix[-2]  # Vector for the poor query
good_query_vector = tfidf_matrix[-1]  # Vector for the well-defined query
doc_vectors = tfidf_matrix[:-2]  # Vectors for documents

poor_similarities = cosine_similarity(poor_query_vector, doc_vectors).flatten()
good_similarities = cosine_similarity(good_query_vector, doc_vectors).flatten()

# 📌 Display search results
def display_results(query, similarities):
    sorted_indices = np.argsort(similarities)[::-1]
    print(f"\n🔍 Query: {query}")
    print("📊 Search Results:")
    for idx in sorted_indices:
        print(f"- ({similarities[idx]:.4f}) {documents[idx]}")

# 🧐 Compare poor query vs. good query results
display_results(poor_query, poor_similarities)
display_results(good_query, good_similarities)

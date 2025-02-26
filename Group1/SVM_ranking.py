import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


documents = [
    "Machine learning is great for data analysis.",
    "Natural language processing helps computers understand text.",
    "I love deep learning algorithms for solving real-world problems.",
    "Data science is a field combining statistics and machine learning."
]

queries = [
    "data analysis tools",
    "understanding natural language",
    "machine learning applications",
    "statistics and data science"
]

# Relevance judgments (1 if relevant, 0 if not relevant)
relevance_judgments = [
    [1, 0, 1, 0],  
    [0, 1, 1, 0],  
    [1, 0, 1, 1],  
    [0, 1, 0, 1],  
]

# Convert documents and queries to tf-idf vectors
vectorizer = TfidfVectorizer()
X_documents = vectorizer.fit_transform(documents)
X_queries = vectorizer.transform(queries)


def rank_documents(query_idx):

    query_vec = X_queries[query_idx]
    
    # Calculate cosine similarity between query and documents
    sim_scores = np.dot(X_documents, query_vec.T).toarray().flatten()
    ranked_doc_idx = np.argsort(sim_scores)[::-1]  # Sort documents by similarity score
    return ranked_doc_idx, sim_scores[ranked_doc_idx]

# Create data base on relevance_judgments 
y_train = np.array(relevance_judgments).flatten()
X_train = np.tile(X_documents.toarray(), (len(queries), 1))  

# Train SVM with the expanded dataset
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Test with a specific query (query 0)
ranked_idx, scores = rank_documents(0)
print(f"Ranked Documents for Query 1:")
for idx in ranked_idx:
    print(f"Document: {documents[idx]}, Similarity Score: {scores[idx]}")

# Evaluate the model by comparing predicted relevance with actual relevance judgments
y_pred = svm_model.predict(X_train).reshape(len(queries), -1)
print(f"\nSVM Model Evaluation (Accuracy):")
print(f"Accuracy: {accuracy_score(y_train, y_pred.flatten()):.2f}")

from collections import Counter
from difflib import SequenceMatcher

def jaccard_similarity(doc1, doc2):
    """Calculates Jaccard similarity between two documents."""
    set1 = set(doc1.split())
    set2 = set(doc2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def similarity_score(doc1, doc2):
    """Calculates similarity score using Jaccard similarity and SequenceMatcher."""
    jaccard_sim = jaccard_similarity(doc1, doc2)
    seq_matcher = SequenceMatcher(None, doc1, doc2)
    seq_sim = seq_matcher.ratio()
    # Combine scores with a weight (adjust as needed)
    combined_score = (jaccard_sim * 0.6) + (seq_sim * 0.4) 
    return combined_score

def search_method(document, web_documents, threshold=0.5):
    """Finds nearly identical documents from a set of web documents."""
    results = []
    for web_doc in web_documents:
        score = similarity_score(document, web_doc)
        if score >= threshold:
            results.append((web_doc, score))
    return results

def discovery_method(web_documents, threshold=0.5):
    """Finds pairs of nearly identical documents within a set."""
    results = []
    for i in range(len(web_documents) - 1):
        for j in range(i + 1, len(web_documents)):
            doc1 = web_documents[i]
            doc2 = web_documents[j]
            score = similarity_score(doc1, doc2)
            if score >= threshold:
                results.append((doc1, doc2, score))
    return results

# Example Usage
document = "The quick brown fox jumps over the lazy dog."
web_documents = [
    "The quick brown fox jumps over the lazy dog.",  # Identical
    "The quick brown fox jumps.",  # Similar
    "A lazy dog sleeps under the tree.",  # Less similar
    "The fast red cat chases the mouse."  # Unrelated
]

# Search Method
similar_docs = search_method(document, web_documents)
print("Search Method Results:")
for doc, score in similar_docs:
    print(f"Document: {doc}, Score: {score:.2f}")

# Discovery Method
identical_pairs = discovery_method(web_documents)
print("\nDiscovery Method Results:")
for doc1, doc2, score in identical_pairs:
    print(f"Pair: ({doc1}, {doc2}), Score: {score:.2f}")


def find_largest_flat_area(bits):
    """
    Finds the largest flat area of the distribution in a sequence of bits.
    A flat area is a region with mostly 0s that has maximum 1s outside it.

    Args:
        bits: A list of 0s and 1s, where 1 represents a tag and 0 represents a non-tag token.

    Returns:
        A tuple (i, j) where i is the start index and j is the end index of the flat area.
    """
    n = len(bits)
    if n == 0:
        return 0, 0
    
    # Calculate prefix sums for O(1) range sum queries
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + bits[i]
    
    max_score = float('-inf')
    best_i = 0
    best_j = 0
    
    # For each possible range [i,j]
    for i in range(n):
        for j in range(i, n):
            # Count tags outside the range and zeros inside the range
            tags_outside = prefix_sum[i] + (prefix_sum[n] - prefix_sum[j + 1])
            zeros_inside = (j - i + 1) - (prefix_sum[j + 1] - prefix_sum[i])
            
            # Score is weighted sum of tags outside and zeros inside
            score = tags_outside + zeros_inside
            
            if score > max_score:
                max_score = score
                best_i = i
                best_j = j
    
    return best_i, best_j

# Example usage:
if __name__ == "__main__":
    test_cases = [
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    ]
    
    for bits in test_cases:
        i, j = find_largest_flat_area(bits)
        print(f"Input: {bits}")
        print(f"Largest flat area: i={i}, j={j}")
        print(f"Segment: {bits[i:j+1]}\n")

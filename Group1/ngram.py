import nltk
from nltk.util import ngrams
from collections import Counter

# Sample text
text = "to be or not to be"
words = text.split()

def generate_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

# Generate bigrams and trigrams using nltk
n = int(input("Enter the value of n: "))
n_grams = list(ngrams(words, n))

# Generate bigrams and trigrams manually
# manual_n_grams = generate_ngrams(words, n)

# Count frequency of n-grams
n_gram_counts = Counter(n_grams)

# Display results

print(f"{n}-grams:", n_grams)
print(f"{n}-gram Frequency:", n_gram_counts)

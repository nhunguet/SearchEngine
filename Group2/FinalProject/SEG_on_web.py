from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import re
import os
# Add pyngrok for tunneling
from pyngrok import ngrok
from underthesea import word_tokenize, text_normalize
# Additional imports for topic extraction and visualization
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_url_path='', static_folder='.')

def preprocess_text(text):
    """Preprocesses Vietnamese text using underthesea.

    Args:
        text: The input Vietnamese text.

    Returns:
        The preprocessed text.
    """

    # 1. Text Normalization:
    text = text.lower()
    text = text_normalize(text)  

    # 2. Word Segmentation (Tokenization):
    tokens = word_tokenize(text)  

    # 3. (Optional) Remove Stopwords, punctuation, special characters:
    # You may need to define a list of stopwords based on your needs and remove them here
    stop_words = [
        "tôi", "bạn", "chúng tôi", "họ", "nó", "ông", "bà", "cô", "chúng ta", "hắn", "mình",
        "ở", "trong", "ngoài", "trên", "dưới", "với", "đến", "từ",
        "và", "hoặc", "nếu", "vì", "nên", "rồi", "mà", "khi", "sau khi",
        "rất", "cũng", "chỉ", "đã", "đang", "sẽ", "nữa", "mới", "lại", "thế",
        "à", "ơi", "nhé", "hả", "có", "phải", "vậy", "thôi", "được"
    ]
    # stop_words = [] # Remove this line if you want to use the stop words
    tokens = [word for word in tokens if word not in stop_words]

    # 4. Join tokens back into a string:
    preprocessed_text = " ".join(tokens)
    preprocessed_text = preprocessed_text.replace("cc", "con chim")

    return preprocessed_text

# Helper function to compute text similarity
def compute_text_similarity(text1, text2):
    """Compute Jaccard similarity between two text strings."""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    # Handle empty sets
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    
    # Compute Jaccard similarity: intersection over union
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    
    return intersection / union if union > 0 else 0.0

# Topic extraction function - updated with similarity deduplication
def extract_topics(comments_list, top_n=5):
    """Extract top topics/keywords from a list of comments using longer n-grams with deduplication."""
    if not comments_list:
        return []
    len_comments = len(comments_list)
    top_n = top_n if top_n <= len_comments else len_comments
    # Similarity threshold - adjust based on desired strictness
    SIMILARITY_THRESHOLD = 0.6  # Topics with similarity above this are considered duplicates
        
    # Create a TF-IDF vectorizer with longer n-grams (5-10 word phrases)
    topic_vectorizer = TfidfVectorizer(
        max_df=0.95,      # Ignore terms that appear in >95% of documents
        min_df=1,         # Lower threshold to capture more phrases
        max_features=200, # Only consider the top 200 features
        stop_words=stop_words,
        ngram_range=(5, 10)  # Extract longer phrases (5-10 words) for Vietnamese
    )
    
    # Fit and transform the comments
    try:
        tfidf_matrix = topic_vectorizer.fit_transform(comments_list)
        feature_names = topic_vectorizer.get_feature_names_out()
        
        # Sum up the TF-IDF scores for each term across all documents
        tfidf_sums = tfidf_matrix.sum(axis=0).A1
        
        # Get indices sorted by TF-IDF score (descending)
        sorted_indices = tfidf_sums.argsort()[::-1]
        
        # Find diverse topics using similarity threshold
        diverse_topics = []
        
        # Process potential topics in order of TF-IDF score
        for idx in sorted_indices:
            term = feature_names[idx]
            score = tfidf_sums[idx]
            
            # Skip terms with zero score
            if score <= 0:
                continue
                
            # Check if this term is too similar to already selected terms
            is_duplicate = False
            for existing_term, _ in diverse_topics:
                similarity = compute_text_similarity(term, existing_term)
                if similarity > SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
            
            # If not a duplicate, add it to our diverse topics
            if not is_duplicate:
                diverse_topics.append((term, score))
            
            # Stop when we have enough topics
            if len(diverse_topics) >= top_n:
                break
        # if len_comments < len(diverse_topics):
        #     return diverse_topics[:len_comments]
        return diverse_topics
        
    except Exception as e:
        print(f"Error in TF-IDF extraction: {e}")
        # Fallback to n-gram extraction with deduplication
        try:
            from nltk.util import ngrams
            from nltk.tokenize import word_tokenize
            from collections import Counter
            
            all_ngrams = []
            for comment in comments_list:
                words = word_tokenize(comment.lower())
                # Extract 5-10 grams for longer phrases
                for n in range(5, 11):
                    if len(words) >= n:
                        all_ngrams.extend([' '.join(gram) for gram in ngrams(words, n)])
            
            # Count ngrams
            ngram_counts = Counter(all_ngrams)
            
            # Filter out ngrams with stop words
            for word in stop_words:
                for key in list(ngram_counts.keys()):
                    if word in key.split():
                        ngram_counts[key] = 0
            
            # Select diverse topics
            diverse_topics = []
            for ngram, count in ngram_counts.most_common(top_n * 3):  # Get more candidates
                if count <= 0:
                    continue
                    
                # Check if this ngram is too similar to already selected ngrams
                is_duplicate = False
                for existing_ngram, _ in diverse_topics:
                    similarity = compute_text_similarity(ngram, existing_ngram)
                    if similarity > SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        break
                
                # If not a duplicate, add it
                if not is_duplicate:
                    diverse_topics.append((ngram, count))
                
                # Stop when we have enough topics
                if len(diverse_topics) >= top_n:
                    break
                    
            return diverse_topics
            
        except Exception as e:
            print(f"Error in n-gram fallback: {e}")
            # Ultimate fallback - extract phrases with deduplication
            all_text = " ".join(comments_list)
            words = all_text.lower().split()
            
            # Extract phrases
            phrases = []
            scores = []
            
            for i in range(len(words) - 5):
                if i < len(words) - 9:
                    phrases.append(' '.join(words[i:i+10]))  # 10-word phrases
                    scores.append(10)  # Score based on length
                elif i < len(words) - 4:
                    phrases.append(' '.join(words[i:i+5]))   # 5-word phrases
                    scores.append(5)   # Score based on length
            
            # Select diverse phrases
            diverse_topics = []
            for i, phrase in enumerate(phrases):
                # Skip phrases with stop words
                contains_stop_word = False
                for word in stop_words:
                    if word in phrase.split():
                        contains_stop_word = True
                        break
                
                if contains_stop_word:
                    continue
                
                # Check similarity with existing topics
                is_duplicate = False
                for existing_phrase, _ in diverse_topics:
                    similarity = compute_text_similarity(phrase, existing_phrase)
                    if similarity > SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        break
                
                # If not a duplicate, add it
                if not is_duplicate:
                    diverse_topics.append((phrase, scores[i]))
                
                # Stop when we have enough topics
                if len(diverse_topics) >= top_n:
                    break
            
            return diverse_topics

# Create bar chart for top topics
def create_topic_barchart(topics, title="Top Topics"):
    if not topics:
        return None
    
    # Extract terms and scores
    terms, scores = zip(*topics) if topics else ([], [])
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(terms))
    ax.barh(y_pos, scores, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Score')
    ax.set_title(title)
    plt.tight_layout()
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

# Create a placeholder image for no data
def create_no_data_image(title="No Data Available"):
    """Generate a placeholder image when no data is available."""
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, "No data available", 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=18, color='gray')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    
    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

# Define stop words list globally
stop_words = [
    "tôi", "bạn", "chúng tôi", "họ", "nó", "ông", "bà", "cô", "chúng ta", "hắn", "mình",
    "ở", "trong", "ngoài", "trên", "dưới", "với", "đến", "từ",
    "và", "hoặc", "nếu", "vì", "nên", "rồi", "mà", "khi", "sau khi",
    "rất", "cũng", "chỉ", "đã", "đang", "sẽ", "nữa", "mới", "lại", "thế",
    "à", "ơi", "nhé", "hả", "có", "phải", "vậy", "thôi", "được"
]
# stop_words = []  # Remove this line if you want to use the stop words
# Try to load the model and vectorizer
model_loaded = False
try:
    # Try to load both model and vectorizer
    with open('best_model_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer_v2.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    model_loaded = True
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Model or vectorizer file not found. Using a dummy model for demonstration.")
    
    # Dummy vectorizer and model for demonstration
    class DummyModel:
        def __init__(self):
            # Define the class labels and their indices for consistent mapping
            # Only two classes - Negative and Positive
            self.classes_ = ["Negative", "Positive"]
        
        def predict(self, text):
            if not isinstance(text, str):
                text = str(text)
            if not text:
                return "No input"
            elif "good" in text.lower():
                return "Positive"
            else:
                # Default to Negative for other cases
                return "Negative"
        
        # Add a method to provide probability estimates
        def predict_proba(self, text):
            if not isinstance(text, list):
                text = [text]
            results = []
            for item in text:
                if not isinstance(item, str):
                    item = str(item)
                
                # Create probabilities for two classes: [Negative, Positive]
                if not item:
                    # No input: equal probabilities
                    results.append([0.5, 0.5])  # [Negative, Positive]
                elif "good" in item.lower():
                    # Positive: higher probability for Positive (index 1)
                    results.append([0.2, 0.8])   # [Negative, Positive]
                else:
                    # Negative: higher probability for Negative (index 0)
                    results.append([0.8, 0.2])   # [Negative, Positive]
            return results
    
    model = DummyModel()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_status=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment', '')
    
    if not comment:
        return jsonify({'label': 'Please enter a comment', 'confidence': 0})
    
    # Make prediction
    try:
        # Process the comment
        processed_comment = preprocess_text(comment)
        
        if model_loaded:
            # Process for real model
            vectorized_comment = tfidf.transform([processed_comment])
            
            # Get label
            label = model.predict(vectorized_comment)[0]
            
            # Try to get probability/confidence
            try:
                # If the model supports predict_proba
                proba = model.predict_proba(vectorized_comment)[0]
                # Get the confidence of the predicted class
                class_index = model.classes_.tolist().index(label)
                confidence = round(proba[class_index] * 100, 2)
            except Exception as e:
                print(f"Error getting confidence: {e}")
                confidence = "N/A"
        else:
            # Process for dummy model
            # Use raw comment for dummy model as it's designed for English keywords
            label = model.predict(comment)
            
            # Use consistent approach for confidence calculation
            proba = model.predict_proba(comment)[0]
            # There are only two classes now: "Negative" (0) and "Positive" (1)
            class_index = 0 if label == "Negative" else 1
            confidence = round(proba[class_index] * 100, 2)
        
        return jsonify({'label': label, 'confidence': confidence})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'label': f'Error: {str(e)}', 'confidence': 0})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    batch_text = data.get('batch_text', '')
    
    if not batch_text:
        return jsonify({
            'error': 'Please enter comments',
            'positive_count': 0,
            'negative_count': 0
        })
    
    # Split comments using the separator
    comments = re.split(r'\s*&&&&\s*', batch_text.strip())
    comments = [comment for comment in comments if comment.strip()]
    
    if not comments:
        return jsonify({
            'error': 'No valid comments found',
            'positive_count': 0,
            'negative_count': 0
        })
    
    positive_comments = []
    negative_comments = []
    results = []
    
    # Process each comment
    for comment in comments:
        try:
            processed_comment = preprocess_text(comment)
            
            if model_loaded:
                # Real model
                vectorized_comment = tfidf.transform([processed_comment])
                label = model.predict(vectorized_comment)[0]
                
                try:
                    # Get confidence if available
                    proba = model.predict_proba(vectorized_comment)[0]
                    class_index = model.classes_.tolist().index(label)
                    confidence = round(proba[class_index] * 100, 2)
                except:
                    confidence = "N/A"
            else:
                # Dummy model
                label = model.predict(comment)
                proba = model.predict_proba(comment)[0]
                class_index = 0 if label == "negative" else 1
                confidence = round(proba[class_index] * 100, 2)
                
            # Add to appropriate list
            if label == "positive":
                positive_comments.append(processed_comment)
            elif label == "negative":
                negative_comments.append(processed_comment)
            else:
                pass
                
            # Add to results
            results.append({
                'comment': comment,
                'label': label,
                'confidence': confidence
            })
                
        except Exception as e:
            print(f"Error processing comment: {e}")
            # Skip problematic comments
            continue
            
    # Extract topics from positive and negative comments
    positive_topics = extract_topics(positive_comments)
    negative_topics = extract_topics(negative_comments)
    
    # Generate visualizations - handle empty comment cases
    if positive_comments:
        positive_barchart = create_topic_barchart(positive_topics, "Top Topics in Positive Comments")
    else:
        positive_barchart = create_no_data_image("No Topics for Positive Comments")
    
    if negative_comments:
        negative_barchart = create_topic_barchart(negative_topics, "Top Topics in Negative Comments")
    else:
        negative_barchart = create_no_data_image("No Topics for Negative Comments")
    
    return jsonify({
        'results': results,
        'positive_count': len(positive_comments),
        'negative_count': len(negative_comments),
        'positive_topics': [{'term': term, 'score': float(score)} for term, score in positive_topics],
        'negative_topics': [{'term': term, 'score': float(score)} for term, score in negative_topics],
        'positive_barchart': positive_barchart,
        'negative_barchart': negative_barchart
    })

if __name__ == '__main__':
    # Try to set up ngrok tunnel for external access, but make it optional
    try:
        public_url = ngrok.connect(5000)
        print(f"Public URL: {public_url}")
        print("Your app is publicly accessible at the above URL")
    except Exception as e:
        print(f"Ngrok Error: {str(e)}")
        print("Continuing without ngrok. App will only be available locally at http://127.0.0.1:5000")
        print("To access from other devices on your network, use your local IP address")
        print("If you need public access, ensure no other ngrok sessions are running")

    # Run the Flask app - adding host parameter to make it accessible on the local network
    app.run(debug=False, host='0.0.0.0')
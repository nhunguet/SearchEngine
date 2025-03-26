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
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vietnamese Comment Sentiment Analysis</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap">
    <style>
        :root {
            /* New tech color palette with pale neon green and dark grays */
            --neon-green: #a8ff80;
            --neon-green-light: #c5ffaa;
            --neon-green-dark: #7ad65c;
            --dark-gray: #2a2a2a;
            --darker-gray: #1a1a1a;
            --medium-gray: #3f3f3f;
            --light-gray: #e0e0e0;
            --off-white: #f8f8f8;
            
            /* Extended palette */
            --primary-color: var(--neon-green);
            --primary-light: var(--neon-green-light);
            --primary-dark: var(--neon-green-dark);
            --secondary-color: #80ffea;
            --success-color: #b8ffb2; /* Lighter green for positive */
            --danger-color: #ffb3b3; /* Lighter red for negative */
            --warning-color: #ffe999;
            --background-color: var(--off-white);
            --card-color: #ffffff;
            --text-color: var(--dark-gray);
            --text-muted: #6e6e6e;
            --border-radius: 0.5rem;
            --box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --transition: all 0.3s ease;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--dark-gray), var(--darker-gray));
            color: white;
            padding: 3rem 0;
            text-align: center;
            position: relative;
            overflow: hidden;
            margin-bottom: 2rem;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMCAwIEwxMDAgMTAwIE0xMDAgMCBMMCAxMDAiIHN0cm9rZT0icmdiYSgxNjgsMjU1LDEyOCwwLjEpIiBzdHJva2Utd2lkdGg9IjEiLz48L3N2Zz4=');
            opacity: 0.2;
        }
        
        .header-content {
            position: relative;
            z-index: 1;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 800;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
            opacity: 0.9;
        }
        
        /* Container */
        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1.5rem;
        }
        
        /* Main content */
        .main-content {
            position: relative;
            z-index: 10;
            margin-bottom: 3rem;
        }
        
        /* Card styles */
        .card {
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            overflow: hidden;
            margin-bottom: 2rem;
            transition: var(--transition);
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .card-header {
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: var(--dark-gray);
            color: white;
        }
        
        .card-header h2 {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .card-header h2 i {
            color: var(--neon-green);
        }
        
        .card-body {
            padding: 1.5rem;
        }
        
        /* Grid layout */
        .grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 1.5rem;
        }
        
        .col-12 {
            grid-column: span 12;
        }
        
        .col-8 {
            grid-column: span 8;
        }
        
        .col-6 {
            grid-column: span 6;
        }
        
        .col-4 {
            grid-column: span 4;
        }
        
        /* Form elements */
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--text-color);
        }
        
        textarea {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 1px solid #e2e8f0;
            border-radius: var(--border-radius);
            font-family: inherit;
            font-size: 1rem;
            resize: vertical;
            min-height: 120px;
            transition: var(--transition);
            background-color: var(--off-white);
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--neon-green);
            box-shadow: 0 0 0 3px rgba(168, 255, 128, 0.25);
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.75rem 1.5rem;
            border-radius: var(--border-radius);
            font-weight: 500;
            font-size: 1rem;
            cursor: pointer;
            transition: var(--transition);
            border: none;
            gap: 0.5rem;
        }
        
        .btn-primary {
            background-color: var(--dark-gray);
            color: white;
            border: 2px solid var(--neon-green);
        }
        
        .btn-primary:hover {
            background-color: var(--darker-gray);
            box-shadow: 0 0 10px rgba(168, 255, 128, 0.5);
        }
        
        .btn-outline {
            background-color: transparent;
            border: 1px solid var(--neon-green);
            color: var(--dark-gray);
        }
        
        .btn-outline:hover {
            background-color: var(--neon-green-light);
            color: var(--dark-gray);
        }
        
        /* Result display */
        .result {
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: var(--border-radius);
            background-color: var(--off-white);
            border-left: 4px solid var(--neon-green);
            display: none;
        }
        
        .result-label {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }
        
        .result-value {
            font-size: 1.25rem;
            font-weight: 500;
            word-break: break-word;
        }
        
        .confidence-container {
            margin-top: 1rem;
        }
        
        .confidence-label {
            font-weight: 500;
            margin-bottom: 0.25rem;
            color: var(--text-muted);
        }
        
        .confidence-value {
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .confidence-bar-container {
            height: 8px;
            background-color: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .confidence-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--neon-green), var(--neon-green-light));
            width: 0;
            transition: width 0.5s ease;
        }
        
        /* Prediction log */
        .prediction-log {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .log-entries {
            flex-grow: 1;
            overflow-y: auto;
            max-height: 400px;
        }
        
        .log-entry {
            padding: 1rem;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            transition: var(--transition);
        }
        
        .log-entry:hover {
            background-color: rgba(168, 255, 128, 0.05);
        }
        
        .log-comment {
            margin-bottom: 0.5rem;
            word-break: break-word;
            line-height: 1.5;
        }
        
        .log-result {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .log-timestamp {
            font-size: 0.875rem;
            color: var(--text-muted);
        }
        
        .log-label {
            font-weight: 500;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
        }
        
        .log-label.Positive {
            background-color: rgba(184, 255, 178, 0.3);
            color: #166534;
        }
        
        .log-label.Negative {
            background-color: rgba(255, 179, 179, 0.3);
            color: #991b1b;
        }
        
        .empty-log {
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 2rem;
        }
        
        /* Sentiment counts */
        .sentiment-counts {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .count-card {
            padding: 1.5rem;
            border-radius: var(--border-radius);
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: var(--box-shadow);
            position: relative;
            overflow: hidden;
            z-index: 1;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .count-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0));
            z-index: -1;
        }
        
        .count-card div:first-child {
            font-weight: 500;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .count-card div:first-child i {
            font-size: 1.25rem;
        }
        
        .count-card div:last-child {
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .positive-count {
            background-color: var(--success-color);
            color: #166534;
        }
        
        .negative-count {
            background-color: var(--danger-color);
            color: #991b1b;
        }
        
        /* Topic lists */
        .topic-list {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }
        
        .topic-item {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            transition: var(--transition);
        }
        
        .topic-item:hover {
            background-color: rgba(168, 255, 128, 0.05);
        }
        
        .topic-item:last-child {
            border-bottom: none;
        }
        
        .topic-content {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
        }
        
        .topic-term {
            font-weight: 500;
            flex-grow: 1;
            word-break: break-word;
            line-height: 1.5;
        }
        
        .topic-score {
            color: var(--dark-gray);
            white-space: nowrap;
            font-variant-numeric: tabular-nums;
            font-weight: 600;
            background-color: rgba(168, 255, 128, 0.2);
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            background-color: var(--off-white);
            border-radius: 0.5rem 0.5rem 0 0;
            overflow: hidden;
        }
        
        .tab {
            padding: 1rem 1.5rem;
            cursor: pointer;
            font-weight: 500;
            color: var(--text-muted);
            transition: var(--transition);
            position: relative;
        }
        
        .tab:hover {
            color: var(--dark-gray);
            background-color: rgba(168, 255, 128, 0.1);
        }
        
        .tab.active {
            color: var(--dark-gray);
            background-color: white;
        }
        
        .tab.active::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background-color: var(--neon-green);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Chart container */
        .chart-container {
            background-color: white;
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--box-shadow);
            margin-bottom: 1.5rem;
            position: relative;
        }
        
        .chart-container h3 {
            margin-top: 0;
            margin-bottom: 1.5rem;
            color: var(--dark-gray);
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .viz-image {
            width: 100%;
            height: auto;
            border-radius: var(--border-radius);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: var(--transition);
        }
        
        .viz-image:hover {
            transform: scale(1.01);
        }
        
        /* Team section */
        .team-section {
            padding: 4rem 0;
            background: linear-gradient(135deg, var(--off-white), #f1f5f9);
            position: relative;
            overflow: hidden;
        }
        
        .team-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMCAwIEwxMDAgMTAwIE0xMDAgMCBMMCAxMDAiIHN0cm9rZT0icmdiYSgxNjgsIDI1NSwgMTI4LCAwLjEpIiBzdHJva2Utd2lkdGg9IjEiLz48L3N2Zz4=');
            opacity: 0.5;
        }
        
        .team-header {
            text-align: center;
            margin-bottom: 3rem;
            position: relative;
            z-index: 1;
        }
        
        .team-header h2 {
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--dark-gray);
            margin-bottom: 1rem;
        }
        
        .team-header p {
            color: var(--text-muted);
            max-width: 600px;
            margin: 0 auto;
            font-size: 1.1rem;
        }
        
        .team-members {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 2rem;
            position: relative;
            z-index: 1;
        }
        
        .team-member {
            background-color: white;
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--box-shadow);
            transition: var(--transition);
            text-align: center;
            position: relative;
        }
        
        .team-member:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        
        .team-member::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--neon-green), var(--neon-green-light));
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }
        
        .team-member:hover::after {
            transform: scaleX(1);
        }
        
        .member-image {
            width: 100%;
            height: 220px;
            overflow: hidden;
            position: relative;
        }
        
        .member-image::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(to bottom, rgba(0,0,0,0) 70%, rgba(0,0,0,0.7) 100%);
            z-index: 1;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .team-member:hover .member-image::before {
            opacity: 1;
        }
        
        .member-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }
        
        .team-member:hover .member-image img {
            transform: scale(1.05);
        }
        
        .member-info {
            padding: 1.5rem;
            position: relative;
            background-color: white;
        }
        
        .member-name {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            color: var(--dark-gray);
        }
        
        .member-title {
            font-size: 0.9rem;
            color: var(--text-muted);
            white-space: pre-line;
            line-height: 1.5;
        }
        
        /* Footer */
        .footer {
            background-color: var(--darker-gray);
            color: white;
            padding: 3rem 0;
            text-align: center;
        }
        
        .footer-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1.5rem;
        }
        
        .footer-logo {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--neon-green);
        }
        
        .footer-links {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .footer-link {
            color: rgba(255, 255, 255, 0.8);
            text-decoration: none;
            transition: var(--transition);
        }
        
        .footer-link:hover {
            color: var(--neon-green);
        }
        
        .footer-copyright {
            opacity: 0.8;
            font-size: 0.9rem;
            margin-top: 1rem;
        }
        
        /* Loading spinner */
        .spinner {
            display: inline-block;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(168, 255, 128, 0.3);
            border-radius: 50%;
            border-top-color: var(--neon-green);
            animation: spin 1s ease-in-out infinite;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Model status badge */
        .model-status {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
            margin-top: 1.5rem;
        }
        
        .model-status.success {
            background-color: rgba(184, 255, 178, 0.2);
            color: #166534;
        }
        
        .model-status.warning {
            background-color: rgba(255, 233, 153, 0.2);
            color: #92400e;
        }
        
        .model-status i {
            margin-right: 0.5rem;
        }
        
        /* Visualization enhancements */
        .viz-card {
            background-color: white;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            overflow: hidden;
            transition: var(--transition);
            height: 100%;
            display: flex;
            flex-direction: column;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .viz-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-5px);
        }
        
        .viz-card-header {
            padding: 1.25rem;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            background-color: var(--dark-gray);
            color: white;
        }
        
        .viz-card-header h3 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .viz-card-header h3 i {
            color: var(--neon-green);
        }
        
        .viz-card-body {
            padding: 1.5rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        
        .viz-image-container {
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            background-color: var(--off-white);
            border-radius: var(--border-radius);
            border: 1px solid rgba(168, 255, 128, 0.2);
        }
        
        /* Data summary */
        .data-summary {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }
        
        .data-stat {
            flex: 1;
            min-width: 150px;
            padding: 1rem;
            background-color: white;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            text-align: center;
            border-top: 4px solid var(--neon-green);
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--dark-gray);
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Responsive adjustments */
        @media (max-width: 992px) {
            .col-lg-6 {
                grid-column: span 6;
            }
            
            .col-lg-12 {
                grid-column: span 12;
            }
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }
            
            .col-md-12 {
                grid-column: span 12;
            }
            
            .sentiment-counts {
                grid-template-columns: 1fr;
            }
            
            .team-members {
                grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            }
            
            .member-image {
                height: 160px;
            }
        }
        
        @media (max-width: 576px) {
            .header {
                padding: 2rem 0;
            }
            
            .header h1 {
                font-size: 1.75rem;
            }
            
            .card-body {
                padding: 1.25rem;
            }
            
            .team-members {
                grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            }
            
            .member-image {
                height: 140px;
            }
            
            .tabs {
                flex-direction: column;
            }
            
            .tab {
                text-align: center;
            }
        }
        
        /* Utility classes */
        .mb-4 {
            margin-bottom: 1rem;
        }
        
        .mb-8 {
            margin-bottom: 2rem;
        }
        
        .text-center {
            text-align: center;
        }
        
        .empty-list {
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 1.5rem;
        }
    </style>
</head>
<body>
    <!-- Header Section -->
    <header class="header">
        <div class="header-content">
            <h1>Vietnamese Comment Sentiment Analysis</h1>
            <p>Analyze product reviews and comments to determine sentiment and extract key topics</p>
        </div>
    </header>
    
    <!-- Main Content -->
    <div class="container">
        <div class="main-content">
            <!-- Single Comment Analysis -->
            <div class="grid">
                <div class="col-8 col-lg-12">
                    <div class="card">
                        <div class="card-header">
                            <h2><i class="fas fa-comment-alt"></i> Single Comment Analysis</h2>
                        </div>
                        <div class="card-body">
                            <div class="form-group">
                                <label for="comment">Enter your comment:</label>
                                <textarea id="comment" placeholder="Type your comment in Vietnamese here..."></textarea>
                            </div>
                            
                            <button id="submit-btn" class="btn btn-primary">
                                <i class="fas fa-search"></i> Analyze Sentiment
                            </button>
                            
                            <div class="result" id="result">
                                <div class="result-label">Classification:</div>
                                <div class="result-value" id="result-value">-</div>
                                <div class="confidence-container">
                                    <div class="confidence-label">Confidence:</div>
                                    <div class="confidence-value" id="confidence-value">-</div>
                                    <div class="confidence-bar-container">
                                        <div class="confidence-bar" id="confidence-bar"></div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="model-status {% if model_status %}success{% else %}warning{% endif %}">
                                {% if model_status %}
                                    <i class="fas fa-check-circle"></i> Model loaded successfully
                                {% else %}
                                    <i class="fas fa-exclamation-triangle"></i> Using demo model (model file not found)
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-4 col-lg-12">
                    <div class="card">
                        <div class="card-header">
                            <h2><i class="fas fa-history"></i> Recent Predictions</h2>
                        </div>
                        <div class="card-body">
                            <div class="prediction-log">
                                <div class="log-entries" id="log-entries">
                                    <div class="empty-log">No predictions yet</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Batch Analysis -->
            <div class="card">
                <div class="card-header">
                    <h2><i class="fas fa-layer-group"></i> Batch Comment Analysis</h2>
                </div>
                <div class="card-body">
                    <p class="mb-4">Enter multiple comments separated by "&&&&" for batch processing:</p>
                    
                    <div class="form-group">
                        <textarea id="batch-comments" rows="6" placeholder="Comment 1 &&&& Comment 2 &&&& Comment 3..."></textarea>
                    </div>
                    
                    <button id="batch-submit-btn" class="btn btn-primary">
                        <i class="fas fa-cogs"></i> Process Batch
                    </button>
                    
                    <div id="batch-loading" style="display: none; margin-top: 1rem;">
                        <div class="spinner"></div> Processing comments...
                    </div>
                </div>
            </div>
            
            <!-- Visualization Container -->
            <div class="card" id="viz-container" style="display: none;">
                <div class="card-header">
                    <h2><i class="fas fa-chart-bar"></i> Analysis Results</h2>
                </div>
                <div class="card-body">
                    <div class="tabs">
                        <div class="tab active" data-tab="overview">Overview</div>
                        <div class="tab" data-tab="topics">Topic Visualization</div>
                    </div>
                    
                    <!-- Overview Tab -->
                    <div class="tab-content active" id="overview-tab">
                        <div class="sentiment-counts">
                            <div class="count-card positive-count">
                                <div><i class="fas fa-smile"></i> Positive Comments</div>
                                <div id="positive-count">0</div>
                            </div>
                            <div class="count-card negative-count">
                                <div><i class="fas fa-frown"></i> Negative Comments</div>
                                <div id="negative-count">0</div>
                            </div>
                        </div>
                        
                        <div class="grid">
                            <div class="col-6 col-md-12">
                                <div class="viz-card">
                                    <div class="viz-card-header">
                                        <h3><i class="fas fa-thumbs-up"></i> Top Topics in Positive Comments</h3>
                                    </div>
                                    <div class="viz-card-body">
                                        <ul class="topic-list" id="positive-topics">
                                            <li class="empty-list">No data available</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                            <div class="col-6 col-md-12">
                                <div class="viz-card">
                                    <div class="viz-card-header">
                                        <h3><i class="fas fa-thumbs-down"></i> Top Topics in Negative Comments</h3>
                                    </div>
                                    <div class="viz-card-body">
                                        <ul class="topic-list" id="negative-topics">
                                            <li class="empty-list">No data available</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Topic Charts Tab -->
                    <div class="tab-content" id="topics-tab">
                        <div class="grid">
                            <div class="col-6 col-md-12">
                                <div class="viz-card">
                                    <div class="viz-card-header">
                                        <h3><i class="fas fa-chart-bar"></i> Positive Topics</h3>
                                    </div>
                                    <div class="viz-card-body">
                                        <div class="viz-image-container">
                                            <img id="positive-barchart" class="viz-image" src="/placeholder.svg" alt="Positive Topics Chart">
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-6 col-md-12">
                                <div class="viz-card">
                                    <div class="viz-card-header">
                                        <h3><i class="fas fa-chart-bar"></i> Negative Topics</h3>
                                    </div>
                                    <div class="viz-card-body">
                                        <div class="viz-image-container">
                                            <img id="negative-barchart" class="viz-image" src="/placeholder.svg" alt="Negative Topics Chart">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Team Section -->
    <section class="team-section">
        <div class="container">
            <div class="team-header">
                <h2>Our Team</h2>
                <p>Meet the talented individuals behind this Vietnamese sentiment analysis project</p>
            </div>
            
            <div class="team-members">
                <div class="team-member">
                    <div class="member-image">
                        <img src="/images/team1.jpg" alt="Nguyen Van Phong" onerror="this.src='https://ui-avatars.com/api/?name=Nguyen+Van+Phong&background=2a2a2a&color=a8ff80'">
                    </div>
                    <div class="member-info">
                        <div class="member-name">Nguyen Van Phong</div>
                        <div class="member-title">Team Leader\nData Labeling\nModel Training</div>
                    </div>
                </div>
                
                <div class="team-member">
                    <div class="member-image">
                        <img src="/images/team2.jpg" alt="Tran Trung Nhan" onerror="this.src='https://ui-avatars.com/api/?name=Tran+Trung+Nhan&background=2a2a2a&color=a8ff80'">
                    </div>
                    <div class="member-info">
                        <div class="member-name">Tran Trung Nhan</div>
                        <div class="member-title">URL Collection\nData Labeling</div>
                    </div>
                </div>
                
                <div class="team-member">
                    <div class="member-image">
                        <img src="/images/team3.jpg" alt="Huynh Ngoc Nhu Quynh" onerror="this.src='https://ui-avatars.com/api/?name=Huynh+Ngoc+Nhu+Quynh&background=2a2a2a&color=a8ff80'">
                    </div>
                    <div class="member-info">
                        <div class="member-name">Huynh Ngoc Nhu Quynh</div>
                        <div class="member-title">URL Collection\nData Crawling\nData Labeling</div>
                    </div>
                </div>
                
                <div class="team-member">
                    <div class="member-image">
                        <img src="/images/team4.jpg" alt="Huynh Anh Phuong" onerror="this.src='https://ui-avatars.com/api/?name=Huynh+Anh+Phuong&background=2a2a2a&color=a8ff80'">
                    </div>
                    <div class="member-info">
                        <div class="member-name">Huynh Anh Phuong</div>
                        <div class="member-title">URL Collection\nData Labeling</div>
                    </div>
                </div>
                
                <div class="team-member">
                    <div class="member-image">
                        <img src="/images/team5.jpg" alt="Dao Anh Khoa" onerror="this.src='https://ui-avatars.com/api/?name=Dao+Anh+Khoa&background=2a2a2a&color=a8ff80'">
                    </div>
                    <div class="member-info">
                        <div class="member-name">Dao Anh Khoa</div>
                        <div class="member-title">URL Collection\nData Labeling</div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    
    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <div class="footer-logo">Vietnamese Sentiment Analysis</div>
                <div class="footer-links">
                    <a href="#" class="footer-link">Home</a>
                    <a href="#" class="footer-link">About</a>
                    <a href="#" class="footer-link">Documentation</a>
                    <a href="#" class="footer-link">Contact</a>
                </div>
                <div class="footer-copyright">
                    &copy; 2023 Vietnamese Comment Sentiment Analysis. All rights reserved.
                </div>
            </div>
        </div>
    </footer>

    <script>
        // Store prediction history
        const predictionLog = [];
        
        document.addEventListener('DOMContentLoaded', () => {
            // Set up tab functionality
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs and contents
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    tab.classList.add('active');
                    
                    // Show corresponding content
                    const tabId = tab.getAttribute('data-tab');
                    document.getElementById(`${tabId}-tab`).classList.add('active');
                });
            });
        });

        document.getElementById('submit-btn').addEventListener('click', async () => {
            const comment = document.getElementById('comment').value.trim();
            const resultDiv = document.getElementById('result');
            const resultValue = document.getElementById('result-value');
            const confidenceValue = document.getElementById('confidence-value');
            const confidenceBar = document.getElementById('confidence-bar');
            
            if (!comment) {
                resultValue.textContent = 'Please enter a comment';
                confidenceValue.textContent = '-';
                confidenceBar.style.width = '0%';
                resultDiv.style.display = 'block';
                return;
            }
            
            // Show loading indicator
            resultValue.innerHTML = '<div class="spinner"></div> Analyzing...';
            confidenceValue.textContent = '-';
            confidenceBar.style.width = '0%';
            resultDiv.style.display = 'block';
            
            try {
                // Start both the API request and a timer
                const [response] = await Promise.all([
                    fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ comment }),
                    }),
                    new Promise(resolve => setTimeout(resolve, 500)) // Minimum 500ms delay
                ]);
                
                const data = await response.json();
                resultValue.textContent = data.label;
                
                // Display confidence if available
                if (data.confidence !== undefined && data.confidence !== 'N/A') {
                    confidenceValue.textContent = `${data.confidence}%`;
                    confidenceBar.style.width = `${data.confidence}%`;
                } else if (data.confidence === 'N/A') {
                    confidenceValue.textContent = 'Not available';
                    confidenceBar.style.width = '0%';
                } else {
                    confidenceValue.textContent = '-';
                    confidenceBar.style.width = '0%';
                }
                
                // Add to prediction log
                addToPredictionLog(comment, data.label, data.confidence);
                
            } catch (error) {
                resultValue.textContent = `Error: ${error.message}`;
                confidenceValue.textContent = '-';
                confidenceBar.style.width = '0%';
            }
        });
        
        function addToPredictionLog(comment, label, confidence) {
            // Store in memory
            const timestamp = new Date();
            predictionLog.unshift({ comment, label, confidence, timestamp });
            
            // Limit log size
            if (predictionLog.length > 50) {
                predictionLog.pop();
            }
            
            // Update UI
            const logEntries = document.getElementById('log-entries');
            const emptyLog = logEntries.querySelector('.empty-log');
            if (emptyLog) {
                emptyLog.remove();
            }
            
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            
            // Format timestamp
            const timeString = timestamp.toLocaleTimeString();
            
            // Include confidence if available
            const confidenceDisplay = confidence !== undefined && confidence !== 'N/A' 
                ? ` (${confidence}%)` 
                : '';
            
            entry.innerHTML = `
                <div class="log-comment">${comment}</div>
                <div class="log-result">
                    <span class="log-timestamp">${timeString}</span>
                    <span class="log-label ${label}">${label}${confidenceDisplay}</span>
                </div>
            `;
            
            // Add to the top of the log
            logEntries.insertBefore(entry, logEntries.firstChild);
        }
        
        // Batch processing functionality
        document.getElementById('batch-submit-btn').addEventListener('click', async () => {
            const batchText = document.getElementById('batch-comments').value.trim();
            const loadingElement = document.getElementById('batch-loading');
            const vizContainer = document.getElementById('viz-container');
            
            if (!batchText) {
                alert('Please enter comments for batch processing');
                return;
            }
            
            // Show loading indicator
            loadingElement.style.display = 'block';
            
            try {
                const response = await fetch('/batch_predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ batch_text: batchText }),
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    loadingElement.style.display = 'none';
                    return;
                }
                
                // Update UI with results
                document.getElementById('positive-count').textContent = data.positive_count;
                document.getElementById('negative-count').textContent = data.negative_count;
                
                // Update topics lists
                updateTopicsList('positive-topics', data.positive_topics);
                updateTopicsList('negative-topics', data.negative_topics);
                
                // Update chart images
                if (data.positive_barchart) {
                    document.getElementById('positive-barchart').src = `data:image/png;base64,${data.positive_barchart}`;
                }
                if (data.negative_barchart) {
                    document.getElementById('negative-barchart').src = `data:image/png;base64,${data.negative_barchart}`;
                }
                
                // Show visualization container
                vizContainer.style.display = 'block';
                
                // Hide loading indicator
                loadingElement.style.display = 'none';
                
                // Scroll to visualization results
                vizContainer.scrollIntoView({ behavior: 'smooth' });
                
            } catch (error) {
                console.error('Error processing batch:', error);
                alert('Error processing batch: ' + error.message);
                loadingElement.style.display = 'none';
            }
        });
        
        function updateTopicsList(elementId, topics) {
            const topicsList = document.getElementById(elementId);
            
            // Clear existing content
            topicsList.innerHTML = '';
            
            // If no topics, show empty message
            if (!topics || topics.length === 0) {
                topicsList.innerHTML = '<li class="empty-list">No data available</li>';
                return;
            }
            
            // Add each topic to the list
            topics.forEach(topic => {
                const topicItem = document.createElement('li');
                topicItem.className = 'topic-item';
                
                // Display full term without truncation
                topicItem.innerHTML = `
                    <div class="topic-content">
                        <span class="topic-term">${topic.term}</span>
                        <span class="topic-score">${topic.score.toFixed(4)}</span>
                    </div>
                `;
                topicsList.appendChild(topicItem);
            });
        }
    </script>
</body>
</html>

        ''')

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
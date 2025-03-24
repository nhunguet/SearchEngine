from flask import Flask, render_template, request, jsonify
import pickle
import re
from underthesea import word_tokenize

app = Flask(__name__)

# Hàm làm sạch văn bản - giữ nguyên từ code gốc
def clean_text(text):
    # 1. Chuyển chữ thường
    text = text.lower()
    
    # 2. Loại bỏ số
    text = re.sub(r'\d+', '', text)
    
    # 3. Loại bỏ ký tự đặc biệt
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Loại bỏ khoảng trắng dư thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Danh sách stopwords
    stopwords = set(["và", "là", "của", "trong", "khi", "một", "những", "được",
                     "có", "cho", "để", "này", "với", "cũng", "như", "rằng",
                     "vì", "ở", "thì", "lại", "sẽ", "đã", "nên", "hoặc", "hay",
                     "thế", "nào", "gì", "này", "ấy", "đó", "vậy", "hơn", "lên", "xuống"])
    
    # 6. Tách từ và loại bỏ stopwords
    words = [word for word in word_tokenize(text) if word not in stopwords]
    
    # Trả về văn bản sau khi loại bỏ stopwords
    return " ".join(words)

def load_model():
    try:
        # Tải mô hình XGBoost từ file pickle
        model = pickle.load(open("xgboost_text_classifier.pkl", "rb"))
        # Tải vectorizer từ file pickle
        vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
        
        # Ánh xạ cố định từ ID lớp đến tên danh mục
        id_to_category = {
            0: 'công nghệ', 
            1: 'giáo dục', 
            2: 'pháp luật', 
            3: 'sức khỏe', 
            4: 'thời sự', 
            5: 'kinh tế'
        }
        
        print("Mô hình XGBoost đã được tải thành công!")
        return model, vectorizer, id_to_category
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        raise

# Tải mô hình khi khởi động
model, vectorizer, id_to_category = load_model()
print("Model đã được tải!")

@app.route('/')
def home():
    return render_template('index.html')    

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        if not text:
            return jsonify({'error': 'Không có văn bản nào được cung cấp'})

        # Làm sạch văn bản (sử dụng hàm từ code gốc)
        cleaned_text = clean_text(text)
        
        # Chuyển đổi văn bản thành vector đặc trưng sử dụng TF-IDF
        features = vectorizer.transform([cleaned_text])
        
        # Dự đoán sử dụng mô hình XGBoost
        prediction = model.predict(features)[0]
        
        # Nếu mô hình hỗ trợ dự đoán xác suất, lấy xác suất của các lớp
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[0]
            probabilities = {id_to_category[i]: float(prob) for i, prob in enumerate(probs)}
        
        # Lấy tên lớp dự đoán
        predicted_category = id_to_category[prediction]

        # Tạo kết quả trả về
        result = {
            'category': predicted_category,
            'text': text[:100] + '...' if len(text) > 100 else text
        }
        
        # Thêm xác suất nếu có
        if probabilities:
            result['probabilities'] = probabilities
            
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)  # Bật debug mode để dễ dàng theo dõi lỗi trong quá trình phát triển
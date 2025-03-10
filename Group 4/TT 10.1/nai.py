from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dữ liệu mẫu: Văn bản và nhãn phân loại
texts = [
    "I love this movie, it's fantastic!",
    "This film was terrible, I hated it.",
    "An amazing experience, I really enjoyed it.",
    "Worst movie ever, do not watch!",
    "Absolutely loved it, great acting!",
    "Horrible script, waste of time.",
]
labels = [1, 0, 1, 0, 1, 0]  # 1: Tích cực, 0: Tiêu cực

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Xây dựng mô hình Naïve Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Thử nghiệm với một câu mới
new_text = ["I think the movie was fantastic and very enjoyable."]
prediction = model.predict(new_text)
print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Dữ liệu mẫu: Nhãn thực tế (y_true) và Nhãn dự đoán (y_pred)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # Giá trị thực tế
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])  # Dự đoán của mô hình

# Tính Accuracy và F1-score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Microaveraging trong đánh giá phân loại
y_true_multi = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
y_pred_multi = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

micro_precision = precision_score(y_true_multi.argmax(axis=1), y_pred_multi.argmax(axis=1), average='micro')
micro_recall = recall_score(y_true_multi.argmax(axis=1), y_pred_multi.argmax(axis=1), average='micro')
micro_f1 = f1_score(y_true_multi.argmax(axis=1), y_pred_multi.argmax(axis=1), average='micro')

# Hiển thị kết quả
print(f"Accuracy: {accuracy:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"Micro Precision: {micro_precision:.2f}")
print(f"Micro Recall: {micro_recall:.2f}")
print(f"Micro F1-score: {micro_f1:.2f}")

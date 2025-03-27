from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import librosa
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
app = Flask(__name__)

# Tải các mô hình CNN2D đã huấn luyện
model_emotion = tf.keras.models.load_model(fr'best_emotion_recognition_model.keras')
model_region = tf.keras.models.load_model(fr'best_region_recognition_model.keras')  # Mô hình SRR
model_gender = tf.keras.models.load_model(fr'best_gender_recognition_model.keras')  # Mô hình SGR

# Khai báo biến toàn cục
audio_window = []

# Hàm chuẩn hóa độ lớn âm thanh
def match_loudness(audio, target_rms=0.015823712572455406):
    audio = np.array(audio, dtype=np.float32)  # Chuyển list thành mảng NumPy
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms == 0:
        return audio
    scalar = target_rms / current_rms
    return audio * scalar

# Hàm chuẩn hóa âm thanh
def normalize_audio(data):
    return librosa.util.normalize(data)

# Hàm trích xuất đặc trưng
def extract_features(data, sampling_rate, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=data, sr=sampling_rate).mean(axis=1)
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sampling_rate).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sampling_rate).mean()
    return np.hstack((mfccs, chroma, [spectral_centroid], [spectral_bandwidth]))

# Hàm tạo dữ liệu 1D
def make1Ddata(sound, sr):
    data_temp = np.array(sound, dtype=np.float32)  # Đảm bảo dữ liệu là NumPy array
    data = match_loudness(data_temp)
    data = normalize_audio(data)
    features = extract_features(data, sr)
    return features

# Hàm chuyển sang 2D
def convert_1dto2d_reality(input_1d):
    input_1d = input_1d.flatten()
    mfcc = input_1d[:40].reshape(4, 10)
    chroma = input_1d[40:52].reshape(2, 6)
    spectral = input_1d[52:54].reshape(1, 2)

    chroma_padded = np.pad(chroma, ((0, 0), (0, 4)), mode='constant')
    spectral_padded = np.pad(spectral, ((0, 0), (0, 8)), mode='constant')

    matrix_2d = np.vstack((mfcc, chroma_padded, spectral_padded))
    return np.expand_dims(matrix_2d, axis=(0, -1))

# Hàm chuẩn hóa dữ liệu 2D cho SER (Emotion)
def make2dData_reality_emotion(X_test):
    scaler_mfcc = joblib.load(fr"scaler_mfcc_emo.pkl")
    scaler_chroma = joblib.load(fr"scaler_chroma_emo.pkl")
    scaler_spectral = joblib.load(fr"scaler_spectral_emo.pkl")

    X_test_mfcc = scaler_mfcc.transform(X_test[:40].reshape(1, -1))
    X_test_chroma = scaler_chroma.transform(X_test[40:52].reshape(1, -1))
    X_test_spectral = scaler_spectral.transform(X_test[52:54].reshape(1, -1))

    X_test_scaled = np.hstack((X_test_mfcc, X_test_chroma, X_test_spectral))
    X_test_2D = convert_1dto2d_reality(X_test_scaled)
    return X_test_2D

# Hàm chuẩn hóa dữ liệu 2D cho SRR (Region)
def make2dData_reality_region(X_test):
    scaler_mfcc = joblib.load(fr"scaler_mfcc_reg.pkl")
    scaler_chroma = joblib.load(fr"scaler_chroma_reg.pkl")
    scaler_spectral = joblib.load(fr"scaler_spectral_reg.pkl")

    X_test_mfcc = scaler_mfcc.transform(X_test[:40].reshape(1, -1))
    X_test_chroma = scaler_chroma.transform(X_test[40:52].reshape(1, -1))
    X_test_spectral = scaler_spectral.transform(X_test[52:54].reshape(1, -1))

    X_test_scaled = np.hstack((X_test_mfcc, X_test_chroma, X_test_spectral))
    X_test_2D = convert_1dto2d_reality(X_test_scaled)
    return X_test_2D

# Hàm chuẩn hóa dữ liệu 2D cho SGR (Gender)
def make2dData_reality_gender(X_test):
    scaler_mfcc = joblib.load(fr"scaler_mfcc_gen.pkl")
    scaler_chroma = joblib.load(fr"scaler_chroma_gen.pkl")
    scaler_spectral = joblib.load(fr"scaler_spectral_gen.pkl")

    X_test_mfcc = scaler_mfcc.transform(X_test[:40].reshape(1, -1))
    X_test_chroma = scaler_chroma.transform(X_test[40:52].reshape(1, -1))
    X_test_spectral = scaler_spectral.transform(X_test[52:54].reshape(1, -1))

    X_test_scaled = np.hstack((X_test_mfcc, X_test_chroma, X_test_spectral))
    X_test_2D = convert_1dto2d_reality(X_test_scaled)
    return X_test_2D

# Hàm đầu vào cho CNN2D
def input_of_CNN2D(sound, sr, model_type):
    data1D = make1Ddata(sound, sr)
    if model_type == 'emotion':
        data2D = make2dData_reality_emotion(data1D)
    elif model_type == 'region':
        data2D = make2dData_reality_region(data1D)
    elif model_type == 'gender':
        data2D = make2dData_reality_gender(data1D)
    return data2D

# Route cho trang thành viên
@app.route('/members')
def members():
    return render_template('members.html')

# Route cho trang chính (hệ thống)
@app.route('/')
def index():
    return render_template('index.html')

# Route cho trang feedback
@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

result_past = {
        'emotion': 'Đang chờ dữ liệu...',
        'emotion_accuracy': None,
        'region': 'Đang chờ dữ liệu...',
        'region_accuracy': None,
        'gender': 'Đang chờ dữ liệu...',
        'gender_accuracy': None
    }
count = 0
@app.route('/predict', methods=['POST'])
def predict():

    global audio_window, result_past, count

    # Nhận dữ liệu âm thanh 1 giây từ frontend
    new_audio = np.array(request.json['audio'], dtype=np.float32)
    sample_rate = 48000
    samples_per_second = sample_rate  # 22050 mẫu mỗi giây

    # Thêm dữ liệu mới vào cửa sổ
    audio_window.extend(new_audio)

    # Giới hạn cửa sổ ở 3 giây (66150 mẫu = 22050 * 3)
    if len(audio_window) > samples_per_second * 3:
        audio_window = audio_window[-samples_per_second * 3:]

    # Dự đoán khi có dữ liệu (ít nhất 1 giây, tối đa 3 giây)
    if len(audio_window) >= samples_per_second:  # Bắt đầu dự đoán từ giây đầu tiên
        # Nếu chưa đủ 3 giây, pad dữ liệu bằng 0
        if len(audio_window) < samples_per_second * 3:
            padded_audio = np.pad(audio_window, (0, samples_per_second * 3 - len(audio_window)), mode='constant')
        else:
            padded_audio = np.array(audio_window, dtype=np.float32)

        # Dự đoán cảm xúc (SER)
        print(np.sum(np.abs(padded_audio)))
    if np.sum(np.abs(padded_audio)) > 500:
        data_2d_emotion = input_of_CNN2D(padded_audio, sample_rate, model_type='emotion')
        prediction_emotion = model_emotion.predict(data_2d_emotion)
        emotion_index = np.argmax(prediction_emotion)
        emotion_accuracy = prediction_emotion[0][emotion_index]
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

        # Dự đoán vùng miền (SRR)
        data_2d_region = input_of_CNN2D(padded_audio, sample_rate, model_type='region')
        prediction_region = model_region.predict(data_2d_region)
        region_index = np.argmax(prediction_region)
        region_accuracy = prediction_region[0][region_index]
        regions = ['Northern', 'Central', 'Southern' ]  # Điều chỉnh theo nhãn của bạn

        # Dự đoán giới tính (SGR)
        data_2d_gender = input_of_CNN2D(padded_audio, sample_rate, model_type='gender')
        prediction_gender = model_gender.predict(data_2d_gender)
        gender_index = np.argmax(prediction_gender)
        gender_accuracy = prediction_gender[0][gender_index]
        genders = ['Male', 'Female']  # Điều chỉnh theo nhãn của bạn

        result = {
            'emotion': emotions[emotion_index],
            'emotion_accuracy': float(emotion_accuracy),
            'region': regions[region_index],
            'region_accuracy': float(region_accuracy),
            'gender': genders[gender_index],
            'gender_accuracy': float(gender_accuracy)
        }
        result_past = result
    else:
        result = result_past
        count +=1
        if count == 3:
            count = 0
            result = result_past = {
                                    'emotion': 'Đang chờ dữ liệu...',
                                    'emotion_accuracy': None,
                                    'region': 'Đang chờ dữ liệu...',
                                    'region_accuracy': None,
                                    'gender': 'Đang chờ dữ liệu...',
                                    'gender_accuracy': None
                                    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)
import librosa
import numpy as np
import tensorflow as tf
import joblib
import logging
import os

# Thiết lập logging
logger = logging.getLogger(__name__)

def match_loudness(audio, target_rms=0.015823712572455406):
    """
    Điều chỉnh âm lượng của tín hiệu âm thanh để phù hợp với mức RMS mục tiêu.
    """
    try:
        current_rms = np.sqrt(np.mean(audio**2))
        logger.info(f"📊 RMS hiện tại của âm thanh: {current_rms}")
        
        if current_rms == 0:
            logger.warning("⚠️ Âm thanh im lặng (RMS = 0), không thể điều chỉnh âm lượng")
            return audio
            
        scalar = target_rms / current_rms
        logger.info(f"📊 Hệ số điều chỉnh âm lượng: {scalar}")
        
        return audio * scalar
    except Exception as e:
        logger.error(f"❌ Lỗi khi điều chỉnh âm lượng: {str(e)}")
        return audio  # Trả về âm thanh gốc nếu có lỗi

def load_model(model_path):
    """Load the Keras model from the given path."""
    try:
        logger.info(f"Đang tải mô hình từ: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Đã tải mô hình thành công")
        return model
    except Exception as e:
        logger.error(f"❌ Lỗi khi load model: {e}")
        raise

def normalize_audio(data):
    """Chuẩn hóa âm thanh."""
    try:
        return librosa.util.normalize(data)
    except Exception as e:
        logger.error(f"❌ Lỗi khi chuẩn hóa âm thanh: {str(e)}")
        return data

def extract_features(y, sr):
    """
    Trích xuất đặc trưng MFCC, chroma, spectral từ âm thanh.
    """
    try:
        logger.info(f"Đang trích xuất đặc trưng, độ dài âm thanh: {len(y)}, sr: {sr}")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).mean(axis=1)  # 40 giá trị
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)[:12]  # 12 giá trị
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()  # 1 giá trị
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()  # 1 giá trị

        # Gộp lại thành 54 giá trị
        features = np.hstack((mfccs, chroma, [spectral_centroid], [spectral_bandwidth]))

        # Kiểm tra đầu ra có đúng 54 giá trị không
        if len(features) != 54:
            logger.error(f"❌ extract_features() lỗi: Đầu ra có {len(features)} giá trị, cần đúng 54.")
            raise ValueError(f"❌ extract_features() lỗi: Đầu ra có {len(features)} giá trị, cần đúng 54.")

        logger.info(f"✅ Đã trích xuất đặc trưng thành công, shape: {features.shape}")
        return features
    except Exception as e:
        logger.error(f"❌ Lỗi khi trích xuất đặc trưng: {str(e)}")
        # Tạo đặc trưng giả với đúng kích thước
        dummy_features = np.zeros(54)
        logger.warning("⚠️ Trả về đặc trưng giả")
        return dummy_features

def make1Ddata(sound, sr):
    """
    Xử lý âm thanh và trích xuất đặc trưng 1D.
    """
    try:
        data = match_loudness(sound)
        data = normalize_audio(data)
        features = extract_features(data, sr)
        return features
    except Exception as e:
        logger.error(f"❌ Lỗi khi tạo dữ liệu 1D: {str(e)}")
        return np.zeros(54)

def convert_1dto2d_reality(input_1d):
  """
  Chuyển đổi đặc trưng 1D thành 2D (7, 10, 1).
  """
  try:
      input_1d = input_1d.flatten()
      mfcc = input_1d[:40].reshape(4, 10)
      chroma = input_1d[40:52].reshape(2, 6)
      spectral = input_1d[52:54].reshape(1, 2)

      chroma_padded = np.pad(chroma, ((0, 0), (0, 4)), mode='constant')
      spectral_padded = np.pad(spectral, ((0, 0), (0, 8)), mode='constant')

      matrix_2d = np.vstack((mfcc, chroma_padded, spectral_padded))

      return np.expand_dims(matrix_2d, axis=(0, -1))
  except Exception as e:
      logger.error(f"❌ Lỗi khi chuyển đổi 1D sang 2D: {str(e)}")
      raise

def make2dData_reality(X_test):
  """
  Chuẩn hóa dữ liệu sử dụng scaler đã lưu.
  Chuyển đổi thành dạng 2D (7, 10, 1) phù hợp với model.
  """
  try:
      # Đảm bảo X_test là mảng 2D với shape (n_samples, n_features)
      if X_test.ndim == 1:
          X_test = np.expand_dims(X_test, axis=0)
          
      # Load scaler đã train
      scaler_paths = {
          "mfcc": "C:\\Users\\MSI\\TT\\thy\\w8_system\\scaler_mfcc_emo.pkl",
          "chroma": "C:\\Users\\MSI\\TT\\thy\\w8_system\\scaler_chroma_emo.pkl",
          "spectral": "C:\\Users\\MSI\\TT\\thy\\w8_system\\scaler_spectral_emo.pkl"
      }
      
      # Kiểm tra các file scaler tồn tại
      for name, path in scaler_paths.items():
          if not os.path.exists(path):
              logger.error(f"❌ File scaler {name} không tồn tại: {path}")
              raise FileNotFoundError(f"File scaler {name} không tồn tại: {path}")
      
      logger.info("Đang tải các scaler...")
      scaler_mfcc = joblib.load(scaler_paths["mfcc"])
      scaler_chroma = joblib.load(scaler_paths["chroma"])
      scaler_spectral = joblib.load(scaler_paths["spectral"])
      logger.info("✅ Đã tải các scaler thành công")

      # Scale dữ liệu
      X_test_mfcc = scaler_mfcc.transform(X_test[:, :40])
      X_test_chroma = scaler_chroma.transform(X_test[:, 40:52])
      X_test_spectral = scaler_spectral.transform(X_test[:, 52:54])

      # Gộp lại sau khi scale
      X_test_scaled = np.hstack((X_test_mfcc, X_test_chroma, X_test_spectral))
      logger.info(f"✅ Đã chuẩn hóa dữ liệu, shape: {X_test_scaled.shape}")

      # Chuyển đổi thành ma trận 2D (7, 10, 1)
      X_test_2D = convert_1dto2d_reality(X_test_scaled)

      return X_test_2D  # Trả về dữ liệu đã chuẩn hóa & định dạng đúng
  except Exception as e:
      logger.error(f"❌ Lỗi khi chuẩn hóa dữ liệu: {str(e)}")
      raise

def input_of_CNN2D(sound, sr):
  """
  Xử lý âm thanh và chuyển đổi thành định dạng đầu vào cho mô hình CNN 2D.
  """
  try:
      data1D = make1Ddata(sound, sr)
      data2D = make2dData_reality(data1D)
      return data2D
  except Exception as e:
      logger.error(f"❌ Lỗi khi tạo đầu vào cho CNN2D: {str(e)}")
      raise

def preprocess_audio(file_path, sr=22050):
  """
  Tiền xử lý file âm thanh:
  - Tải file .wav
  - Xử lý và chuyển đổi thành định dạng đầu vào cho mô hình
  """
  try:
      logger.info(f"Đang xử lý file âm thanh: {file_path}")
      
      # Kiểm tra file tồn tại
      if not os.path.exists(file_path):
          logger.error(f"❌ File không tồn tại: {file_path}")
          raise FileNotFoundError(f"File không tồn tại: {file_path}")
          
      # Kiểm tra kích thước file
      file_size = os.path.getsize(file_path)
      if file_size == 0:
          logger.error(f"❌ File rỗng: {file_path}")
          raise ValueError(f"File rỗng: {file_path}")
          
      logger.info(f"📊 Kích thước file: {file_size} bytes")
      
      # Tải file âm thanh
      y, sr = librosa.load(file_path, sr=sr)
      logger.info(f"✅ Đã tải file âm thanh, độ dài: {len(y)}")
      
      # Xử lý âm thanh và chuyển đổi thành định dạng đầu vào cho mô hình
      features_2d = input_of_CNN2D(y, sr)
      logger.info(f"✅ Đã xử lý xong, kết quả shape: {features_2d.shape}")

      return features_2d
  except Exception as e:
      logger.error(f"❌ Lỗi khi xử lý file âm thanh: {str(e)}")
      import traceback
      logger.error(traceback.format_exc())
      raise

def create_dummy_features():
    """Tạo đặc trưng giả để test"""
    # Tạo một mảng numpy với kích thước phù hợp với mô hình (1, 7, 10, 1)
    dummy_features = np.zeros((1, 7, 10, 1))
    logger.info("⚠️ Đã tạo đặc trưng giả với shape: (1, 7, 10, 1)")
    return dummy_features


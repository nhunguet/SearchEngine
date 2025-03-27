import os
from pydub import AudioSegment
import librosa
import numpy as np

# def classify_audio_type(audio_path):
#     y, sr = librosa.load(audio_path, sr=None)
#     frame_length = 2048
#     hop_length = 512
#     energy = np.array([sum(abs(y[i:i + frame_length] ** 2)) for i in range(0, len(y), hop_length)])
#     noise_energy = np.percentile(energy, 10)
#     signal_energy = np.mean(energy[energy > noise_energy])
#     snr = 10 * np.log10(signal_energy / (noise_energy + 1e-10))
#
#     pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
#     pitch_variability = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
#
#     if snr > 20:
#         return "1"
#     elif 5 <= snr <= 20:
#         return "2"
#     else:
#         if pitch_variability > 50:
#             return "3"
#         return "2"

def classify_audio_type(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        raise ValueError(f"Không thể tải file âm thanh {audio_path}: {e}")

    frame_length = 2048
    hop_length = 512
    energy = np.array([sum(abs(y[i:i + frame_length] ** 2)) for i in range(0, len(y), hop_length)])
    noise_energy = np.percentile(energy, 10)
    signal_energy = np.mean(energy[energy > noise_energy]) if np.any(energy > noise_energy) else noise_energy
    snr = 10 * np.log10(signal_energy / (noise_energy + 1e-10))

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch_variability = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    if snr > 20:
        return "1"
    elif 5 <= snr <= 20:
        return "2"
    else:
        if pitch_variability > 50:
            return "3"
        return "2"


def cut_segments(input_dir, output_dir, segment_length=3, combine_files=False):
    segment_length_ms = segment_length * 1000
    os.makedirs(output_dir, exist_ok=True)
    folder_name = os.path.basename(input_dir).lower()

    gender = "1" if "female" in folder_name else "0" if "male" in folder_name else "unknown"
    if combine_files:
        # Nối tất cả file WAV thành một file duy nhất
        combined_audio = AudioSegment.empty()
        for file_name in sorted(os.listdir(input_dir)):  # Sắp xếp để giữ thứ tự
            if file_name.endswith(".wav"):
                file_path = os.path.join(input_dir, file_name)
                audio = AudioSegment.from_file(file_path, format="wav")
                combined_audio += audio

        # Lưu tạm file nối để phân loại
        temp_path = os.path.join(output_dir, "combined_temp.wav")
        combined_audio.export(temp_path, format="wav")
        audio_type = classify_audio_type(temp_path)


        # Cắt file nối thành đoạn 3 giây
        total_length_ms = len(combined_audio)
        num_segments = total_length_ms // segment_length_ms
        segment_count = 1
        for i in range(int(num_segments)):
            start_ms = i * segment_length_ms
            end_ms = (i + 1) * segment_length_ms
            segment = combined_audio[start_ms:end_ms]
            segment_name = f"{audio_type}-{gender}-{segment_count}.wav"
            segment_path = os.path.join(output_dir, segment_name)
            segment.export(segment_path, format="wav")
            print(f"Exported segment: {segment_path}")
            segment_count += 1

        os.remove(temp_path)
    else:
        # Xử lý từng file riêng lẻ (phiên bản hiện tại)
        segment_count = 1
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(input_dir, file_name)
                audio = AudioSegment.from_file(file_path, format="wav")

                audio_type = classify_audio_type(file_path)


                total_length_ms = len(audio)
                num_segments = total_length_ms // segment_length_ms
                for i in range(int(num_segments)):
                    start_ms = i * segment_length_ms
                    end_ms = (i + 1) * segment_length_ms
                    segment = audio[start_ms:end_ms]
                    segment_name = f"{audio_type}-{gender}-{segment_count}.wav"
                    segment_path = os.path.join(output_dir, segment_name)
                    segment.export(segment_path, format="wav")
                    print(f"Exported segment: {segment_path}")
                    segment_count += 1

if __name__ == "__main__":
    base_dir = r"/Vietnamese_accent_3_regions/Central/Unknown"
    output_dir = base_dir
    # Không nối (phiên bản hiện tại)
    # cut_segments(os.path.join(base_dir, "female"), output_dir, combine_files=True)
    #cut_segments(os.path.join(base_dir, "female"), output_dir, combine_files=False)
    # Có nối (tùy chọn)
    cut_segments(os.path.join(base_dir, "male"), output_dir, combine_files=True)
    #cut_segments(os.path.join(base_dir, "male"), output_dir, combine_files=False)
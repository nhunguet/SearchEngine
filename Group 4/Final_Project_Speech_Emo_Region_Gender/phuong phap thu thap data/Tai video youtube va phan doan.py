import os
import yt_dlp
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torchaudio
import re

def read_links(file_path):
    with open(file_path, 'r') as file:
        links = [line.strip() for line in file if line.strip()]
    return links

def download_and_extract_voice(link, output_base_dir, voice_output_dir="female"):
    # Tải âm thanh từ YouTube
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': '%(title)s.%(ext)s',
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        title = re.sub(r'[_!@#$%^&*()+=\{\}\[\]|\\:;"\'<>,?/]', '_', info.get('title', 'unknown'))
        output_dir = os.path.join(output_base_dir, title)
        os.makedirs(output_dir, exist_ok=True)


        final_output_path = os.path.join(output_dir, f"{title}.mp3")
        ydl_opts['outtmpl'] = os.path.join(output_dir, f"{title}.%(ext)s")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        print(f"Downloaded audio: {final_output_path}")

    # Trích xuất giọng người
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                        use_auth_token="Your Token")
    waveform, sample_rate = torchaudio.load(final_output_path)
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    audio = AudioSegment.from_file(final_output_path, format="mp3")
    voice_output_dir = os.path.join(output_dir, voice_output_dir)
    os.makedirs(voice_output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(final_output_path))[0]
    segment_count = 1
    for segment in output.get_timeline().support():
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        voice_segment = audio[start_ms:end_ms]

        segment_name = f"segment_{segment_count}.wav"
        segment_path = os.path.join(voice_output_dir, segment_name)
        voice_segment.export(segment_path, format="wav")
        print(f"Exported voice segment: {segment_path}")
        segment_count += 1

    print(f"Extracted and saved {segment_count - 1} voice segments to {voice_output_dir}")
    # Xóa file MP3 gốc nếu không cần
    os.remove(final_output_path)

if __name__ == "__main__":
    links_file = "Links.txt"
    links = read_links(links_file)
    for link in links:
        download_and_extract_voice(link, "youtube_audio")
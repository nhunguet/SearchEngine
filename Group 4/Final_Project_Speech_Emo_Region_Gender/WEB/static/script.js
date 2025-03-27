let audioContext, analyser, microphone, dataArray, recording = false;
const canvas = document.getElementById('waveform');
const canvasCtx = canvas.getContext('2d');
const emotionDiv = document.getElementById('emotionResult');
const emotionAccuracyDiv = document.getElementById('emotionAccuracy');
const regionDiv = document.getElementById('regionResult');
const regionAccuracyDiv = document.getElementById('regionAccuracy');
const genderDiv = document.getElementById('genderResult');
const genderAccuracyDiv = document.getElementById('genderAccuracy');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
let processor;

// Bắt đầu ghi âm
startBtn.onclick = async () => {
    if (!recording) {
        recording = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(analyser);

        // Tạo ScriptProcessor để xử lý âm thanh thô
        processor = audioContext.createScriptProcessor(2048, 1, 1);
        microphone.connect(processor);
        processor.connect(audioContext.destination);

        // Gửi dữ liệu mỗi 3 giây
        let audioBuffer = [];
        processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            audioBuffer.push(...inputData);

            // Mỗi 3 giây, gửi dữ liệu lên server
            if (audioBuffer.length >= audioContext.sampleRate * 3) {
                sendAudioToServer([...audioBuffer]);
                audioBuffer = []; // Reset buffer
            }
        };

        drawWaveform();
    }
};

// Dừng ghi âm
stopBtn.onclick = () => {
    if (recording) {
        recording = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        microphone.disconnect();
        processor.disconnect();
        audioContext.close();
        emotionDiv.innerHTML = 'Stop recording...';
        emotionAccuracyDiv.innerHTML = 'accuracy: 0%';
        regionDiv.innerHTML = '...';
        regionAccuracyDiv.innerHTML = 'accuracy: 0%';
        genderDiv.innerHTML = '...';
        genderAccuracyDiv.innerHTML = 'accuracy: 0%';
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    }
};

// Gửi dữ liệu âm thanh lên server
async function sendAudioToServer(audioData) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio: audioData })
    });
    const result = await response.json();
    emotionDiv.innerHTML = result.emotion;
    emotionAccuracyDiv.innerHTML = `accuracy: ${(result.emotion_accuracy * 100).toFixed(2)}%`;
    regionDiv.innerHTML = result.region;
    regionAccuracyDiv.innerHTML = `accuracy: ${(result.region_accuracy * 100).toFixed(2)}%`;
    genderDiv.innerHTML = result.gender;
    genderAccuracyDiv.innerHTML = `accuracy: ${(result.gender_accuracy * 100).toFixed(2)}%`;
}

// Vẽ biểu đồ sóng âm thanh
function drawWaveform() {
    if (!recording) return;

    requestAnimationFrame(drawWaveform);
    analyser.getByteTimeDomainData(dataArray);

    // Xóa canvas nhưng không vẽ nền trắng (để trong suốt)
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    // Tăng độ dày của đường sóng
    canvasCtx.lineWidth = 6; // Đường sóng to hơn (tăng từ 4 lên 6)

    // Đặt màu cho đường sóng giống phông chữ
    canvasCtx.strokeStyle = '#FFFFFF'; // Màu trắng
    canvasCtx.shadowColor = '#00FFFF'; // Hiệu ứng phát sáng màu cyan
    canvasCtx.shadowBlur = 10; // Độ mờ của hiệu ứng phát sáng

    canvasCtx.beginPath();

    const sliceWidth = canvas.width / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2; // Giữ nguyên biên độ sóng

        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);

        x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}
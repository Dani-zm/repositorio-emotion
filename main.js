/**
 * main.js - Emotion AI Web Integration
 * 
 * Archivo principal de JavaScript para la interfaz de usuario.
 * Controla el acceso a la cámara web, captura los fotogramas del video para 
 * enviarlos al servidor (Flask) y obtener la predicción de la emoción. 
 * También maneja la grabación de sesiones y la conexión con el endpoint de Gemini.
 */

document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const emotionResult = document.getElementById('emotion-result');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const webcamContainer = document.getElementById('webcam-container');
    
    // Gemini Integration Elements
    const recordBtn = document.getElementById('record-btn');
    const sessionContextInput = document.getElementById('session-context');
    const analysisResultsContainer = document.querySelector('.gemini-analysis-results');
    const geminiResultText = document.getElementById('gemini-result');
    const recordingTimer = document.createElement('div');
    recordingTimer.className = 'recording-timer';
    recordingTimer.style.display = 'none';
    webcamContainer.querySelector('.emotion-overlay').appendChild(recordingTimer);
    
    let stream = null;
    let predictionInterval = null;
    
    // Session State
    let isRecordingSession = false;
    let recordedEmotions = [];
    let recordingCountdown = 30;
    let timerInterval = null;

    // Start webcam
    startBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            recordBtn.disabled = false; // Enable record button when webcam starts
            emotionResult.textContent = 'Analizando...';

            // Wait for video to be ready before starting predictions
            video.addEventListener('loadeddata', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                // Start sending frames to backend every 500ms
                predictionInterval = setInterval(sendFrameForPrediction, 1000);
            });

        } catch (error) {
            console.error('Error accessing webcam:', error);
            emotionResult.textContent = 'Error: No se pudo acceder a la cámara.';
            
            let errMsg = `No se pudo acceder a la cámara.\n\nError: ${error.name}\nMensaje: ${error.message}\n\n`;
            if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
                errMsg += "👉 POSIBLE CAUSA: Otra aplicación (Zoom, un script de Python anterior, etc.) o pestaña del navegador está usando la cámara. Ciérrala e intenta de nuevo.";
            } else if (error.name === 'NotAllowedError') {
                errMsg += "👉 POSIBLE CAUSA: Denegaste el permiso de cámara en el navegador. Por favor, permítelo en el ícono de candado de la URL.";
            } else {
                errMsg += "👉 POSIBLE CAUSA: ¿Abriste el archivo con doble clic (file://...)? Debes entrar por http://127.0.0.1:5000/ o http://localhost:5000/";
            }
            alert(errMsg);
        }
    });

    // Stop webcam
    stopBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        if (predictionInterval) {
            clearInterval(predictionInterval);
            predictionInterval = null;
        }
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        recordBtn.disabled = true; // Disable record button when webcam stops
        
        // Stop recording if active
        if (isRecordingSession) {
            stopRecording();
        }
        
        emotionResult.textContent = 'Cámara pausada.';
    });

    // Toggle fullscreen
    fullscreenBtn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            webcamContainer.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    });

    // --- Gemini Recording Session Logic ---

    recordBtn.addEventListener('click', () => {
        if (!isRecordingSession) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    function startRecording() {
        const contextText = sessionContextInput.value.trim();
        if (!contextText) {
            alert('Por favor, escribe de qué vas a hablar en el campo de contexto antes de grabar.');
            sessionContextInput.focus();
            return;
        }

        isRecordingSession = true;
        recordedEmotions = [];
        recordingCountdown = 30;
        
        recordBtn.textContent = 'Detener Grabación';
        recordBtn.classList.add('pulse-animation'); // optional styling
        sessionContextInput.disabled = true;
        
        // Setup UI
        analysisResultsContainer.style.display = 'none';
        recordingTimer.style.display = 'block';
        recordingTimer.textContent = `00:${recordingCountdown}`;
        
        timerInterval = setInterval(() => {
            recordingCountdown--;
            const formattedTime = recordingCountdown < 10 ? `0${recordingCountdown}` : recordingCountdown;
            recordingTimer.textContent = `00:${formattedTime}`;
            
            if (recordingCountdown <= 0) {
                stopRecordingAndAnalyze();
            }
        }, 1000);
    }

    function stopRecording() {
        isRecordingSession = false;
        clearInterval(timerInterval);
        
        recordBtn.textContent = 'Grabar Análisis (30s)';
        recordBtn.classList.remove('pulse-animation');
        sessionContextInput.disabled = false;
        recordingTimer.style.display = 'none';
        emotionResult.style.display = 'inline';
    }

    async function stopRecordingAndAnalyze() {
        stopRecording();
        
        const contextText = sessionContextInput.value.trim();
        if (recordedEmotions.length === 0) {
            alert('No se detectaron emociones durante la sesión. Intenta iluminar mejor tu rostro.');
            return;
        }

        // Show loading state
        analysisResultsContainer.style.display = 'block';
        geminiResultText.textContent = '⏱️ Gemini está analizando tu discurso y tus expresiones faciales. Por favor espera...';
        
        try {
            const response = await fetch('/analyze_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    context: contextText,
                    emotions: recordedEmotions
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                geminiResultText.textContent = `❌ Error: ${data.error}`;
                geminiResultText.style.color = 'var(--accent-color)';
            } else if (data.analysis) {
                geminiResultText.textContent = data.analysis;
                geminiResultText.style.color = 'var(--text-light)';
            }
        } catch (error) {
            console.error('Error in analyze_session:', error);
            geminiResultText.textContent = '❌ Ocurrió un error al contactar con la IA para revisar la sesión.';
            geminiResultText.style.color = 'var(--accent-color)';
        }
    }

    // Send frame to Flask backend
    async function sendFrameForPrediction() {
        if (!video.videoWidth) return;

        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64 jpeg
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: dataUrl })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            if (data.emotion) {
                emotionResult.textContent = `Emoción detectada: ${data.emotion}`;
                
                // Map API response to visual classes
                const emotionKey = data.emotion.toLowerCase(); // angry, happy, sad, surprised
                
                // Track emotion if recording
                if (isRecordingSession && emotionKey !== 'no face detected') {
                    // map to internal array
                    const emMap = {
                        'angry': 'Enojo',
                        'happy': 'Alegría',
                        'sad': 'Tristeza',
                        'surprised': 'Sorpresa'
                    };
                    recordedEmotions.push(emMap[emotionKey] || data.emotion);
                }

                // Update UI based on emotion (Optional polish)
                const overlay = document.querySelector('.emotion-overlay');
                overlay.className = 'emotion-overlay'; // Reset
                overlay.classList.add(`emotion-${data.emotion.toLowerCase()}`);
            }

        } catch (error) {
            console.error('Prediction error:', error);
            // Don't show error to user constantly to avoid flicker if just one frame fails,
            // but log to console.
        }
    }
});

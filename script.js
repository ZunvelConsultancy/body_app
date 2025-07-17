import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingElement = document.getElementById("loading");

let poseLandmarker;
let drawingUtils;

function onResize() {
    const videoAspectRatio = video.videoWidth / video.videoHeight;
    const windowAspectRatio = window.innerWidth / window.innerHeight;
    let newWidth, newHeight;

    if (windowAspectRatio > videoAspectRatio) {
        newWidth = window.innerWidth;
        newHeight = window.innerWidth / videoAspectRatio;
    } else {
        newWidth = window.innerHeight * videoAspectRatio;
        newHeight = window.innerHeight;
    }

    canvasElement.style.width = newWidth + 'px';
    canvasElement.style.height = newHeight + 'px';
    canvasElement.style.left = (window.innerWidth - newWidth) / 2 + 'px';
    canvasElement.style.top = (window.innerHeight - newHeight) / 2 + 'px';
}

async function setup() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });
    drawingUtils = new DrawingUtils(canvasCtx);
    console.log("PoseLandmarker cargado.");
    enableCam();
}

function enableCam() {
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: { exact: "environment" }, // Usar la cámara trasera
            width: { ideal: 1280 }, 
            height: { ideal: 720 } 
        } 
    })
        .then(stream => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                loadingElement.style.display = 'none';
                canvasElement.width = video.videoWidth;
                canvasElement.height = video.videoHeight;
                window.addEventListener('resize', onResize);
                onResize(); // Llamar una vez para el ajuste inicial
                predictWebcam();
            });
        })
        .catch(err => {
            console.error("Error al acceder a la cámara: ", err);
            loadingElement.innerText = "Error al acceder a la cámara.";
        });
}

async function predictWebcam() {
    const startTimeMs = performance.now();
    const results = poseLandmarker.detectForVideo(video, startTimeMs);

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Dibujar el fotograma del video en el canvas
    canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
            drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });
        }
    }
    canvasCtx.restore();

    window.requestAnimationFrame(predictWebcam);
}

setup();

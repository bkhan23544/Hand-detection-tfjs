const stats = new Stats();
stats.showPanel(0);
document.body.prepend(stats.domElement);

let model, ctx, videoWidth, videoHeight, video, canvas;

const state = {
  backend: 'webgl'
};

const gui = new dat.GUI();
gui.add(state, 'backend', ['wasm', 'webgl', 'cpu']).onChange(async backend => {
  await tf.setBackend(backend);
  addFlagLables();
});

async function addFlagLables() {
  if(!document.querySelector("#simd_supported")) {
    const simdSupportLabel = document.createElement("div");
    simdSupportLabel.id = "simd_supported";
    simdSupportLabel.style = "font-weight: bold";
    const simdSupported = await tf.env().getAsync('WASM_HAS_SIMD_SUPPORT');
    simdSupportLabel.textContent = `SIMD supported: ${simdSupported}`;
    document.querySelector("#description").appendChild(simdSupportLabel);
  }

  if(!document.querySelector("#threads_supported")) {
    const threadSupportLabel = document.createElement("div");
    threadSupportLabel.id = "threads_supported";
    threadSupportLabel.style = "font-weight: bold";
    const threadsSupported = await tf.env().getAsync('WASM_HAS_MULTITHREAD_SUPPORT');
    threadSupportLabel.textContent = `Threads supported: ${threadsSupported}`;
    document.querySelector("#description").appendChild(threadSupportLabel);
  }
}

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction = async () => {
  stats.begin();
    const predictions = await model.estimateHands(video);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    if (predictions.length > 0) {
     console.log(predictions[0].boundingBox.topLeft,"pred")
   var topx = predictions[0].boundingBox.topLeft[0]
   var topy = predictions[0].boundingBox.topLeft[1]
   var bottomx = predictions[0].boundingBox.bottomRight[0]
   var bottomy = predictions[0].boundingBox.bottomRight[1]
   ctx.beginPath()
   ctx.rect(topx,topy,bottomx-topx,bottomy-topy)
   ctx.lineWidth = 6
   ctx.strokeStyle = 'red'
   ctx.fillStyle = 'red'
   ctx.stroke()
    }

//   const returnTensors = false;
//   const flipHorizontal = true;
//   const annotateBoxes = true;
//   const predictions = await model.estimateFaces(
//     video, returnTensors, flipHorizontal, annotateBoxes);

//   if (predictions.length > 0) {
//     ctx.clearRect(0, 0, canvas.width, canvas.height);

//     for (let i = 0; i < predictions.length; i++) {
//       if (returnTensors) {
//         predictions[i].topLeft = predictions[i].topLeft.arraySync();
//         predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
//         if (annotateBoxes) {
//           predictions[i].landmarks = predictions[i].landmarks.arraySync();
//         }
//       }

//       const start = predictions[i].topLeft;
//       const end = predictions[i].bottomRight;
//       const size = [end[0] - start[0], end[1] - start[1]];
//       ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
//       ctx.fillRect(start[0], start[1], size[0], size[1]);

//       if (annotateBoxes) {
//         const landmarks = predictions[i].landmarks;

//         ctx.fillStyle = "blue";
//         for (let j = 0; j < landmarks.length; j++) {
//           const x = landmarks[j][0];
//           const y = landmarks[j][1];
//           ctx.fillRect(x, y, 5, 5);
//         }
//       }
//     }
//   }

  stats.end();

  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await tf.setBackend(state.backend);
  addFlagLables();
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('canvas');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

  model = await handpose.load();

  renderPrediction();
};

setupPage();

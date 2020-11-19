import logo from './logo.svg';
import React ,{useEffect} from 'react' 
import './App.css';
import * as tf from '@tensorflow/tfjs'
import { loadGraphModel } from '@tensorflow/tfjs-converter'
import FPSStats from "react-fps-stats";
import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';
import { createBrowserHistory } from "history";

const MODEL_URL = "web_model/model.json"
var imgWidth = 640
var imgHeight = 480

function App() {

const history = createBrowserHistory()
const [isVideoStreamReady,setIsVideoStreamReady] = React.useState(false)
const [isModelReady,setIsModelReady] = React.useState(false)
const [backend,setBackend] = React.useState('wasm')
const [fps,setFps] = React.useState(0)


useEffect(()=>{
  initWebcamStream()
    loadCustomModel()



},[])


const loadCustomModel =async()=> {
  setWasmPaths('./tfjs-backend-wasm.wasm')
tf.setBackend('wasm').then(()=>console.log("backend is "+tf.getBackend()))
  addFlagLables()
  // load the model with loadGraphModel
 var models =  await loadGraphModel(MODEL_URL)
      setIsModelReady(true)
      document.getElementById("video").onloadeddata = function(){
       
          detectObjects(models)
      
       
    }
}

const addFlagLables=async()=> {
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


 const initWebcamStream =async()=> {
   var video = document.getElementById("video")
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function (stream) {
        setIsVideoStreamReady(true)
        video.srcObject = stream;
      })
      .catch(function (error) {
        console.log("Something went wrong!");
      });
  }
  
  }


  const detectObjects = async (models) => {

    if (models) {
      var t0 = performance.now()
      var img = document.getElementById("video")
      const tfImg = tf.browser.fromPixels(img)
      const smallImg = tf.image.resizeBilinear(tfImg, [100, 100]) // 600, 450
      const resized = tf.cast(smallImg, 'int32')
      const tf4d = tf.tensor4d(Array.from(resized.dataSync()), [1, 100, 100, 3])
      const tf4d1 = tf.cast(tf4d, 'int32') // 600, 450
      
      let predictions = await models.executeAsync({ input_tensor: tf4d1 }, ['StatefulPartitionedCall/Postprocessor/BatchMultiClassNonMaxSuppression/stack_6', 'StatefulPartitionedCall/Postprocessor/Cast_4', 'StatefulPartitionedCall/Postprocessor/BatchMultiClassNonMaxSuppression/stack_7'])
      // console.log(predictions[2].dataSync(),"prediction")
      var t1 = performance.now()

      renderPredictionBoxes(predictions[0].dataSync(), predictions[1].dataSync(), predictions[2].dataSync())
      tfImg.dispose()
      smallImg.dispose()
      resized.dispose()
      tf4d.dispose()
setFps(Math.round(1000/(t1-t0)))
    }
    requestAnimationFrame(() => detectObjects(models));

  }

  const renderPredictionBoxes = async (predictionBoxes, totalPredictions, predictionScores) => {
    // get the context of canvas
    const ctx = document.getElementById("canvas").getContext("2d")
    // clear the canvas
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    // draw results
    for (let i = 0; i < totalPredictions; i++) {
      const minY = predictionBoxes[i * 4] * imgHeight
      const minX = predictionBoxes[i * 4 + 1] * imgWidth
      const maxY = predictionBoxes[i * 4 + 2] * imgHeight
      const maxX = predictionBoxes[i * 4 + 3] * imgWidth
      const score = predictionScores[i * 3] * 100
      if (score > 75) {
        ctx.beginPath()
        ctx.rect(minX, minY, maxX - minX, maxY - minY)
        ctx.lineWidth = 6
        ctx.strokeStyle = 'red'
        ctx.fillStyle = 'red'
        ctx.stroke()
        ctx.font = '30px Arial bold'
          ctx.fillText(
            `${score.toFixed(1)} Hand`,
            minX,
            minY > 10 ? minY - 5 : 10
          )
        // ctx.fillRect((minX + maxX) / 2,(minY + maxY) / 2,6,6);
      }
    }
  }

  const changeBackend=async(e)=>{
    history.push(`/`)
    history.go(0)
    
  }




  return (
    <div className="App">
        <p>{fps} Fps</p>
<div>
  <label>Select Backend:</label><select onChange={changeBackend} defaultValue={backend}>
    <option value="wasm">wasm</option>
    <option value="webgl">webgl</option>
  </select>
  {!isVideoStreamReady && <h3>Starting Webcam</h3>}
  {!isModelReady && <h3>Loading Model..</h3>}
  <div className="video-div">

  <div className="row">
  <video width={imgWidth} height={imgHeight} autoPlay id="video"></video>
      <canvas width={imgWidth} height={imgHeight} id="canvas"></canvas>
    </div>
    <div id="description">

    </div>
  

    </div>

</div>
    </div>
  );
}

export default App;

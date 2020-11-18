import logo from './logo.svg';
import React ,{useEffect} from 'react' 
import './App.css';
import * as tf from '@tensorflow/tfjs'
import { loadGraphModel } from '@tensorflow/tfjs-converter'
import FPSStats from "react-fps-stats";
import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';

const MODEL_URL = "web_model/model.json"
var imgWidth = 640
var imgHeight = 480
var array = new Array(91)
console.log(array)

function App() {

const [isVideoStreamReady,setIsVideoStreamReady] = React.useState(false)
const [isModelReady,setIsModelReady] = React.useState(false)
const [model,setModel] = React.useState()


useEffect(()=>{
  initWebcamStream()
    loadCustomModel()



},[])


const loadCustomModel =async()=> {
  // setWasmPaths('tfjs-wasm');
  tf.setBackend('webgl')
  // load the model with loadGraphModel
 var models =  await loadGraphModel(MODEL_URL)
      setModel(models)
      setIsModelReady(true)
      // console.log('model loaded: ', models)
      document.getElementById("video").onloadeddata = function(){
      setInterval(() => {
        detectObjects(models)
      });
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


  const detectObjects =async(models)=> {
    if(models){
var img = document.getElementById("video")
    const tfImg = tf.browser.fromPixels(img)
    const smallImg = tf.image.resizeBilinear(tfImg, [70, 70]) // 600, 450
    const resized = tf.cast(smallImg, 'int32')
    const tf4d = tf.tensor4d(Array.from(resized.dataSync()), [1, 70, 70, 3])
    const tf4d1 = tf.cast(tf4d, 'int32') // 600, 450
    let predictions = await models.executeAsync({ input_tensor: tf4d1 }, ['StatefulPartitionedCall/Postprocessor/BatchMultiClassNonMaxSuppression/stack_6','StatefulPartitionedCall/Postprocessor/Cast_4','StatefulPartitionedCall/Postprocessor/BatchMultiClassNonMaxSuppression/stack_7'])
    // console.log(predictions[2].dataSync(),"prediction")
    renderPredictionBoxes(predictions[0].dataSync(),predictions[1].dataSync(),predictions[2].dataSync())
    tfImg.dispose()
    smallImg.dispose()
    resized.dispose()
    tf4d.dispose()
  }

  }

  const renderPredictionBoxes =async(predictionBoxes, totalPredictions,predictionScores)=> {
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
      const score = predictionScores[i*3] * 100
      if (score > 75) {
        ctx.beginPath()
        ctx.rect(minX, minY, maxX - minX, maxY - minY)
        ctx.lineWidth = 6
        ctx.strokeStyle = 'red'
        ctx.fillStyle = 'red'
        ctx.stroke()
      }
    }
  }




  return (
    <div className="App">
       <FPSStats />
<div>
  {!isVideoStreamReady && <h3>Initializing Webcam Stream..</h3>}
  <h3>Loading Model..</h3>
  <h3>Failed to init stream and/or model -</h3>
  <div className="video-div">

  <div className="row">
  <video width={imgWidth} height={imgHeight} autoPlay id="video"></video>
      <canvas width={imgWidth} height={imgHeight} id="canvas"></canvas>
    </div>
  

    </div>

</div>
    </div>
  );
}

export default App;

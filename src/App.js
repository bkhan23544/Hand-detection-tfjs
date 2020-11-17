import logo from './logo.svg';
import React ,{useEffect} from 'react' 
import './App.css';
import * as tf from '@tensorflow/tfjs'
import { loadGraphModel } from '@tensorflow/tfjs-converter'
import FPSStats from "react-fps-stats";

const MODEL_URL = "web_model/web_model/model.json"

function App() {

const [isVideoStreamReady,setIsVideoStreamReady] = React.useState(false)
const [isModelReady,setIsModelReady] = React.useState(false)
const [model,setModel] = React.useState()


useEffect(()=>{
  initWebcamStream()
    loadCustomModel()



},[])


const loadCustomModel =async()=> {
  // load the model with loadGraphModel
 var models =  await tf.loadGraphModel(MODEL_URL)
      setModel(models)
      setIsModelReady(true)
      console.log('model loaded: ', models)
      detectObjects(models)
      
    // .catch((error) => {
    //   console.log('failed to load the model', error)
    //   throw (error)
    // })
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
    console.log("entered")
    if(models){
      console.log("inside")
var img = document.createElement("img")
img.src = "hand.jpg"
img.width=2063
img.height = 1523
    const tfImg = tf.browser.fromPixels(img)
    console.log(tfImg,"tfimg")
    const smallImg = tf.image.resizeBilinear(tfImg, [300, 300]) // 600, 450
    const resized = tf.cast(smallImg, 'int32')
    console.log(resized,"resizedimg")
    const tf4d = tf.tensor4d(Array.from(resized.dataSync()), [1, 300, 300, 3])
    const tf4d1 = tf.cast(tf4d, 'int32') // 600, 450
    console.log(tf4d1,"tf4dimg")
    let predictions = await models.executeAsync({ input_tensor: tf4d1 }, ['detection_scores'])
    console.log(predictions,"prediction")
    // renderPredictionBoxes(predictions[0].dataSync(), predictions[1].dataSync(), predictions[2].dataSync(), predictions[3].dataSync())
    tfImg.dispose()
    smallImg.dispose()
    resized.dispose()
    tf4d.dispose()
    // requestAnimationFrame(() => {
    //   detectObjects()
    // })
  }
  else{
    setTimeout(() => {
      detectObjects()
    }, 1000);
   
  }
  }

  const renderPredictionBoxes =async(predictionBoxes, totalPredictions, predictionClasses, predictionScores)=> {
    // get the context of canvas
    const ctx = document.getElementById("canvas").getContext("2d")
    // clear the canvas
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    // draw results
    for (let i = 0; i < totalPredictions[0]; i++) {
      const minY = predictionBoxes[i * 4] * 450
      const minX = predictionBoxes[i * 4 + 1] * 600
      const maxY = predictionBoxes[i * 4 + 2] * 450
      const maxX = predictionBoxes[i * 4 + 3] * 600
      const score = predictionScores[i * 3] * 100
      if (score > 75) {
        ctx.beginPath()
        ctx.rect(minX, minY, maxX - minX, maxY - minY)
        ctx.lineWidth = 3
        ctx.strokeStyle = 'red'
        ctx.fillStyle = 'red'
        ctx.stroke()
        ctx.shadowColor = 'white'
        ctx.shadowBlur = 10
        ctx.font = '14px Arial bold'
        ctx.fillText(
          `${score.toFixed(1)} - Jagermeister bottle`,
          minX,
          minY > 10 ? minY - 5 : 10
        )
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
  <video autoPlay id="video"></video>
      <canvas id="canvas"></canvas>
    </div>
    </div>

</div>
    </div>
  );
}

export default App;

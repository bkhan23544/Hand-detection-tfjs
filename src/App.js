import logo from './logo.svg';
import React ,{useEffect} from 'react' 
import './App.css';
import * as tf from '@tensorflow/tfjs'
import { loadGraphModel } from '@tensorflow/tfjs-converter'
import FPSStats from "react-fps-stats";
import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';
import { createBrowserHistory } from "history";
import { BrowserRouter as Router,Route } from 'react-router-dom';

import Wasm from './wasm-backend'
import WebGL from './webgl-backend'



function App() {



  return (
    <div className="App">
<Router>
  <Route exact path='/wasm' component={Wasm}/>
  <Route exact path='/' component={WebGL}/>
</Router>
    </div>
  );
}

export default App;

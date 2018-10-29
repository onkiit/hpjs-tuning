import React from 'react';
import * as tf from '@tensorflow/tfjs';
import * as hpjs from 'hyperparameters';
import {generateData} from './data'

const optimizers = {
  sgd: tf.train.sgd,
  adagrad: tf.train.adagrad,
  adam: tf.train.adam, 
  adamax: tf.train.adamax,
  rmsprop: tf.train.rmsprop,
}

const trainData = generateData()

class App extends React.Component { 
  componentDidMount() {
    this.hyperTFJS();
  }
    // An optimization function. The parameters are optimizer and epochs and will use the loss returned by the fn to measure which parameters are "best"
    // Input and output data are passed as second argument
    optFunction = async ({ learningRate, optimizer }, { xs, ys }) => {

      // Create a simple sequential model.
      const model = tf.sequential();


      // add a dense layer to the model and compile
      model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
      model.compile({
        loss: 'meanSquaredError',
        optimizer: optimizers[optimizer](learningRate),
      });

      // train model using defined data
      const h = await model.fit(xs, ys, { epochs: 135 });

      //printint out each optimizer and its loss
      console.log(optimizer);
      console.log('learning rate: ', learningRate, 'loss: ', h.history.loss[h.history.loss.length - 1]);
      return { loss: h.history.loss[h.history.loss.length - 1], status: hpjs.STATUS_OK } ;
    };

    hyperTFJS = async () => {

      // Generating some data for training (y = 2x - 1) in tf tensors and defining its shape
      const xs = trainData.xs
      const ys = trainData.ys
      

      // defining a search space we want to optimize. Using hpjs parameters here
      const space = {
        learningRate: hpjs.uniform(0.0001, 0.2),
        optimizer: hpjs.choice(['sgd', 'adagrad', 'adam', 'adamax', 'rmsprop']),
      };

      // finding the optimal hyperparameters using hpjs.fmin. Here, 6 is the # of times the optimization function will be called (this can be changed)
      const trials = await hpjs.fmin(
        this.optFunction, space, hpjs.search.randomSearch, 10,
        { rng: new hpjs.RandomState(654321), xs, ys }
      );

      const opt = trials.argmin;

      //printing out data
      console.log('trials', trials);
      console.log('best optimizer:', opt.optimizer);
      console.log('best learning rate:', opt.learningRate);
    }
    render() {
      return( 
        <div>  
        </div> 
      );
  }
}
export default App;
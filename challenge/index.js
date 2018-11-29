const tf = require('@tensorflow/tfjs');
const iris = require('./iris.json');
const irisTesting = require('./testingIris.json');

const trainingData = tf.tensor2d(iris.map(item=> [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]),[130,4]);

const testingData = tf.tensor2d(irisTesting.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]), [14, 4]);

// convert data


// creating model


// compiling model


// predicting model
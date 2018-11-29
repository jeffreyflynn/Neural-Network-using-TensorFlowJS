const tf = require('@tensorflow/tfjs');
const iris = require('./iris.json');
const irisTesting = require('./testingIris.json');

const trainingData = tf.tensor2d(iris.map(item=> [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]),[130,4]);

const testingData = tf.tensor2d(irisTesting.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]), [14, 4]);

/* The output will be based on the neuron activation. 
We will write the output function such that we get an array of length three 
every time with one of the values closer to one and the rest two closer to zero. */
const outputData = tf.tensor2d(iris.map(item => [
  item.species === 'setosa' ? 1 : 0,
  item.species === 'virginica' ? 1 : 0,
  item.species === 'versicolor' ? 1 : 0
]), [130,3]);

// convert data


// creating model


// compiling model


// predicting model
const tf = require('@tensorflow/tfjs');
const iris = require('./training.json');
const irisTesting = require('./testing.json');

// convert training data
const trainingData = tf.tensor2d(iris.map(item=> [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]),[130,4]);

// convert testing data
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

// creating the model
const model = tf.sequential();

// adding input layer to model
model.add(tf.layers.dense({
  inputShape: [4],
  activation: "sigmoid",
  units: 10
}));

// adding output layer to model
model.add(tf.layers.dense({
  inputShape: [10],
  activation: "softmax",
  units: 3
}));

// compiling model
model.compile({
  loss: "categoricalCrossEntropy",
  optimizer: tf.train.adam()
});

// predicting model
async function train_data(){
  console.log('......Loss History.......');

  for(let i = 0; i < 15; i++){
    const res = await model.fit(trainingData, outputData,{epochs: 40});  
    console.log(`Iteration ${i}: ${res.history.loss[0]}`);         
  }
}

async function main() {
  let train = await train_data();
  console.log('....Model Prediction .....');
  model.predict(testingData).print();
}

main();
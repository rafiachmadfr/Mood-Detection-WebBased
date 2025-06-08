let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var neutralSamples = 0, happySamples = 0, sadSamples = 0, angrySamples = 0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  console.log('MobileNet loaded with output shape:', layer.outputShape);
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(4); // Updated for 4 classes

  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: [7, 7, 1024]}),
      tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernel_regularizer: tf.regularizers.l2({l2: 0.01})
      }),
      tf.layers.dropout({rate: 0.5}),
      tf.layers.dense({units: 64, activation: 'relu'}),
      tf.layers.dense({units: 4, activation: 'softmax'}) // Updated for 4 classes
    ]
  });

  const optimizer = tf.train.adam(0.0001);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  console.log('Starting training...');
  await model.fit(dataset.xs, dataset.ys, {
    epochs: 50,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(5)}, Accuracy = ${logs.acc.toFixed(5)}`);
      }
    }
  });
  console.log('Training completed.');
}

function handleButton(elem) {
  switch (elem.id) {
    case "0":
      neutralSamples++;
      document.getElementById("neutralsamples").innerText = "Neutral samples: " + neutralSamples;
      break;
    case "1":
      happySamples++;
      document.getElementById("happysamples").innerText = "Happy samples: " + happySamples;
      break;
    case "2":
      sadSamples++;
      document.getElementById("sadsamples").innerText = "Sad samples: " + sadSamples;
      break;
    case "3":
      angrySamples++;
      document.getElementById("angrysamples").innerText = "Angry samples: " + angrySamples;
      break;
  }
  const label = parseInt(elem.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    let predictionText = "";
    switch (classId) {
      case 0:
        predictionText = "I see Neutral";
        break;
      case 1:
        predictionText = "I see Happy";
        break;
      case 2:
        predictionText = "I see Sad";
        break;
      case 3:
        predictionText = "I see Angry";
        break;
    }
    document.getElementById("prediction").innerText = predictionText;
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function doTraining() {
  train().then(() => alert("Training Done!")).catch(err => console.error('Training failed:', err));
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
}

function saveModel() {
  model.save('downloads://my_model');
}

async function init() {
  try {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture())); // Warm up MobileNet
    console.log('Initialization completed.');
  } catch (error) {
    console.error('Initialization failed:', error);
  }
}

init();

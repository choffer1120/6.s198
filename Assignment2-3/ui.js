
import embed from 'vega-embed';

const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const imagesElement = document.getElementById('images');

export function isTraining() {
  statusElement.innerText = 'Training...';
}
export function trainingLog(message) {
  messageElement.innerText = `${message}\n`;
  console.log(message);
}

export function showTestResults(batch, predictions, labels, img_shape, label_id) {
  statusElement.innerText = 'Testing...';
  const testExamples = batch.xs.shape[0];
  let totalCorrect = 0;
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    canvas.className = 'prediction-canvas';
    renderToCanvas(image.flatten().dataSync(), canvas, img_shape);

    const inx = document.createElement('div');
    inx.innerText = `index: ${batch.indices[i]}`;

    const pred = document.createElement('div');

    const prediction = (label_id == null ? predictions[i] : label_id[predictions[i]]);
    const label = labels[i]
    const correct = predictions[i] === label;
    if (correct) {
      totalCorrect += 1;
    }

    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `pred: ${prediction}`;

    div.appendChild(inx);
    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }

  document.getElementById('last-test-acc').innerText = (totalCorrect/testExamples)
}

const lossLabelElement = document.getElementById('loss-label');
const accuracyLabelElement = document.getElementById('accuracy-label');
export function plotLosses(lossValues) {
  embed(
      '#lossCanvas', {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'values': lossValues},
        'mark': {'type': 'line'},
        'width': 260,
        'orient': 'vertical',
        'encoding': {
          'x': {'field': 'batch', 'type': 'ordinal'},
          'y': {'field': 'loss', 'type': 'quantitative'},
          'color': {'field': 'set', 'type': 'nominal', 'legend': null},
        }
      },
      {width: 360});
  lossLabelElement.innerText =
      'last loss: ' + lossValues[lossValues.length - 1].loss.toFixed(2);
}

export function plotAccuracies(accuracyValues) {
  embed(
      '#accuracyCanvas', {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'values': accuracyValues},
        'width': 260,
        'mark': {'type': 'line', 'legend': null},
        'orient': 'vertical',
        'encoding': {
          'x': {'field': 'batch', 'type': 'ordinal'},
          'y': {'field': 'accuracy', 'type': 'quantitative'},
          'color': {'field': 'set', 'type': 'nominal', 'legend': null},
        }
      },
      {'width': 360});
  accuracyLabelElement.innerText = 'last accuracy: ' +
      (accuracyValues[accuracyValues.length - 1].accuracy * 100).toFixed(2) +
      '%';
}

function renderToCanvas(a, canvas, shape) {
  const [height, width, depth] = shape;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = a;
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * depth;
    if (depth > 1) {
      imageData.data[j + 0] = Math.round(255 * data[k + 0]);
      imageData.data[j + 1] = Math.round(255 * data[k + 1]);
      imageData.data[j + 2] = Math.round(255 * data[k + 2]);
      imageData.data[j + 3] = 255;
    } else {
      const pixel = Math.round(255 * data[k]);
      imageData.data[j+0] = pixel;
      imageData.data[j+1] = pixel;
      imageData.data[j+2] = pixel;
      imageData.data[j+3] = 255;
    }
      ctx.putImageData(imageData, 0, 0);
   }
 }
export function showResults(classes, label_id) {
  const predictionsElement = document.getElementById('predictions');
  predictionsElement.removeChild(predictionsElement.lastChild);
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const probsContainer = document.createElement('div');

  const row = document.createElement('div');
  row.className = 'row';

  const classTitle = document.createElement('div');
  classTitle.className = 'cell';
  classTitle.innerText = 'Class';
  row.appendChild(classTitle);

  const probsTitle = document.createElement('div');
  probsTitle.className = 'cell';
  probsTitle.innerText = 'Prob.';
  row.appendChild(probsTitle);

  probsContainer.appendChild(row);
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = (label_id == null ? i : label_id[i]);
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

export function showimage(image, shape) {
  renderToCanvas(image, document.getElementById("imageHolder"), shape);
}

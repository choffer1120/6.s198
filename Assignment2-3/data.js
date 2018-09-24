let IMAGE_SIZE;
let IMAGE_WIDTH;
let IMAGE_HEIGHT;
let IMAGE_DEPTH;
let NUM_CLASSES;
let NUM_DATASET_ELEMENTS;
let TRAIN_TEST_RATIO;
let NUM_TEST_ELEMENTS;
let NUM_TRAIN_ELEMENTS;

let IMAGES_SPRITE_PATH;
let LABELS_PATH;

import * as tf from '@tensorflow/tfjs';

/**
 * A class that fetches the sprited dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class Data {
  constructor(image_path, labels_path, images_shape, labels_shape, train_test_ratio) {
    IMAGES_SPRITE_PATH = image_path;
    LABELS_PATH = labels_path;

    [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH] = images_shape;
    IMAGE_SIZE = IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH;
    NUM_CLASSES = labels_shape[0];

    TRAIN_TEST_RATIO = train_test_ratio 

    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  async load() {
    // Make a request for the sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;
        
        NUM_DATASET_ELEMENTS = img.height;
        NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO*NUM_DATASET_ELEMENTS);
        NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
        
        const datasetBytesBuffer =
            new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * chunkSize * IMAGE_SIZE * 4,
              IMAGE_SIZE * chunkSize);
          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            for (let k=0; k<IMAGE_DEPTH; k++){
              datasetBytesView[j*IMAGE_DEPTH+k] = imageData.data[j * 4 + k] / 255;
            }
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);
        resolve();
      };
      img.src = IMAGES_SPRITE_PATH;
    });
   
    const labelsRequest = fetch(LABELS_PATH);
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages =
        this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
        this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
        this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
        batchSize, [this.trainImages, this.trainLabels], () => {
          this.shuffledTrainIndex =
              (this.shuffledTrainIndex + 1) % this.trainIndices.length;
          return this.trainIndices[this.shuffledTrainIndex];
        });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextTotalBatch(batchSize) {
    return this.nextBatch(batchSize, [this.datasetImages, this.datasetLabels], () => {
      const indices = []
      for (let i = 0; i <= batchSize; i++){
        indices.push(Math.floor(Math.random()*NUM_DATASET_ELEMENTS))
      }
      return Math.floor(Math.random()*NUM_DATASET_ELEMENTS)
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);
    const indices = [];
    for (let i = 0; i < batchSize; i++) {
      const idx = index();
      indices.push(idx);
      const image =
          data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label =
          data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels, indices};
  }

  getNthImage(n) {
    const start = n * IMAGE_SIZE;
    const end = start + IMAGE_SIZE;
    return this.datasetImages.slice(start, end);
  }
}

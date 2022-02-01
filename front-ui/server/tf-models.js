const tf = require("@tensorflow/tfjs-node");
const fastifyPlugin = require("fastify-plugin");
const path = require("path");
var pixels = require("image-pixels");
// const { loadImage } = require("canvas");
async function loadModel(fastify, options) {
  try {
    model_path = path.resolve(
      __dirname,
      "../",
      "models",
      "image-classifier",
      "model.json"
    );
    const model = await tf.loadLayersModel("file://" + model_path);

    fastify.decorate("tf_model_image_classifier", model);

    mapIndexToClassType = ["1", "10", "2", "3", "4", "5", "6", "7", "8", "9"];

    async function processImage(path_to_image) {
      let image_pixels = await pixels(path_to_image);
      let tensor = tf.browser
        .fromPixels(image_pixels, 3)
        .resizeBilinear([300, 300])
        .toFloat()
        .expandDims(0);
      const result = await model.predict(tensor).data();
      const index_type = tf.argMax(result).dataSync()[0];
      return mapIndexToClassType[index_type];
    }

    fastify.decorate("tf_func_image_classifier", processImage);
  } catch (e) {
    console.log("Issues setting up tensorflow models!");
    console.log(e);
  }
}

module.exports = fastifyPlugin(loadModel);

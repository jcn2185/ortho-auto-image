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
      "eo-feature",
      "model.json"
    );
    console.log("Loading model at:", model_path);
    const model = await tf.loadLayersModel("file://" + model_path);

    fastify.decorate("tf_model_eo_attribute", model);

    mapIndexToClassType = ["1", "10", "2", "3", "4", "5", "6", "7", "8", "9"];

    async function processImage(path_to_image) {
      let image_pixels = await pixels(path_to_image);
      try {
        let tensor = tf.browser
          .fromPixels(image_pixels, 3)
          .resizeBilinear([512, 512])
          .toFloat()
          .expandDims(0);
        console.log(tensor);

        return await model.predict(tensor).data();
      } catch (error) {
        console.log(error);
        return null;
      }
    }

    fastify.decorate("tf_func_eo_feature", processImage);
  } catch (e) {
    console.log("Issues setting up tensorflow models!");
    console.log(e);
  }
}

module.exports = fastifyPlugin(loadModel);

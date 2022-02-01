const pump = require("pump");
const fs = require("fs");
async function routes(fastify, opts) {
  const classifyImage = fastify.tf_func_image_classifier;
  const feature_extractor_left = fastify.tf_func_eo_feature;

  fastify.post("/", async function(req, reply) {
    // stores files to tmp dir and return files
    const parts = req.files();
    files_to_process = [];
    for await (const part of parts) {
      let storagePath = "files/" + part.filename;
      files_to_process.push({
        url: storagePath,
        name: part.filename,
      });
      await pump(part.file, fs.createWriteStream(storagePath));
    }

    // Now, lets process them
    for (file of files_to_process) {
      console.log("Classifying image:", file.url);
      file.classification = await classifyImage(file.url);
      console.log(file.url, file.classification);
      if (file.classification == 6) {
        console.log("found the side shot....");
        file.attributes = await feature_extractor_left(file.url);
        console.log(file.attributes);
      }
    }

    console.log("Finished classifying image...");

    return files_to_process;
  });
}

module.exports = routes;

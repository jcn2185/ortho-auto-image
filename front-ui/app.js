const Fastify = require("fastify");
const path = require("path");
const url = require("url");
const AutoLoad = require("fastify-autoload");
const fastify = Fastify();

fastify.register(require("fastify-static"), {
  root: path.join(__dirname, "files"),
  prefix: "/files/",
});
fastify.register(require("fastify-static"), {
  root: path.join(__dirname, "dist"),
  decorateReply: false,
});

fastify.register(require("fastify-multipart"), {
  limits: {
    fieldNameSize: 100, // Max field name size in bytes
    fieldSize: 1000000, // Max field value size in bytes
    fields: 10, // Max number of non-file fields
    fileSize: 1000000, // For multipart forms, the max file size
    files: 10, // Max number of file fields
    headerPairs: 2000, // Max number of header key=>value pairs
  },
});
fastify.register(import("./server/tf-models.js"));
fastify.register(import("./server/tf-eo-feature.js"));
fastify.register(AutoLoad, {
  dir: path.join(__dirname, "routes"),
});

// fastify.register(import("./server/route.js"));

// // Launch the server
// const start = async () => {
//   try {
//     await fastify.listen(process.env.PORT || 3000);
//   } catch (err) {
//     console.log(err);
//     process.exit(10);
//   }
// };

console.log("lets look at the process:");
console.log(process.env.PORT || 8080);

// start();
const port = process.env.PORT ? parseInt(process.env.PORT) : 3000;
fastify.listen({ port, host: "0.0.0.0" }, (err, address) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`Server listening at ${address}`);
});

module.exports = async function(fastify, opts) {
  fastify.get("/", async function(request, reply) {
    return "Hello from root";
  });
  fastify.get("/sup", async function(request, reply) {
    return "hello from sup";
  });
};

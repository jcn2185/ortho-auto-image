async function route(fastify, opts) {
  fastify.get("/hi", async function(request, reply) {
    return "We are from here!";
  });
}

module.exports = route;

// Require the framework and instantiate it
const fastify = require('fastify')({ logger: true })
const path = require('path')

fastify.register(require('fastify-static'), {
  root: path.join(__dirname, 'public'),
  prefix: '/public/' // optional: default '/'
})

// Declare a route
fastify.get('/', async (request, reply) => {
  return { hello: 'world' }
})

// Run the server!
const start = async () => {
  try {
    await fastify.listen(8081)
  } catch (err) {
    fastify.log.error(err)
    process.exit(1)
  }
}
start()

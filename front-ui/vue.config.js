module.exports = {
  devServer: {
    proxy: {
      "^/api": {
        target: "http://localhost:3000",
        changeOrigin: true,
      },
      "^/files": {
        target: "http://localhost:3000",
        changeOrigin: true,
      },
    },
  },
};

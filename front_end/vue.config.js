module.exports = {
  transpileDependencies: ['/@yabby-business/'],
  devServer: {
    // host: 'localhost',
    port: 8080,
    proxy: {
      '/api' : {
        target: 'http://172.25.19.95:5000',
        ws: true,
        changeOrigin: true,
        pathRewrite: {
          '^/api': ''
        }
      }
    },
    // disableHostCheck: true,
  }
}

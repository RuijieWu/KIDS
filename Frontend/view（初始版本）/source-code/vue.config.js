module.exports = {
  lintOnSave: false,
  devServer: {
    proxy: {
      '/alarm/message/list': {
        target: 'http://43.138.200.89:8080',
        changeOrigin: true
      },
      '/kairos': {
        target: 'http://43.138.200.89:8080',
        changeOrigin: true
      },
      '/blacklist': {
        target: 'http://43.138.200.89:8080',
        changeOrigin: true
      },
      '/data': {
        target: 'http://43.138.200.89:8080',
        changeOrigin: true
      }
    }
  },
  configureWebpack: {
    optimization: {
      splitChunks: {
        chunks: 'all',
        minSize: 30000,
        maxSize: 0,
        minChunks: 1,
        maxAsyncRequests: 5,
        maxInitialRequests: 3,
        automaticNameDelimiter: '~',
        name: true,
        cacheGroups: {
          vendors: {
            test: /[\\/]node_modules[\\/]/,
            priority: -10
          },
          default: {
            minChunks: 2,
            priority: -20,
            reuseExistingChunk: true
          }
        }
      }
    }
  }
};
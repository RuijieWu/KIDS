module.exports = {
  lintOnSave: false,

    devServer: {
      proxy: {
        '/alarm/message/list': {
          target: 'http://43.138.200.89:8080',
          changeOrigin: true
        }
      }
    }
};

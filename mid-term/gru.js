// gru.js
// Exposes a simple GRU-based binary classifier compatible with tabular data by treating each row as a sequence of length 1.
// Input shape: [batch, time=1, features=D] → GRU(32) → Dense(16, relu) → Dense(1, sigmoid)

(() => {
  function buildGRUModel(featureDim) {
    const input = tf.input({ shape: [1, featureDim] }); // time=1, features=D
    // GRU block
    const g = tf.layers.gru({
      units: 32,
      activation: 'tanh',
      recurrentActivation: 'sigmoid',
      returnSequences: false,
      dropout: 0.1,
      recurrentDropout: 0.0,
    }).apply(input);
    // Dense head
    const h1 = tf.layers.dense({ units: 16, activation: 'relu' }).apply(g);
    const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(h1);
    const model = tf.model({ inputs: input, outputs: out });
    model.compile({
      optimizer: tf.train.adam(),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });
    return model;
  }

  window.ModelFactory = { buildGRUModel };
})();

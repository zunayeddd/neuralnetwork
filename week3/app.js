// app.js (drop-in)
(function(){
  function log(msg){
    const el = document.getElementById('trainingLogs');
    const div = document.createElement('div');
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.appendChild(div); el.scrollTop = el.scrollHeight;
  }
  function setDataStatus(trainCount,testCount){
    document.getElementById('dataStatus').innerHTML =
      `<h3>Data Status</h3><p>Train samples: ${trainCount}</p><p>Test samples: ${testCount}</p>`;
  }
  function setModelInfo(text){ document.getElementById('modelInfo').innerHTML = `<h3>Model Info</h3><p>${text}</p>`; }

  class MNISTApp {
    constructor(){
      this.data = new window.MNISTDataLoader();
      this.model = null;
      this.attachUI();
    }

    attachUI(){
      document.getElementById('loadDataBtn').addEventListener('click',()=>this.onLoadData());
      document.getElementById('trainBtn').addEventListener('click',()=>this.onTrain());
      document.getElementById('evaluateBtn').addEventListener('click',()=>this.onEvaluate());
      document.getElementById('testFiveBtn').addEventListener('click',()=>this.onTestFive());
      document.getElementById('saveModelBtn').addEventListener('click',()=>this.onSave());
      document.getElementById('loadModelBtn').addEventListener('click',()=>this.onLoadModel());
      document.getElementById('resetBtn').addEventListener('click',()=>this.onReset());
      document.getElementById('toggleVisorBtn').addEventListener('click',()=>tfvis.visor().toggle());
      log('Ready.');
    }

    async onLoadData(){
      try{
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile  = document.getElementById('testFile').files[0];
        if(!trainFile || !testFile){ log('ERROR: Select both train and test CSV files'); return; }
        log('Loading training data…');
        const tr = await this.data.loadTrainFromFiles(trainFile);
        log('Loading test data…');
        const te = await this.data.loadTestFromFiles(testFile);
        setDataStatus(tr.count, te.count);
        log('Data loaded successfully.');
      }catch(e){
        log('ERROR loading data: '+e.message);
        console.error(e);
      }
    }

    buildAutoencoder(){
      const input = tf.input({shape:[28,28,1]});
      let x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(input);
      x = tf.layers.maxPooling2d({poolSize:2,strides:2,padding:'same'}).apply(x);
      x = tf.layers.conv2d({filters:64,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
      x = tf.layers.maxPooling2d({poolSize:2,strides:2,padding:'same'}).apply(x);
      x = tf.layers.conv2dTranspose({filters:64,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
      x = tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
      const out = tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x);
      const m = tf.model({inputs:input,outputs:out});
      m.compile({optimizer: tf.train.adam(), loss: 'meanSquaredError'});
      setModelInfo(`CNN Autoencoder • loss: MSE • optimizer: Adam • params: ${m.countParams().toLocaleString()}`);
      return m;
    }

    async onTrain(){
      if(!this.data.trainData){ log('ERROR: Load data first'); return; }
      if(!this.model) this.model = this.buildAutoencoder();
      const epochs = parseInt(document.getElementById('epochs').value,10)||8;
      const batch  = parseInt(document.getElementById('batch').value,10)||128;
      const std    = parseFloat(document.getElementById('noiseStd').value)||0.5;

      const {trainXs, valXs} = this.data.splitTrainVal(this.data.trainData.xs, 0.1);
      const noisyTrain = this.data.addGaussianNoise(trainXs, std);
      const noisyVal   = this.data.addGaussianNoise(valXs, std);

      log(`Training… epochs=${epochs}, batch=${batch}, σ=${std}`);
      await this.model.fit(noisyTrain, trainXs, {
        epochs, batchSize:batch, shuffle:true, validationData:[noisyVal,valXs],
        callbacks: tfvis.show.fitCallbacks({name:'Loss',tab:'Charts'}, ['loss','val_loss'], {callbacks:['onEpochEnd']})
      });
      log('Training complete.');

      noisyTrain.dispose(); noisyVal.dispose(); trainXs.dispose(); valXs.dispose();
    }

    async onEvaluate(){
      if(!this.model || !this.data.testData){ log('ERROR: Need model and test data'); return; }
      const std = parseFloat(document.getElementById('noiseStd').value)||0.5;
      tf.tidy(()=>{
        const noisy = this.data.addGaussianNoise(this.data.testData.xs, std);
        const recon = this.model.predict(noisy);
        const mse = tf.losses.meanSquaredError(this.data.testData.xs, recon).mean();
        mse.data().then(v=>log(`Test MSE @ σ=${std}: ${v[0].toFixed(6)}`));
      });
    }

    async onTestFive(){
  try {
    if (!this.model || !this.data.testData) { log('ERROR: Need model and test data'); return; }
    const std = parseFloat(document.getElementById('noiseStd').value) || 0.5;

    // sample
    const { batchXs, count } = this.data.getRandomTestBatch(this.data.testData.xs, 5);
    log(`Rendering ${count} random samples @ σ=${std} …`);

    // noisy → reconstruct
    const noisy = this.data.addGaussianNoise(batchXs, std);
    const recon = this.model.predict(noisy);

    // draw
    const cont = document.getElementById('previewContainer');
    cont.innerHTML = '';
    for (let i = 0; i < count; i++) {
      const col = document.createElement('div'); col.className = 'preview-item';
      const c1 = document.createElement('canvas');
      const c2 = document.createElement('canvas');
      const c3 = document.createElement('canvas');

      const n = noisy.slice([i,0,0,0],[1,28,28,1]).squeeze();
      const r = recon.slice([i,0,0,0],[1,28,28,1]).squeeze();
      const c = batchXs.slice([i,0,0,0],[1,28,28,1]).squeeze();

      this.data.draw28x28ToCanvas(n, c1, 4);
      this.data.draw28x28ToCanvas(r, c2, 4);
      this.data.draw28x28ToCanvas(c, c3, 4);

      col.append(c1, c2, c3);
      cont.appendChild(col);

      n.dispose(); r.dispose(); c.dispose();
    }

    noisy.dispose(); recon.dispose(); batchXs.dispose();
    log('Preview rendered.');
  } catch (e) {
    log('ERROR in Test 5 Random: ' + e.message);
    console.error(e);
  }
}


    async onSave(){
      if(!this.model){ log('ERROR: No model to save'); return; }
      await this.model.save('downloads://mnist_denoiser');
      log('Model saved (download).');
    }

    async onLoadModel(){
      const jf = document.getElementById('modelJsonFile').files[0];
      const wf = document.getElementById('modelWeightsFile').files[0];
      if(!jf || !wf){ log('ERROR: Choose model.json and weights.bin'); return; }
      if(this.model) this.model.dispose();
      this.model = await tf.loadLayersModel(tf.io.browserFiles([jf,wf]));
      setModelInfo(`Loaded • params: ${this.model.countParams().toLocaleString()}`);
      log('Model loaded.');
    }

    onReset(){
      if(this.model){ this.model.dispose(); this.model=null; }
      this.data.dispose();
      document.getElementById('previewContainer').innerHTML='';
      setDataStatus(0,0); setModelInfo('No model loaded'); log('Reset complete.');
    }
  }

  document.addEventListener('DOMContentLoaded', ()=>{ window.__app = new MNISTApp(); });
})();

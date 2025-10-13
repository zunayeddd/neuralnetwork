
// data-loader.js (drop-in)
(function(){
  function splitSmart(line){
    // Handles: commas, semicolons, tabs, multiple spaces
    return line.trim().split(/[,;\t\s]+/);
  }

  window.MNISTDataLoader = class MNISTDataLoader {
    constructor(){ this.trainData=null; this.testData=null; }

    async loadCSVFile(file){
      return new Promise((resolve,reject)=>{
        const reader = new FileReader();
        reader.onload = e => {
          try{
            const txt = e.target.result;
            const lines = txt.split(/\r?\n/).filter(l=>l.trim().length);
            const labels = new Int32Array(lines.length);
            const pixels = new Float32Array(lines.length*784);

            let li=0, pi=0, count=0;
            for (const line of lines){
              const arr = splitSmart(line);
              if (arr.length < 785) continue; // skip short rows
              labels[li++] = parseInt(arr[0],10);
              for (let j=1;j<=784;j++) pixels[pi++] = parseFloat(arr[j]) / 255.0;
              count++;
            }
            if (count===0) throw new Error('No valid rows in CSV');

            const xs = tf.tensor4d(pixels, [count,28,28,1]);
            const ys = tf.oneHot(tf.tensor1d(labels.slice(0,count),'int32'), 10);
            resolve({ xs, ys, count });
          }catch(err){ reject(err); }
        };
        reader.onerror = ()=>reject(new Error('File read failed'));
        reader.readAsText(file);
      });
    }

    async loadTrainFromFiles(file){ this.trainData = await this.loadCSVFile(file); return this.trainData; }
    async loadTestFromFiles(file){ this.testData  = await this.loadCSVFile(file); return this.testData; }

    splitTrainVal(xs, valRatio=0.1){
      return tf.tidy(()=>{
        const n=xs.shape[0], nv=Math.floor(n*valRatio), nt=n-nv;
        const tr = xs.slice([0,0,0,0],[nt,28,28,1]);
        const va = xs.slice([nt,0,0,0],[nv,28,28,1]);
        return { trainXs:tr, valXs:va };
      });
    }

    addGaussianNoise(x,std=0.5){
      return tf.tidy(()=> tf.clipByValue(x.add(tf.randomNormal(x.shape,0,std,'float32')),0,1));
    }

    getRandomTestBatch(xs, k = 5){
  return tf.tidy(() => {
    const n = xs.shape[0];
    const kk = Math.min(k, n);
    if (kk === 0) { throw new Error('Test set has 0 samples'); }

    // tf.gather works best with a Tensor index; don’t pass a raw array
    const idxArr = tf.util.createShuffledIndices(n).slice(0, kk);
    const idx = tf.tensor1d(Array.from(idxArr), 'int32');
    const batch = tf.gather(xs, idx);
    idx.dispose();
    return { batchXs: batch, count: kk };
  });
}

    draw28x28ToCanvas(t, canvas, scale=4){
      const ctx = canvas.getContext('2d');
      const img = new ImageData(28,28);
      const data = t.reshape([28,28]).mul(255).dataSync();
      for(let i=0;i<784;i++){
        const v = data[i]|0;
        img.data[i*4+0]=v; img.data[i*4+1]=v; img.data[i*4+2]=v; img.data[i*4+3]=255;
      }
      const off = document.createElement('canvas');
      off.width=28; off.height=28;
      off.getContext('2d').putImageData(img,0,0);
      canvas.width=28*scale; canvas.height=28*scale;
      ctx.imageSmoothingEnabled=false;
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(off,0,0,28*scale,28*scale);
    }

    dispose(){
      if(this.trainData){ this.trainData.xs.dispose(); this.trainData.ys.dispose(); this.trainData=null; }
      if(this.testData){  this.testData.xs.dispose();  this.testData.ys.dispose();  this.testData=null;  }
    }
  };
})();

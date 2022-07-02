var model
async function loadModel() {

    // const test = copy the data from test.txt
    // const tensor = tf.tensor4d(test)

    // model = await tf.loadLayersModel('../Train Cnn/TFJS/model.json')
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/FraidoonOmarzai/MathAi/main/Train%20Cnn/TFJS/model.json')
   // console.log(model)
    // const result = model.predict(tensor)
    // const res = tf.argMax(result, axis=-1)
    // res.print()
}

function predictImage() {

    let img = cv.imread(c) // read the canvas that we drawed on
    cv.cvtColor(img,img,cv.COLOR_RGBA2GRAY,0) // convert from rgb to gray
    cv.threshold(img, img, 175, 255, cv.THRESH_BINARY) // threshhold the image

    // find the contours of the image
    let contours = new cv.MatVector()
    let hierarchy = new cv.Mat()
    cv.findContours(img, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    // Bounding Rect
    let cnt = contours.get(0)
    let rect = cv.boundingRect(cnt)
    img = img.roi(rect)

    // resize the image
    var h = img.rows
    var w = img.cols

    if (h > w){
        h = 20
        const scaleFactor = img.rows / h
        w = Math.round(img.cols / scaleFactor)
    }else{
        w = 20
        const scaleFactor = img.cols / w
        h = Math.round(img.rows / scaleFactor)
    }

    let newSize = new cv.Size(w, h)
    cv.resize(img, img, newSize, 0, 0, cv.INTER_AREA)


    // adding padding
    const L = Math.ceil(4+(20-w)/2)
    const R = Math.floor(4+(20-w)/2)
    const T = Math.ceil(4+(20-h)/2)
    const B = Math.floor(4+(20-h)/2)
    // console.log(`l:${L} r:${R} t:${T} b:${B}`)
    let b = new cv.Scalar(0, 0, 0, 0);
    cv.copyMakeBorder(img, img, T, B, L, R, cv.BORDER_CONSTANT, b)


    // Center of Mass
    cv.findContours(img, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours.get(0)
    let Moments = cv.moments(cnt, false) // Image moments help you to calculate some features like center of mass of the object

    const cx = Moments.m10 / Moments.m00
    const cy = Moments.m01 / Moments.m00  // m00 is the area of image or overall mass

    const X_SHIFT = Math.round(img.cols/2-cx)
    const Y_SHIFT = Math.round(img.rows/2-cy)

    let M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT])
    newSize = new cv.Size(img.rows, img.cols)
    cv.warpAffine(img, img, M, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, b) // shifting the img

    // Rescalling
    let pixVal = img.data
    // console.log(pixVal)

    pixVal = Float32Array.from(pixVal)
    pixVal = pixVal.map((item)=>item/255.)
    // console.log(`Scalled array: ${pixVal}`)

    // Making ready the img for prediction
    let X = tf.tensor(pixVal)
    X = X.reshape([1,28,28,1]) // we should rshape it and ready for cnn
    // console.log(X.shape)

    // Making prediction
    const pred = model.predict(X)
    const res = tf.argMax(pred, axis=-1)
    res.print()
    // console.log(tf.memory())

    const output = res.dataSync() // we store the values of tensor on output and return it


    // // // for testing purpose // // //
    // const outputCanvas = document.createElement("CANVAS")
    // cv.imshow(outputCanvas, img)
    // document.body.appendChild(outputCanvas)


    // Cleanup (at the end we cleanup as recommended in documentation)
    img.delete()
    contours.delete()
    cnt.delete()
    hierarchy.delete()
    M.delete()
    X.dispose()
    pred.dispose()
    res.dispose()


    return output
}

function Check(){
   const pred = predictImage()
   document.getElementById('num').innerHTML = predictImage()

}

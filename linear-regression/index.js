
let data = [];
let m = 0;
let b = 0;
const learningRate = 0.06;
const epochs = 500;

function setup() {
    // The setup() function is called once 
    createCanvas(600, 600);
}

function gradientDescent(mNow, bNow) {
    let mGrad = 0;
    let bGrad = 0;
    const n = data.length;

    for (let p of data) {
        const [x, y] = p;
        mGrad += -(2 / n) * x * (y - (mNow * x + bNow));
        bGrad += -(2 / n) * (y - (mNow * x + bNow));
    }

    const newM = mNow - mGrad * learningRate;
    const newB = bNow - bGrad * learningRate;
    return [newM, newB];
}

function drawLine(m, b) {
    const x1 = 0;
    const y1 = m * x1 + b;
    const x2 = 1;
    const y2 = m * x2 + b;

    const mappedX1 = map(x1, 0, 1, 0, width);
    const mappedY1 = map(y1, 0, 1, height, 0);
    const mappedX2 = map(x2, 0, 1, 0, width);
    const mappedY2 = map(y2, 0, 1, height, 0);

    stroke(255);
    strokeWeight(2);
    line(mappedX1, mappedY1, mappedX2, mappedY2);
}

function mousePressed() {
    // The mousePressed() function is called automatically by p5.js
    // whenever the mouse is pressed
    const x = map(mouseX, 0, width, 0, 1);
    const y = map(mouseY, 0, height, 1, 0);
    const point = [x, y];
    data.push(point);
}

function draw() {
    // The draw() function is called repeatedly by p5.js
    background(51);

    for (let i = 0; i < data.length; i++) {
        const x = map(data[i][0], 0, 1, 0, width);
        const y = map(data[i][1], 0, 1, height, 0);
        fill(0, 255, 0);
        stroke(0, 255, 0);
        ellipse(x, y, 4, 4);
    }

    if (data.length > 1) {
        for (let i = 0; i < epochs; i++) {
            [m, b] = gradientDescent(m, b);

        }
        drawLine(m, b);
    }
}



// Hey everyone, our database expert ​@Jayesh Dehankar​​🌠​ is showing off his SQL skills and querying us to join him for a party at Golconda's for his 1st work anniversary.  ​
// ​Lets ```JOIN``` some ```TABLES``` together and create memories ​​. Who’s up for it? 👏
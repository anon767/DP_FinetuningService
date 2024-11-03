let model;
let vocab = {};  // Store the word-index mapping here
const embedding_dim = 100;

// Load the custom Word2Vec model from TensorFlow.js format
async function loadModel() {
    model = await tf.loadGraphModel('/static/model.json');
    console.log('Model Inputs:', model.inputs);
    console.log('Model Outputs:', model.outputs);
    console.log('Custom Word2Vec model loaded');
}


// Load the vocabulary index
async function loadVocab() {
    const response = await fetch('/static/word_index.json');
    vocab = await response.json();
    console.log('Vocabulary loaded');
}

// Call both model and vocab loading functions
loadModel();
loadVocab();


document.getElementById('uploadButton').addEventListener('click', async function () {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const selectedModel = document.getElementById('model').value;

    if (!file) {
        alert('Please select a file before uploading!');
        return;
    }

    // Read file content
    const reader = new FileReader();
    reader.onload = async function (event) {
        const content = event.target.result;
        const perturbedVectors = await perturbText(content);

        // Create a FormData object and append the perturbed vectors
        const formData = new FormData();
        formData.append('model', selectedModel);
        formData.append('perturbedVectors', JSON.stringify(perturbedVectors));  // Send perturbed vectors

        // Upload perturbed vectors to the server
        fetch('/finetune', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => console.log('Upload success:', data))
        .catch(error => console.error('Upload error:', error));
    };

    reader.readAsText(file);
});

// Generate word embeddings using the custom Word2Vec model and perturb them
async function perturbText(text) {
    const words = text.split(/\s+/);  // Tokenize by spaces
    const perturbedVectors = [];

    for (const word of words) {
        const wordIndex = await getWordEmbedding(word);  // Get word embedding
        const perturbedVector = perturbVector(wordIndex);  // Perturb the vector
        perturbedVectors.push( perturbedVector);
    }

    return perturbedVectors;
}

function clean(word) {
    return word.toLowerCase().replace(",","").replace(".","").trim();
}

async function getWordEmbedding(word) {
    let cleaned_word = clean(word);
    const wordIndex = vocab[cleaned_word] || vocab['UNK'];  // Use 'UNK' if the word is not in vocab
    const inputTensor =  tf.tensor2d([[wordIndex]], [1, 1], 'int32').toFloat();  // Shape it as [1, 1]
    const embedding = await model.executeAsync({ "keras_tensor": inputTensor });
    return embedding.arraySync()[0];  // Convert tensor to array and retrieve the embedding
}

// Perturb word vectors with Laplacian noise
/*
function perturbVector(vector) {
    const noiseScale = 0.02;
    return vector.map(v => v + laplacianNoise(noiseScale));
}
*/

function perturbVector(vector) {
    const stdDev = 0.1;
    return vector.map(v => v + gaussianNoise(stdDev));
}

// Generate Gaussian noise
function gaussianNoise(stdDev) {
    // Using Box-Muller transform to generate Gaussian noise
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * stdDev;
}


// Generate Laplacian noise
function laplacianNoise(scale) {
    const u = Math.random() - 0.5;
    return scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}
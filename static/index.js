let model;
let vocab = {};  // Store the word-index mapping here

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

// Get the embedding for a word using the loaded model
async function getWordEmbedding(word) {
    const wordIndex = vocab[word] || vocab['UNK'];  // Use 'UNK' if the word is not in vocab
    const inputTensor = tf.tensor([[wordIndex]]);  // Ensure dtype is int32
    const embedding = await model.executeAsync({ ["keras_tensor"]: inputTensor });
    return embedding.arraySync()[0];  // Convert tensor to array
}

// Perturb word vectors with Laplacian noise
function perturbVector(vector) {
    const noiseScale = 10;
    return vector.map(v => v + laplacianNoise(noiseScale));
}

// Generate Laplacian noise
function laplacianNoise(scale) {
    const u = Math.random() - 0.5;
    return scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}
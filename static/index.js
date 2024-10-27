let model;

// Load the custom Word2Vec model from TensorFlow.js format
async function loadModel() {
    model = await tf.loadGraphModel('/static/model.json');
    console.log('Custom Word2Vec model loaded');
}

loadModel();  // Load the Word2Vec model when the page loads

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
        perturbedVectors.push({ word, vector: perturbedVector });
    }

    return perturbedVectors;
}

// Get the embedding for a word using the loaded model
async function getWordEmbedding(word) {
    const inputTensor = tf.tensor([[word]]);  // Prepare input tensor (you'll need to encode words to indices if not already done)
    const embedding = await model.executeAsync(inputTensor);  // Get the embedding
    return embedding.arraySync()[0];  // Convert tensor to array
}

// Perturb word vectors with Laplacian noise
function perturbVector(vector) {
    const noiseScale = 0.1;
    return vector.map(v => v + laplacianNoise(noiseScale));
}

// Generate Laplacian noise
function laplacianNoise(scale) {
    const u = Math.random() - 0.5;
    return scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}
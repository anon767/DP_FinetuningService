import json

import gensim
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import gensim.downloader as api
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB limit
app.secret_key = 'your_secret_key'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Hugging Face models available for selection
available_models = [
    'gpt2',  # Example models; replace with your Hugging Face models
    'microsoft/DialoGPT-medium',
    'facebook/blenderbot-400M-distill'
]

# Load the custom TensorFlow Word2Vec model
word2vec_model = tf.keras.models.load_model('word2vec_model')

# Load the word mappings
with open('word_index.json', 'r') as f:
    word_index = json.load(f)

with open('index_word.json', 'r') as f:
    index_word = json.load(f)


@app.route('/finetune', methods=['POST'])
def finetune():
    # Get the perturbed vectors from the request (JSON formatted)
    perturbed_vectors = request.form.get('perturbedVectors')
    perturbed_vectors = json.loads(perturbed_vectors)  # Safely parse JSON string to list of dicts

    nearest_words = []

    for item in perturbed_vectors:
        perturbed_vector = np.array(item['vector'])  # Convert the vector to a numpy array

        # Use the TensorFlow model to get the nearest word for each perturbed vector
        nearest_word = find_nearest_word(perturbed_vector)

        nearest_words.append(nearest_word)

    # Combine the nearest words back into a sentence or text
    result_text = ' '.join(nearest_words)

    return jsonify({"perturbed_text": result_text})


def find_nearest_word(perturbed_vector):
    """
    Find the nearest word for a given perturbed vector using the Word2Vec TensorFlow model.
    """
    # Get all word embeddings from the model (assumed that you have the embeddings as part of your model)
    all_word_embeddings = word2vec_model.get_layer('embedding').get_weights()[0]

    # Compute cosine similarity between the perturbed vector and all word embeddings
    similarity = tf.keras.losses.cosine_similarity(perturbed_vector, all_word_embeddings)

    # Find the index of the most similar word
    nearest_word_index = tf.argmax(similarity).numpy()

    # Retrieve the word corresponding to the index
    nearest_word = index_word[str(nearest_word_index)]

    return nearest_word


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', models=available_models)


if __name__ == '__main__':
    app.run(debug=True)

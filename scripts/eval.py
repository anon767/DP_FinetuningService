import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Define and register ClipConstraint
@tf.keras.utils.register_keras_serializable()
class ClipConstraint(Constraint):
    def __call__(self, weights):
        return tf.clip_by_value(weights, -1, 1)


# Load the word mappings
with open('../static/word_index.json', 'r') as f:
    word_index = json.load(f)

with open('../static/index_word.json', 'r') as f:
    index_word = json.load(f)

# Model parameters (adjust these based on your original setup)
vocab_size = len(index_word)+1  # replace with your actual vocab size
embedding_dim = 100  # replace with your actual embedding dimension

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.Reshape((embedding_dim,))
])
model.build(input_shape=(None, 1))

# Load weights from the checkpoint
checkpoint_filepath = 'checkpoint.model.keras'
model.load_weights(checkpoint_filepath)
eval_words = ['harry', 'potter', 'magic', 'wizard', 'wand', 'spell']
eval_indices = [word_index.get(word, word_index['UNK']) for word in eval_words]

def evaluate(model, eval_indices, index_word, top_k=8):
    eval_embeddings = model.predict(np.array(eval_indices))
    all_embeddings = model.layers[0].get_weights()[0]
    for i, eval_embedding in enumerate(eval_embeddings):
        similarities = np.array([cosine_similarity(eval_embedding, embedding) for embedding in all_embeddings])
        nearest = (-similarities).argsort()[:top_k + 1]
        nearest_words = [index_word[str(idx)] for idx in nearest]
        print(f"{all_embeddings[nearest[0]]}")
        print(f'"{eval_words[i]}" nearest neighbors: {", ".join(nearest_words)}')

evaluate(model, eval_indices, index_word, 10)
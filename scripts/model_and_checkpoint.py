import json

import tensorflow as tf
from tensorflow.keras.constraints import Constraint


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
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,  dtype=tf.float32)),
    tf.keras.layers.Reshape((embedding_dim,), name="embedding_output")
])
model.build(input_shape=(None, 1))

# Load weights from the checkpoint
checkpoint_filepath = 'checkpoint.model.keras'
model.load_weights(checkpoint_filepath)

# Save the complete model as an .h5 file
model.export('word2vec_model_with_checkpoint')

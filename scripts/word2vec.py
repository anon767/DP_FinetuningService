import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.constraints import Constraint

import gensim.downloader as api
import json


# Function to calculate cosine similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


# Function to evaluate nearest neighbors
def evaluate(model, eval_indices, index_word, top_k=8):
    eval_embeddings = model.predict(np.array(eval_indices))
    all_embeddings = model.layers[0].get_weights()[0]
    for i, eval_embedding in enumerate(eval_embeddings):
        similarities = np.array([cosine_similarity(eval_embedding, embedding) for embedding in all_embeddings])
        nearest = (-similarities).argsort()[1:top_k + 1]
        nearest_words = [index_word[idx] for idx in nearest]
        print(f'"{eval_words[i]}" nearest neighbors: {", ".join(nearest_words)}')


# Load the Text8 corpus
text8_corpus = api.load("text8")
sentences = list(text8_corpus)

# Tokenize the words (with max vocabulary size of 50k)
max_vocab_size = 200000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_counts = tokenizer.word_counts  # Get word counts
filtered_word_index = {word: idx for word, idx in tokenizer.word_index.items() if word_counts[word] >= 3}
filtered_word_index["UNK"] = 0

sentences = [[word if word in filtered_word_index else 'UNK' for word in sentence] for sentence in sentences]

# Limit the vocabulary to max_vocab_size
filtered_word_index = dict(list(filtered_word_index.items())[:max_vocab_size])

# Reverse the mapping (index to word)
index_word = {index: word for word, index in filtered_word_index.items()}

# Save the filtered word_index and index_word mappings
with open('word_index.json', 'w') as f:
    json.dump(filtered_word_index, f)

with open('index_word.json', 'w') as f:
    json.dump(index_word, f)

vocab_size = len(filtered_word_index) + 1  # Adjusted vocabulary size based on the filtered words

# Convert sentences to sequences of word indices (only keeping filtered words)
tokenizer.word_index = filtered_word_index  # Use filtered word index for tokenization
sequences = tokenizer.texts_to_sequences(sentences)
# Padding sequences to ensure equal length
max_length = 64
sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Prepare training data (target context word and surrounding words)
window_size = 5
X_train = []
y_train = []

for seq in sequences:
    for i, target_word in enumerate(seq):
        if target_word == 0:  # Skip padding and UNK tokens as target words
            continue
        start = max(0, i - window_size)
        end = min(len(seq), i + window_size + 1)
        context_words = [seq[j] for j in range(start, end) if j != i and seq[j] != 0]  # Skip UNK tokens in context
        for context_word in context_words:
            X_train.append(target_word)
            y_train.append(context_word)

X_train = np.array(X_train)
y_train = np.array(y_train)
# Define Noise Contrastive Estimation loss function
embedding_dim = 50
nce_weights = tf.Variable(tf.random.normal([vocab_size, embedding_dim]))
nce_biases = tf.Variable(tf.zeros([vocab_size]))


# Define Noise Contrastive Estimation loss function
def nce_loss(y_true, y_pred):
    loss = tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=tf.reshape(y_true, [-1, 1]),
        inputs=y_pred,
        num_sampled=5,
        num_classes=vocab_size
    )
    return tf.reduce_mean(loss)


class ClipConstraint(Constraint):
    def __call__(self, weights):
        return tf.clip_by_value(weights, -1, 1)

# Define Word2Vec-like model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_constraint=ClipConstraint()),
    tf.keras.layers.Reshape((embedding_dim,))
])

# Define SGD optimizer with momentum
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.00001
)

model.compile(optimizer=optimizer, loss=nce_loss)

# Train the model
checkpoint_filepath = 'checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_best_only=True)

model.fit(X_train, y_train, epochs=100, verbose=1, batch_size=128, callbacks=[model_checkpoint_callback])
model.save('word2vec_model_full.h5')

eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']
eval_indices = [filtered_word_index.get(word, filtered_word_index['UNK']) for word in eval_words]

model.load_weights(checkpoint_filepath)

evaluate(model, eval_indices, index_word)

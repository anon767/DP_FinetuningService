import json

import gensim
from flask import Flask, render_template, request, redirect, url_for, jsonify
from keras.src.constraints import Constraint
from werkzeug.utils import secure_filename
import os
import numpy as np
import gensim.downloader as api
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

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
# Define and register ClipConstraint
@tf.keras.utils.register_keras_serializable()
class ClipConstraint(Constraint):
    def __call__(self, weights):
        return tf.clip_by_value(weights, -1, 1)


# Load the word mappings
with open('static/word_index.json', 'r') as f:
    word_index = json.load(f)

with open('static/index_word.json', 'r') as f:
    index_word = json.load(f)

# Model parameters (adjust these based on your original setup)
vocab_size = len(index_word) + 1  # replace with your actual vocab size
embedding_dim = 100  # replace with your actual embedding dimension

# Define the model architecture
word2vec_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.Reshape((embedding_dim,))
])
word2vec_model.build(input_shape=(None, 1))

# Load weights from the checkpoint
checkpoint_filepath = 'scripts/checkpoint.model.keras'
word2vec_model.load_weights(checkpoint_filepath)


def cosine_similarity(a, b):
    dot_product = np.dot(b, a)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    return dot_product / (norm_a * norm_b)

def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch]),
        "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in batch]),
        "labels": torch.stack([torch.tensor(item["labels"]) for item in batch]),
    }

@app.route('/finetune', methods=['POST'])
def finetune():
    # Get the perturbed vectors from the request (JSON formatted)
    perturbed_vectors = request.form.get('perturbedVectors')
    perturbed_vectors = json.loads(perturbed_vectors)  # Safely parse JSON string to list of dicts
    epochs = 2
    noise_multiplier = 0.1
    max_grad_norm = 1.0
    batch_size = 2

    nearest_words = []

    for item in perturbed_vectors:
        perturbed_vector = np.array(item)  # Convert the vector to a numpy array

        # Use the TensorFlow model to get the nearest word for each perturbed vector
        nearest_word = find_nearest_word(perturbed_vector)

        nearest_words.append(nearest_word)

    # Combine the nearest words back into a sentence or text
    perturbed_text = ' '.join(nearest_words)

    selected_model = request.form.get('model')

    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(selected_model)
    # Configure LoRA
    lora_compatible_modules = []

    # Check each module in the model
    for name, module in model.named_modules():
        # Check if the module is a type supported by LoRA
        if isinstance(module, (nn.Linear, nn.Conv2d)) or \
                (hasattr(nn, "Conv1D") and isinstance(module, nn.Conv1d)):
            lora_compatible_modules.append(name)

    print("Compatible modules for LoRA:", lora_compatible_modules)
    lora_config = LoraConfig(
        r=16,  # Low-rank dimension
        lora_alpha=16,  # Alpha scaling factor
        target_modules=lora_compatible_modules,  # List of compatible modules
        task_type=TaskType.CAUSAL_LM,
    )
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    # Tokenize the perturbed text
    inputs = tokenizer(perturbed_text, return_tensors="pt", truncation=True, padding=True)
    inputs["labels"] = inputs["input_ids"].clone()  # Set labels as input_ids for next-token prediction

    # Prepare the dataset and DataLoader
    dataset = Dataset.from_dict({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": inputs["labels"]
    })
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    # Set up Opacus PrivacyEngine for DPSGD
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(module=model,
                                                               optimizer=optimizer,
                                                               data_loader=dataloader,
                                                               noise_multiplier=noise_multiplier,
                                                               max_grad_norm=max_grad_norm
                                                               )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")

    # Save the fine-tuned model
    model._module.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    return jsonify({"perturbed_text": perturbed_text})


def find_nearest_word(perturbed_vector):
    """
    Find the nearest word for a given perturbed vector using the Word2Vec TensorFlow model.
    """
    # Get all word embeddings from the model (assumed that you have the embeddings as part of your model)
    all_word_embeddings = word2vec_model.get_layer('embedding').get_weights()[0]  # Shape: (vocab_size, embedding_dim)

    # Compute cosine similarity between the perturbed vector and all word embeddings
    similarity = cosine_similarity(perturbed_vector, all_word_embeddings)

    # Find the index of the nearest word
    nearest_index = np.argmax(similarity)
    nearest_word = index_word.get(str(nearest_index), "UNK")

    return nearest_word


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', models=available_models)


if __name__ == '__main__':
    app.run(debug=True)

# DP Finetuning Service

This repositorty contains the code to generate a finetuning service which yields data anonymization during transfer and 
outputs a privacy preserving finetuned (text-based) Model through differentially private mechanisms.

## Datatransfer

Usually, the data scientist requires the data of the data owner (Do), thus requiring the potentially sensitive data to leave the DO's premises.
Using a Local Differentially Private (LDP) Mechanism we can circumvent the problem of data leakage.

For finetuning most LLMs we do not require much data thus we can simply assume a max of 20MB text file for the start.
We need this data to be moved to our premises without us being aware of the content but still be able to finetune a model. 
Typical alternatives include:
- Homomorphic encryption (Slow)
- Trusted Computing Environments (Allow Sidechannel attacks)

Instead, we apply a Word2Vec embedding model and perturb the data on the Client (DO) premises. This can be done by leveraging tensorflow.js on the clients browser.
That means, the client itself perturbs the data such that we do not receive the original ones.
At serverside, we reconstruct the perturbed one losing all sensitive information.

### LDP

We train a W2V model on _text8_ dataset using a vocab of 100k (only inlcude words that occur at least 3 times).
Furthermore the parameters are:
- AdamW with weight Decay for fast second-order learning but interpretable embeddings
- 50 dimensions (Needs to be fast for Inference on Browser)
- Embedding clipped to [-1, 1] for calculating Epsilon
- Windows size of 5 and 8 negative samples for Noise Contrastive Loss

We choose to use a laplace mechanism with an Epsilon of 0.1 which result in a per-sample Epsilon of 20.

Building the model:
```commandline
pip3 install -r requirements.txt
python3 scripts/word2vec.py
```

Transform to GraphModel for JS:
```commandline
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model scripts/word2vec_model_full.h5 statc/
```

## Finetuning

- Low Rank Adaption: We freeze all layers and add low-rank adapters
- We use DPSGD during training
- We use AWS Lambda functions for on-the-fly Async finetuning
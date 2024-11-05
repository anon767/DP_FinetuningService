# DP Finetuning Service

This repository contains the code to create a fine-tuning service that ensures data anonymization during transfer and produces a privacy-preserving, fine-tuned text model using differential privacy mechanisms.

## Datatransfer

Typically, data scientists need access to a data owner's (DO) data, which often requires transferring potentially sensitive information outside the DO's premises. By applying a Local Differentially Private (LDP) mechanism, we can prevent data leakage during transfer.

For most large language models (LLMs), fine-tuning requires only a small amount of data. We start with a 20MB text file limit for uploaded data, which is transferred without revealing the original content. Common alternatives, such as homomorphic encryption (which is slow) or trusted computing environments (which can be vulnerable to side-channel attacks), present limitations.

Instead, we apply a Word2Vec (W2V) embedding model and perturb the data directly on the client (DO) side using TensorFlow.js in the browser. This approach ensures that only perturbed data is received by the server, retaining all sensitive information on the client side.



### LDP

We train a W2V model on _text8_ dataset using a vocab of 100k (only inlcude words that occur at least 3 times).
Furthermore the parameters are:
- AdamW with weight Decay for fast second-order learning but interpretable embeddings
- 50 dimensions (Needs to be fast for Inference on Browser)
- Embedding clipped to [-1, 1] for calculating Epsilon
- Windows size of 5 and 8 negative samples for Noise Contrastive Loss

We apply either a Laplacian or Gaussian noise mechanism; initial experiments show Gaussian noise performs better. With an embedding range of two (global sensitivity) and a noise standard deviation of 0.1, our LDP epsilon value is approximately 96.9 (assuming delta = 10^-5).



Example of perturbation
Input-text:
```commandline
Harry Potter was a highly unusual boy in many ways.
He was always the last to be picked for any sports team.
He was born a wizard, and his life changed forever when he received a letter from Hogwarts.
Harry couldn't believe that he was going to a school for wizards.
"You're a wizard, Harry," said Hagrid, as he handed him the letter.
The scar on his forehead, shaped like a lightning bolt, marked him as someone special.
Voldemort, the dark wizard, had tried to kill Harry when he was just a baby.
Ron Weasley and Hermione Granger quickly became Harry's best friends.
The trio went on many adventures, from discovering secret rooms to battling dark forces.
At Hogwarts, Harry learned the importance of friendship, courage, and loyalty.
```
Perturbed Output-Text
```commandline
harry potter c by highly unusual boy a many ways
there on always the last of be picked upon any sports team
even city born no wizard work was life changed forever as modern received an letter up censoring
harry handloading believe head he then going general york school an wizards guillotin strong wizard
rolex had outscored as a handed him rather arts best precocial model his bachs shaped like an lightning reproduction
marked him religious someone special naslund non dark wizard had tried to kill harry when up c just a baby ron
simplify information lochaber arresting quickly became mallard best friends the iroquoian went on standard adventures
than refereeing secret rooms has pi dark both for gaspard harry learned main importance series friendship courage and loyalty
```

### CDP

To protect sensitive information in models from leaking during inference, we use a Gaussian mechanism to perturb gradients during fine-tuning. By clipping gradients at 1 and adding noise with a standard deviation of 0.1, we achieve an epsilon of approximately 48.45 (assuming delta = 10^-5).

## Building the model:
Train the word2vec model for transfer sensitive data:
```commandline
pip3 install -r requirements.txt
cd scripts
python3 word2vec.py
```

Store the model and save it as JS digestible Tensorflow artifacts (graph-model)
```commandline
cd scripts
chmod +x build.sh
./build.sh
```

## Finetuning

We are using Low-Rank adaption (LoRA) for fast fine-tuning and to make sure only our newly added weights are perturbed.
This preserves the initial performance of the model.

- TODO: We use AWS Lambda functions for on-the-fly Async finetuning
- TODO: Instead of W2V, use pre-trained BPE tokenizer and fine-tune on LDP
### Run

You can simple start the server with:
```commandline
pip3 -r requirements.txt
python3 main.py
```

Choose a huggingface model to fine-tune and upload an example corpus.
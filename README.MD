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

We choose to use either a Laplacian or Gaussian mechanism. In few experiments Gaussian seem to work better.
Since the global sensitivity is two (given the embedding range) and the noise standrad deviation is 0.1 we have a LDP epsilon of: 96.9.
(Assuming a delta of 10^-5)

Example
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
Output-Text
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

In case the model needs to be distributed and to save the potential sensitive training data from leaking,
we use Gaussian mechanism to perturb the gradients during fine-tuning.
We use gradient clipping of 1 and noise of 0.1 arriving at an Epsilon of 48.45 (assuming the same Delta).

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

- TODO: Low Rank Adaption: We freeze all layers and add low-rank adapters
- TODO: We use AWS Lambda functions for on-the-fly Async finetuning
- TODO: Instead of W2V, use pre-trained BPE tokenizer and fine-tune on LDP
### Run

You can simple start the server with:
```commandline
pip3 -r requirements.txt
python3 main.py
```

Choose a huggingface model to fine-tune and upload an example corpus.
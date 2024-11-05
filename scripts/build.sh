#!/bin/zsh
python3 model_and_checkpoint.py
tensorflowjs_converter --input_format=tf_saved_model word2vec_model_with_checkpoint ../static/

#!/bin/bash
pip install keras-nlp keras-core tensorflow-text --no-deps
pip install nltk
pip install bertviz
pip install wandb
pip install datasets
mkdir saved_models
mkdir model_attentions
# Check if nltk_data folder exists
if [ ! -d "nltk_data" ]; then
    # If not, run the Python command to download nltk data
    python3 -c "import nltk; nltk.download('averaged_perceptron_tagger')"    
fi

# Check if aclImdb folder exists
if [ ! -d "aclImdb" ]; then
    # If not, download the dataset and extract it
    curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar -xf aclImdb_v1.tar.gz
fi

# Language Classification

This repository contains a number of ways to classify the language of an input word, including MLP, CNN, logistic regression, and random forest. 
Currently, the classifiers distinguish between English and Spanish.

Each character in a word is converted into a one-hot character vector of length 32. 
All special characters (the five accented vowels and ñ) are placed at the end of the character. 
The input data is the concatenation of all these character vectors, and if the word is shorter than `max_word_length`, the data is padded with zeros. 

There are a number of pre-trained models included in the `models/` directory, although many of them use a different network structure. 
The exact structure of each of model is provided below.

`models/model_full*.ckpt`: ReLU or sigmoid(gamma * x) activations
* Convolution layer with 32 3x3 filters
* Maxpool
* Convolution layer with 32 3x3 filters
* Fully connected layer with 256 nodes
* Fully connected layer with 128 nodes
* Output layer with 2 nodes

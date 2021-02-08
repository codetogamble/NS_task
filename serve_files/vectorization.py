import os
import numpy as np


GLOVE_DIR = "./embeddings/"
GLOVE_FILE = 'glove.6B.50d.txt'
EMBEDDING_DIM = 50

def getEmbeddingWeightsGlove(word_index):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR,GLOVE_FILE))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

from dataset import readDataset,readTestDataset
from textpreprocessing import preprocessDF, getBalancedData
from utils import initTokenizer,loadTokenizer,dataPrep
from mymodels import getModelWithType
import tensorflow as tf
import os

TEST_ON_COMP = False
## assign False if validation set should be extracted from training set itself
DATA_SPLIT = 0.2 
## Not applicable if TEST_ON_COMP is True
BINARY_CLASSIFICATION = False
## If True classes will be seperated for unrelated and related only
OVERSAMPLING_STANCEWISE = True
## Toggle for switching off or on oversampling for data according stance distribution
MODEL_NAME = "model_1"
NEW_TOKENIZER = True
## Toggle for loading previously created tokenizer with model name
MODEL_TYPE = "TRANSFORMER"
## TRANSFORMER or CNN

MAX_LENGTH_ARTICLE = 1200
MAX_LENGTH_HEADLINE = 40
TRAIN_EMBED = False
LOAD_PREV = False

print("reading datasets ... ")
main_set = readTestDataset("./dataset/")
main_set = preprocessDF(main_set,binaryclass=BINARY_CLASSIFICATION)

if(OVERSAMPLING_STANCEWISE):
    main_set = getBalancedData(main_set,binaryclass=BINARY_CLASSIFICATION)

print("MAIN SET balanced.")
tokenizer = loadTokenizer(MODEL_NAME)

print("DATA PREP for TEST set")
testap,testhp,test_labels = dataPrep(main_set,tokenizer,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE)
print("Padded Inputs with Labels TEST READY.")

checkpoint_path = "./checkpoints/"+MODEL_TYPE+"_"+MODEL_NAME+"/weights.hdf5"
model_ = getModelWithType(MODEL_TYPE,BINARY_CLASSIFICATION,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE,TRAIN_EMBED,tokenizer)
model_.load_weights(checkpoint_path)
print(model_.summary())

result = model_.evaluate([testap,testhp],test_labels,batch_size=100)
print(result)





from dataset import readDataset,readTestDataset
from textpreprocessing import preprocessDF, getBalancedData
from utils import initTokenizer,loadTokenizer,dataPrep
from mymodels import getModelWithType
import tensorflow as tf
import os

### THIS FILE IS USED TO TRAIN MODELS WITH SPECIFIED CONFIGURATION. KEEP IN MIND TO USE THE SAME CONF FOR measure.py.


TEST_ON_COMP = True
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

print("reading datasets ... ")
main_set = readDataset("./dataset/")
main_set = preprocessDF(main_set,binaryclass=BINARY_CLASSIFICATION)


if(TEST_ON_COMP):
    train_set = main_set
    test_set = readTestDataset("./dataset/")
    test_set = preprocessDF(test_set,binaryclass=BINARY_CLASSIFICATION)
else:
    totalno = len(main_set)
    split_index = totalno - int(DATA_SPLIT*totalno)

    randomsamp = main_set.sample(frac=1)
    train_set = randomsamp[:split_index]
    test_set = randomsamp[split_index:]

if(OVERSAMPLING_STANCEWISE):
    train_set = getBalancedData(train_set,binaryclass=BINARY_CLASSIFICATION)
    test_set = getBalancedData(test_set,binaryclass=BINARY_CLASSIFICATION)

print("train and test set created with shape",train_set.shape,test_set.shape)


if(NEW_TOKENIZER):
    print("Initializing New Tokenizer")
    tokenizer = initTokenizer(train_set,MODEL_NAME)
else:
    print("Loading preset Tokenizer")
    tokenizer = loadTokenizer(MODEL_NAME)


print("DATA PREP for TRAIN set")
trainap,trainhp,train_labels = dataPrep(train_set,tokenizer,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE)
print("Padded Inputs with Labels TRAIN READY.")
print("DATA PREP for TEST set")
testap,testhp,test_labels = dataPrep(test_set,tokenizer,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE)
print("Padded Inputs with Labels TEST READY.")

model_ = getModelWithType(MODEL_TYPE,BINARY_CLASSIFICATION,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE,TRAIN_EMBED,tokenizer)
print(model_.summary())
checkpoint_path = "./checkpoints/"+MODEL_TYPE+"_"+MODEL_NAME+"/weights.hdf5"

if(not os.path.exists("./checkpoints/"+MODEL_TYPE+"_"+MODEL_NAME)):
    os.mkdir("./checkpoints/"+MODEL_TYPE+"_"+MODEL_NAME)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,monitor='acc',save_best_only=False)

history = model_.fit(x=[trainap,trainhp],y=train_labels,epochs=4,batch_size=100,callbacks=[model_checkpoint_callback],validation_data=([testap,testhp],test_labels))

model_.save_weights(checkpoint_path)
# model_.save("./saved/"+MODEL_TYPE+"_"+MODEL_NAME)


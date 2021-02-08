# NS_task
task for attempting fnc stance detection challenge

setup : 

run in terminal - pip install -r requirements.txt
NOTE: tensorflow version is legacy version of 1.14.0 but the code has also been tested on version 2.3.1. 
Thus using updated tensorflow should not pose any issues.

download glove word embeddding from link : http://nlp.stanford.edu/data/glove.6B.zip
unzip the downloaded zip and move "glove.6B.50d.txt" to "embeddings" folder. 
(File size is greater than git's permissible limit of 25MB, hence not pushed in this repo).


# Configuration for model in train.py file.

TEST_ON_COMP = True (This trains model on all of the data from train.csv while testing on competetion.csv, if False this will split the training data and use it as validation set)

DATA_SPLIT = 0.2 (Data split for the case where TEST_ON_COMP is False)

BINARY_CLASSIFICATION = False (Toggle for building models for only classifying from unrelated and related body headline)

OVERSAMPLING_STANCEWISE = True (Used for balancing the data according to stance distribution)

MODEL_NAME = "model_1"
MODEL_TYPE = "TRANSFORMER" (Both MODEL_NAME and TYPE are used in saving model weights but only MODEL_NAME is used to save tokenizer)
current types = ["TRANSFORMER","CNN"]

NEW_TOKENIZER = True (Toggle for loading tokenizer based on MODEL_NAME)

MAX_LENGTH_ARTICLE = 1200 (these values is chosen becuase majority of data lies below this value, increasing this value would increasing computational cost greatly)
MAX_LENGTH_HEADLINE = 40 
TRAIN_EMBED = False (Toggle if you want to make word embedding layer trainables)

Running "python train.py" command will launch a training session of model with specified configuration. Tokenizer for each run is saved with model name in folder named "tokenizers_confs" for future use.

"python measure.py" command will load and evaluate the model on competetion_test.csv. Make sure to use identical coinfiguration for the measure.py file which were used in train.py while training the model.

# MODEL TYPES:
Both architectures takes mutliple inputs (which is why functional api is used instead of sequential api of keras for ease). First input is article body and second inpurt is statement.

"CNN" : This was my simplest attempt to build a mechanism which would take certain sequences of words into account. For this architecture succesive layers of Conv1D is followed by AveragePooling and Batchnormalization is used after every second layer of Conv1D. Similar architecture was used as one of the models by Cisco Talos team in fnc challenge . https://github.com/Cisco-Talos/fnc-1/tree/master/deep_learning_model

"TRANSFORMER" : This architecure is the encoder part of architecture mentioned in paper "Attention is all you need" https://arxiv.org/abs/1706.03762 and Tensorflow tutorial https://www.tensorflow.org/tutorials/text/transformer. For summary this model contains positional enncoding addition into input and encoder layer which contains multihead attention layer followed by feed forward layers. After encoding the same above model of CNN is used to reduce the dimensions while also maintaining low number of trainable paramters.

No hyperparamter tuning has been done yet for both models. Generally speaking these models are just the implementation of the concepts thus to run quickly and efficiently I have decided to keep the number of trainable paramters to be low.



A version of model of type "TRANSFORMER" is hosted on GCP. To test results change body and headline accordingly and use the request below.

curl --location --request POST 'http://35.233.224.30:5000/query' --header 'Content-Type: application/json' --data-raw '{ "body":"Reddit got Hedgefunds for good, community power is strong. Spite of human nature can move large scale stock markets.", "headline":"Reddit can be strong." }'

sample result : {"result":{"agree":"0.54459983","disagree":"0.001165206","discuss":"0.06469633","unrelated":"0.38953862"}}

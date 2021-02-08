from flask import Flask
from flask import request, send_file
from flask import jsonify
import json
import tensorflow as tf
from textpreprocessing import getModelInput
from utils import loadTokenizer,dataPrepForString
from mymodels import getModelWithType
from tensorflow.python.keras.backend import set_session


config = tf.ConfigProto(device_count = {'GPU': 0})

sess = tf.Session(config=config)
set_session(sess)

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


tokenizer = loadTokenizer(MODEL_NAME)
checkpoint_path = "./checkpoints/"+MODEL_TYPE+"_"+MODEL_NAME+"/weights.hdf5"
model_ = getModelWithType(MODEL_TYPE,BINARY_CLASSIFICATION,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE,TRAIN_EMBED,tokenizer)
model_.load_weights(checkpoint_path)
print(model_.summary())
graph = tf.get_default_graph()


sample_body = "Reddit got Hedgefunds for good, community power is strong. Spite of human nature can move large scale stock markets."
sample_headline = "Reddit can be strong."

app = Flask(__name__)

@app.route('/')
def hello_world():
    global graph
    global model_
    global sess
    out1 = getModelInput(sample_body)
    out2 = getModelInput(sample_headline)
    input1 = dataPrepForString(out1,tokenizer,MAX_LENGTH_ARTICLE)
    input2 = dataPrepForString(out2,tokenizer,MAX_LENGTH_HEADLINE)
    print(input1.shape)
    print(input2.shape)
    with graph.as_default():
        set_session(sess)
        result = model_.predict([input1,input2])
        print(result)
    
    return 'NS task server running!!'

@app.route('/query',methods=["POST"])
def getQuery():
    reqdict = request.json
    resdict = {}
    global graph
    global model_
    global sess
    out1 = getModelInput(reqdict['body'])
    out2 = getModelInput(reqdict['headline'])
    input1 = dataPrepForString(out1,tokenizer,MAX_LENGTH_ARTICLE)
    input2 = dataPrepForString(out2,tokenizer,MAX_LENGTH_HEADLINE)
    print(input1.shape)
    print(input2.shape)
    with graph.as_default():
        set_session(sess)
        result = model_.predict([input1,input2])
        print(result)
        resdict['result'] = {'unrelated':str(result[0][0]),'disagree':str(result[0][1]),'discuss':str(result[0][2]),'agree':str(result[0][3])}
    print(resdict)
    return jsonify(resdict)
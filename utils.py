from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import json

## Utils file is for general preprocessing steps related to models.


def getOneHotLabels(ppdf):
    '''converts int values in stance column from dataframe to one hot encoded array'''
    labels = to_categorical(ppdf["Stance"].values)
    return labels

def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

def initTokenizer(df,modelname,oov_token='<OOV>'):
    '''initializes a New tokenizer for given dataframe and model name'''
    totaltext_body = " ".join(df["articleBody"].unique())
    totaltext_headline = " ".join(df["Headline"].unique())
    totaltext = totaltext_body + " " + totaltext_headline
    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts([totaltext])
    with open("./tokenizers_confs/"+ modelname +"_config.json","w") as fconf:
        fconf.write(tokenizer.to_json())
    return tokenizer


def loadTokenizer(modelname):
    '''loads a previously saved tokenizer from config file'''
    with open("./tokenizers_confs/"+modelname+"_config.json") as fconf:
        jsonstring = fconf.read()
        tokenizer = tokenizer_from_json(jsonstring)
    return tokenizer


def getTokenizedData(ppdf,tokenizer):
    '''converts string statement to list of token values WITHOUT PADDING'''
    articlebody = ppdf["articleBody"].values
    headline = ppdf["Headline"].values
    articlebody_split = [b.split() for b in articlebody]
    headline_split = [b.split() for b in headline]
    artbodytok = tokenizer.texts_to_sequences(articlebody_split)
    headlinetok = tokenizer.texts_to_sequences(headline_split)
    return [artbodytok,headlinetok]


def getPaddedData(tokendata,maxlength):
    '''adds POST padding to the given tokendata'''
    padded = pad_sequences(tokendata,maxlen=maxlength,padding="post",truncating="post")
    return padded

def dataPrep(df,tokenizer,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE):
    '''assembles the operations above for making model compatible inputs'''
    listtok = getTokenizedData(df,tokenizer)
    articlebodytok = listtok[0]
    headlinetok = listtok[1]

    articlepadded = getPaddedData(articlebodytok,MAX_LENGTH_ARTICLE)
    headlinepadded = getPaddedData(headlinetok,MAX_LENGTH_HEADLINE)

    labels = getOneHotLabels(df)
    return articlepadded,headlinepadded,labels

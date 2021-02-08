import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from vectorization import getEmbeddingWeightsGlove
from tensorflow.keras.optimizers import Adam
import numpy as np

def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],np.arange(d_model)[np.newaxis, :],d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.6):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask):

        attn_output, _ = self.mha.call(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


        

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def getModelWithType(MODEL_TYPE,BINARY_CLASSIFICATION,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE,TRAIN_EMBED,tokenizer):
    numclasses = 4
    if(BINARY_CLASSIFICATION):
        numclasses = 2
    
    embdweights = getEmbeddingWeightsGlove(tokenizer.word_index)
    
    Input_body = layers.Input(shape=(MAX_LENGTH_ARTICLE,))
    Input_statement = layers.Input(shape=(MAX_LENGTH_HEADLINE,))
    embed_body = layers.Embedding(embdweights.shape[0],embdweights.shape[1],weights=[embdweights],input_length=MAX_LENGTH_ARTICLE,trainable=TRAIN_EMBED)(Input_body)
    embed_stat = layers.Embedding(embdweights.shape[0],embdweights.shape[1],weights=[embdweights],input_length=MAX_LENGTH_HEADLINE,trainable=TRAIN_EMBED)(Input_statement)

    if(MODEL_TYPE=="CNN"):
        convbody_1 = layers.Conv1D(16,1,activation='relu')(embed_body)
        convbody_1pooled = layers.AveragePooling1D(pool_size=4,strides=4)(convbody_1)
        convbody_2 = layers.Conv1D(32,3,activation='relu')(convbody_1pooled)
        convbody_2 = layers.BatchNormalization()(convbody_2)
        convbody_2pooled = layers.AveragePooling1D(pool_size=4,strides=4)(convbody_2)
        convbody_3 = layers.Conv1D(32,5,activation='relu')(convbody_2pooled)
        convbody_3 = layers.BatchNormalization()(convbody_3)
        convbody_3pooled = layers.AveragePooling1D(pool_size=4,strides=4)(convbody_3)
        convbody_4 = layers.Conv1D(48,5,activation='sigmoid')(convbody_3pooled)
        convbody_4 = layers.BatchNormalization()(convbody_4)

        convstat_1 = layers.Conv1D(16,1,activation='relu')(embed_stat)
        convstat_1 = layers.BatchNormalization()(convstat_1)
        convstat_1pooled = layers.AveragePooling1D(strides=2)(convstat_1)
        convstat_2 = layers.Conv1D(32,3,activation='relu')(convstat_1pooled)
        convstat_2pooled = layers.AveragePooling1D(strides=2)(convstat_2)
        convstat_3 = layers.Conv1D(32,5,activation='relu')(convstat_2pooled)
        convstat_3 = layers.BatchNormalization()(convstat_3)
        convstat_4 = layers.Conv1D(48,5,activation='sigmoid')(convstat_3)
        convstat_4 = layers.BatchNormalization()(convstat_4)

        flatbody = layers.Flatten()(convbody_4)
        flatbody = layers.Dropout(0.8)(flatbody)
        flatstat = layers.Flatten()(convstat_4)
        flatstat = layers.Dropout(0.4)(flatstat)
        mergedlay = layers.Concatenate(axis=1)([flatbody,flatstat])
        outdense1 = layers.Dense(32,activation='relu')(mergedlay)
        outdense2 = layers.Dense(numclasses,activation='softmax')(outdense1)
        opt = Adam(learning_rate=0.0001)
        modelret = Model(inputs=[Input_body,Input_statement],outputs=[outdense2])
        modelret.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["CategoricalAccuracy"])
        return modelret
    
    elif(MODEL_TYPE=="TRANSFORMER"):
        n1,d1 = MAX_LENGTH_ARTICLE, 50
        n2,d2 = MAX_LENGTH_HEADLINE, 50

        pos_encoding_body = positional_encoding(n1, d1)
        pos_encoding_statement = positional_encoding(n2, d2)

        encoding_layer_1 = EncoderLayer(50,2,256)
        encoding_layer_1b = EncoderLayer(50,2,256)

        embed_body *= tf.math.sqrt(tf.cast(50, tf.float32))
        embed_body += pos_encoding_body[:, :MAX_LENGTH_ARTICLE, :]
        embed_stat *= tf.math.sqrt(tf.cast(50, tf.float32))
        embed_stat += pos_encoding_statement[:, :MAX_LENGTH_HEADLINE, :]

        bodymask = create_padding_mask(Input_body)
        statmask = create_padding_mask(Input_statement)
        
        bodyenc = encoding_layer_1(embed_body,mask = bodymask)
        statenc = encoding_layer_1b(embed_stat,mask = statmask)

        convbody_1 = layers.Conv1D(16,1,activation='relu')(bodyenc)
        convbody_1pooled = layers.AveragePooling1D(pool_size=4,strides=4)(convbody_1)
        convbody_2 = layers.Conv1D(32,3,activation='relu')(convbody_1pooled)
        convbody_2 = layers.BatchNormalization()(convbody_2)
        convbody_2pooled = layers.AveragePooling1D(pool_size=4,strides=4)(convbody_2)
        convbody_3 = layers.Conv1D(32,5,activation='relu')(convbody_2pooled)
        convbody_3 = layers.BatchNormalization()(convbody_3)
        convbody_3pooled = layers.AveragePooling1D(pool_size=4,strides=4)(convbody_3)
        convbody_4 = layers.Conv1D(48,5,activation='sigmoid')(convbody_3pooled)
        convbody_4 = layers.BatchNormalization()(convbody_4)

        convstat_1 = layers.Conv1D(16,1,activation='relu')(statenc)
        convstat_1 = layers.BatchNormalization()(convstat_1)
        convstat_1pooled = layers.AveragePooling1D(strides=2)(convstat_1)
        convstat_2 = layers.Conv1D(32,3,activation='relu')(convstat_1pooled)
        convstat_2pooled = layers.AveragePooling1D(strides=2)(convstat_2)
        convstat_3 = layers.Conv1D(32,5,activation='relu')(convstat_2pooled)
        convstat_3 = layers.BatchNormalization()(convstat_3)
        convstat_4 = layers.Conv1D(48,5,activation='sigmoid')(convstat_3)
        convstat_4 = layers.BatchNormalization()(convstat_4)

        flatbody = layers.Flatten()(convbody_4)
        flatbody = layers.Dropout(0.8)(flatbody)
        flatstat = layers.Flatten()(convstat_4)
        flatstat = layers.Dropout(0.4)(flatstat)
        mergedlay = layers.Concatenate(axis=1)([flatbody,flatstat])
        outdense1 = layers.Dense(32,activation='relu')(mergedlay)
        outdense2 = layers.Dense(numclasses,activation='softmax')(outdense1)
        opt = Adam(learning_rate=0.001)
        modelret = Model(inputs=[Input_body,Input_statement],outputs=[outdense2])
        modelret.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["CategoricalAccuracy"])

        return modelret
    
    else:
        print("MODEL TYPE NOT FOUND. PLEASE DEFINE IN mymodels.py")






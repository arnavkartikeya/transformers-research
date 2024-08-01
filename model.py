import keras_nlp
import keras
import tensorflow as tf
from keras import layers
import numpy as np
from tensorflow import math, reshape, shape, transpose, cast, float32
from tensorflow.linalg import matmul
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from keras.backend import softmax

class Config:
    MAX_SEQ = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEADS = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1
    RATIO = 0.4
    DROPOUT_RATE = 0.2 

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """linear warm up - linear decay"""
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / self.decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        epoch = tf.cast(epoch, "float32")
        return self.calculate_lr(epoch)
    
    def get_config(self):
        config = {
            'init_lr': self.init_lr,
            'lr_after_warmup': self.lr_after_warmup,
            'final_lr': self.final_lr,
            'warmup_epochs': self.warmup_epochs,
            'decay_epochs': self.decay_epochs,
            'steps_per_epoch': self.steps_per_epoch,
        }
        return config


class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
 
    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x
 
        # Apply layer normalization to the sum
        return self.layer_norm(add)

class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer
 
    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)
 
        return self.fully_connected2(self.activation(x_fc1))

class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, ratio,**kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(d_k, d_k, d_v, d_model, h, ratio)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, syntactic_masks, padding_masks,training):
        # Multi-head attention layer
        multihead_output = None 
        multihead_output, attention = self.multihead_attention(x, syntactic_masks, padding_masks) 
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output), attention

class MultiHeadAttention(Layer): 
    def __init__(self, d_k, d_q, d_v, d_model, num_heads, ratio_applied, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.ratio = ratio_applied
        self.heads = num_heads
        self.d_k = d_k 
        self.d_v = d_v 
        self.d_q = d_q 
        self.queries = Dense(d_q)
        self.keys = Dense(d_k)
        self.values = Dense(d_v) 
        self.out = Dense(d_model) 
        self.default_batch = 32
        # self.masks = masks #shape (batch, 256, 256) 

        self.indx_apply_masks = np.random.rand(num_heads) > (1-self.ratio) #list of heads to apply the syntactic mask to (currently applies to 40% of masks) 
        self.invert = ~self.indx_apply_masks
        

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1) (32, 8, 256, 2) 
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x
        
    def call(self, embeds, masks, padding_masks):
        batch_size = self.default_batch if embeds.shape[0] is None else embeds.shape[0]
        #padding_mask shape: (batch, 256, 256) 
        #extended_mask shape: (batch, 8, 256, 256) 
        # embeds, masks = inputs #embed shape: (batch, input_seq, embed_dim), masks shape: (batch, input_seq, input_seq) 
        # extended_mask = np.ones((batch_size, self.heads, 256, 256))
        # extended_mask[:, self.indx_apply_masks] = masks[:, np.newaxis, :, :]
        # extended_mask[:, self.invert] = padding_masks[:, np.newaxis, :, :] 

        mask = tf.cast(masks, tf.float32)

        num_heads = self.heads
        mask_shape = mask.shape[1] 
        extended_mask = tf.ones((num_heads, batch_size, mask_shape, mask_shape))
        indx = tf.convert_to_tensor(self.indx_apply_masks, dtype=tf.bool)
        indx_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(indx, axis=-1), axis=-1), axis=-1)
        conditional_mask = tf.tile(indx_expanded, [1, batch_size, mask_shape, mask_shape])
        conditional_mask = tf.cast(conditional_mask, dtype=tf.bool)
        test = tf.tile(tf.expand_dims(mask, axis=0), [num_heads, 1, 1, 1])
        # print(f"test dtype: {test.dtype}") 
        # print(f"conditional dtype: {conditional_mask.dtype}")
        # print(f"extended mask: {extended_mask.dtype}")
        new_extended_mask = tf.where(conditional_mask, test, extended_mask)
        extended_mask = tf.transpose(new_extended_mask, perm=[1,0,2,3]) 
        extended_mask = tf.cast(extended_mask, tf.bool)



        # b_mask = (masks == 1)

        # extended = [] 
        # for i in self.indx_apply_masks: 
        #     if i: 
        #         extended.append(b_mask)
        #     else:
        #         extended.append(np.full((batch_size, masks.shape[1],masks.shape[1]), True)) 
        # extended = np.array(extended) 
        # extended = transpose(extended, perm=[1, 0, 2, 3])
        # self.extended_masks = extended

        key = self.keys(embeds) #shape: (batch, input_seq, embed_dim//num_heads)
        query = self.queries(embeds) #shape: (batch, input_seq, embed_dim//num_heads)
        values = self.values(embeds) #shape: (batch, input_seq, embed_dim//num_heads)

        reshaped_key = self.reshape_tensor(key, self.heads, True) 
        reshaped_query = self.reshape_tensor(query, self.heads, True) 
        reshaped_values = self.reshape_tensor(values, self.heads, True) 

        pre_softmax_scores = matmul(reshaped_query, reshaped_key, transpose_b=True) / math.sqrt(cast(self.d_k, float32)) #shape: (32, num_heads, 256, 256) 

        applied_masks = [] #shaped: (32, num_heads, 256, 256) where eaech mask is either 

        masked_scores = tf.where(extended_mask, pre_softmax_scores, -1e9)

        softmaxed_scores = softmax(masked_scores) 

        head_outputs = matmul(softmaxed_scores, reshaped_values) 
        
        output = self.reshape_tensor(head_outputs, self.heads, False)

        # pre_softmax_scores = matmul(queries, keys.transpose(-2, -1))/ math.sqrt(cast(d_k, float32)) #shape (32, 256, 256) 
        return self.out(output), softmaxed_scores
    
        

class TransformerModel(keras.Model): 
    def __init__(self, config): #config is from the ModelConfig class 
        # super(TransformerModel, self).__init__(**kwargs)
        super().__init__()
        self.config = config 
        #input layer 
        self.input_layer = keras.layers.Input((config.MAX_SEQ, ), dtype="int64") #not sure why we need the dtype (batch, max_seq) --> tensor version 
        #word embeddings 
        self.word_embedding = keras.layers.Embedding(config.VOCAB_SIZE, config.EMBED_DIM, input_length=config.MAX_SEQ) #output shape: (batch, input_length, embed_dim)
        #positional embeddings (does masking make this useless, because most of the information not between heuristics connections is gone??) 
        self.positional_embedding = keras_nlp.layers.SinePositionEncoding() # I think this is the correct one? https://keras.io/api/keras_nlp/modeling_layers/sine_position_encoding/
        # self.positional_embedding = keras_nlp.layers.PositionEmbedding(config.MAX_SEQ, initializer="glorot_uniform") #output shape: (batch, input_length, embed_dim) 

        #layer normalize 
        self.layer_norm = keras.layers.LayerNormalization() #output (batch, input_length, embed_dim) 
        
        #need multihead attention layer
        # self.encoder_layers = [MultiHeadAttention(d_k=(config.EMBED_DIM // config.NUM_HEAD), d_q=(config.EMBED_DIM // config.NUM_HEAD), d_v=(config.EMBED_DIM // config.NUM_HEAD), d_model=config.EMBED_DIM, num_heads=config.NUM_HEADS, config.RATIO) for i in range(config.NUM_LAYERS)] #output (batch, input_length, embed_dim) 

        self.encoder_layers = [EncoderLayer(h=config.NUM_HEADS, d_k=(config.EMBED_DIM // config.NUM_HEADS), d_v=(config.EMBED_DIM // config.NUM_HEADS), d_model=config.EMBED_DIM, d_ff=config.FF_DIM, rate=config.DROPOUT_RATE, ratio=config.RATIO) for _ in range(config.NUM_LAYERS)]
    

        #final dense layer to predict mlm output 
        self.output_layer = Dense(config.VOCAB_SIZE, activation="softmax") 
        

    def call(self, inputs, output_attention=False): 
        #steps: 
        #unpack inputs into masks and actual training sequence 
        real_inputs, masks, padding_masks = inputs 
        attentions = [] 
        x = None 
        # x = self.input_layer(real_inputs)
        #wordembeddings(input_layer)
        word_embeddings = self.word_embedding(real_inputs) 
        #positional_embeddings(word_embeddings) 
        pos_embeddings = self.positional_embedding(word_embeddings) 
        #embeddings = word_embeddings + positional_embedding (Not sure why we do this, huggingface bert doesn't seem to do this) 
        embeddings = word_embeddings + pos_embeddings
        #layernorm(embeddings) #not including dropout cause that could fuck up the hueristic (for now) 
        # x = self.layer_norm(embeddings)
        x = embeddings
        #config.num_head attention layers 
        attention = None 
        for i, layer in enumerate(self.encoder_layers):
            x, attention = layer(x, masks, padding_masks)
            attentions.append(attention)
        if output_attention:
            return self.output_layer(x), attentions 
        return self.output_layer(x), attentions

def compile_model(config=None):
    if config: 
        mlm_model = TransformerModel(config)
        optimizer = keras.optimizers.Adam(learning_rate=config.LR)
        mlm_model.compile(optimizer=optimizer, metrics=["accuracy"]) #add parameter run_eagerly=True if this shit doesnt work 
        return mlm_model 
    config = Config() 
    mlm_model = TransformerModel(config)
    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer, metrics=["accuracy"]) #add parameter run_eagerly=True if this shit doesnt work 
    return mlm_model 

def get_optimizer():
    optimizer = keras.optimizers.Adam(learning_rate=CustomSchedule())
    return optimizer 



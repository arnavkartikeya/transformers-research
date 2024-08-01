import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_nlp
import keras
import tensorflow as tf
from keras import layers
from keras.layers import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
from nltk.tokenize import WhitespaceTokenizer
import re 
import sys 

class Dataset(): 
    def __init__(self, config): 
        self.config = config 
        self.vectorize_layer = None 
        self.id2token = None 
        self.mask_token_id = -1

    def get_text_list_from_files(self, files):
        text_list = []
        for name in files:
            with open(name) as f:
                for line in f:
                    text_list.append(line)
        return text_list
    
    
    def get_data_from_text_files(self, folder_name):
    
        pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
        pos_texts = self.get_text_list_from_files(pos_files)
        neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
        neg_texts = self.get_text_list_from_files(neg_files)
        df = pd.DataFrame(
            {
                "review": pos_texts + neg_texts,
                "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
            }
        )
        df = df.sample(len(df)).reset_index(drop=True)
        return df


    def custom_standardization(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~\""), "" # original line: "!#$%&'()*+,-./:;<=>?@\^_`{|}~"
        )

    
    def get_vectorize_layer(self, texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
        """Build Text vectorization layer
    
        Args:
          texts (list): List of string i.e input texts
          vocab_size (int): vocab size
          max_seq (int): Maximum sequence lenght.
          special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].
    
        Returns:
            layers.Layer: Return TextVectorization Keras Layer
        """
        vectorize_layer = TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            standardize=self.custom_standardization,
            output_sequence_length=max_seq,
        )
        vectorize_layer.adapt(texts)
    
        # Insert mask token in vocabulary
        vocab = vectorize_layer.get_vocabulary()
        vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[mask]"]
        vectorize_layer.set_vocabulary(vocab)
        self.vectorize_layer = vectorize_layer

        id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
        self.id2token = id2token


        mask_token_id = vectorize_layer(["[mask]"]).numpy()[0][0]
        # print(f"mask_token_id: {mask_token_id}")
        self.mask_token_id = mask_token_id

        return vectorize_layer
    
    def decode(self, tokens):
        return " ".join([self.id2token[t] for t in tokens if t != 0])
    
    def encode(self, texts):
        encoded_texts = self.vectorize_layer(texts)
        return encoded_texts.numpy()


    def generate_mask(self, vectorized_txt): 
        #generating padding_masks
        
        
        # print('vectorized_txt:') 
        # print(len(vectorized_txt))
        # print(vectorized_txt)
        pos_vals = {"D", "J", "N", "P", "V", "R"}
        map_to_pos = {"D":"verb", "J":"adj", "N":"noun", "P":"pronoun", "V":"verb", "R": "adverb"}
        relationships = {"noun": {"verb", "adj", "pronoun"},  "verb":{"noun", "adverb", "pronoun"}, "adj":{"noun", "pronoun"}, "pronoun":{"verb", "adj", "adverb", "noun"}, "adverb": {"verb"}}
    
        
        decoded_txt = "" 
        word_count =  0 
        #assuming the shape of vectorized_txt is (256,) 
        for i in range(len(vectorized_txt)):
            vectorized_word = vectorized_txt[i] 
            if vectorized_word == 0:
                break 
            else:
                #doing this in case the last word is a specialized space token and it gets stripped at the end 
                if i != len(vectorized_txt)-1 and vectorized_txt[i+1] != 0 :
                    decoded_txt += self.decode([vectorized_word]) + " " 
                else:
                    decoded_txt += self.decode([vectorized_word]) 
                word_count += 1 


        #generating padding mask: 
        padding_mask = np.zeros((256, 256))
        padding_mask[:word_count, :word_count] = 1 
        padding_mask = padding_mask.astype('b') 
        
        # print(word_count)
        # print(decoded_txt)
    
        remaining_vals = self.config.MAX_SEQ - word_count #change this to config.MAX_SIZE afterwards 
        
        # tokenized = word_tokenize(decoded_txt)
        # tokenized=WhitespaceTokenizer().tokenize(decoded_txt)
        tokenized = decoded_txt.split(" ")
        tagged = nltk.pos_tag(tokenized)
        # print(f'length of tagged before processing {len(tagged)}') 
        # print(tagged)
        word_pos = []
        for j in range(len(tagged)): 
            word, pos = tagged[j] 
            if word == ']' and j != 0 and (tagged[j-1][0] == 'mask' or tagged[j-1][0] == 'UNK'):
                word_pos.pop(-1)
                word_pos.pop(-1)
                    # print(decoded_txt) 
                    # sys.quit() 
                word_pos.append(None) 
            elif ord(word[0]) >= 127 and len(word) == 1 or pos[0] not in pos_vals: 
                word_pos.append(None)
            else:
                word_pos.append(map_to_pos[pos[0]]) 
    
        # print(f'after processing {len(word_pos)}')
        # print(word_pos)
    
        mask = np.ones((len(word_pos), len(word_pos)))
    
        for row in range(mask.shape[0]):
            row_word_pos = word_pos[row] 
            if row_word_pos: #the [CLS], [SEP], and [MASK] tokens will be allowed to be influenced by every word 
                allowed_pos = relationships[row_word_pos] 
                for col in range(mask.shape[1]): 
                    #for now, don't worry about the [mask] token not being allowed to influence the words 
                    if word_pos[col] not in allowed_pos: 
                        mask[row,col] = 0
                    # else:
                    #     mask[row, col] = 0 
    
        np.fill_diagonal(mask, 1) 
    
        if remaining_vals > 0: 
            true_mask = np.zeros((self.config.MAX_SEQ, self.config.MAX_SEQ)) #change this to config.MAX_SIZE afterwards 
            try:
                true_mask[:word_count, :word_count] = mask 
            except:
                print(f'mask shape: {mask.shape}') 
                print(f'word_count: {word_count}')
                mask_for_debugging = vectorized_txt 
                print(decoded_txt)
                sys.quit()
            mask = true_mask
    
        mask = mask.astype('b')
        
        return mask, padding_mask 
    
        
    def get_masked_input_and_labels(self, encoded_texts, verbose=0):
        # 15% BERT masking
        inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
        # Do not mask special tokens
        inp_mask[encoded_texts <= 2] = False
        # Set targets to -1 by default, it means ignore
        labels = -1 * np.ones(encoded_texts.shape, dtype=int)
        # Set labels for masked tokens
        labels[inp_mask] = encoded_texts[inp_mask]
    
        # Prepare input
        encoded_texts_masked = np.copy(encoded_texts)
        # Set input to [MASK] which is the last token for the 90% of tokens
        # This means leaving 10% unchanged
        inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
        encoded_texts_masked[
            inp_mask_2mask
        ] = self.mask_token_id  # mask token is the last in the dict
    
        # Set 10% to a random token
        inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
        encoded_texts_masked[inp_mask_2random] = np.random.randint(
            3, self.mask_token_id, inp_mask_2random.sum()
        )
    
        # Prepare sample_weights to pass to .fit() method
        sample_weights = np.ones(labels.shape)
        sample_weights[labels == -1] = 0
    
        # y_labels would be same as encoded_texts i.e input tokens
        y_labels = np.copy(encoded_texts)
    
        # all_syntactic_masks = []
        all_padding_masks = np.empty((encoded_texts_masked.shape[0], self.config.MAX_SEQ, self.config.MAX_SEQ), dtype='b') 
        all_syntactic_masks = np.empty((encoded_texts_masked.shape[0], self.config.MAX_SEQ, self.config.MAX_SEQ), dtype='b')
        # c = 0 
        count = 0 
        for sentence in encoded_texts_masked:
            # print(c)
            if verbose == 1 and count%1000 == 0: 
                print(f"starting generating mask for {count}/{encoded_texts_masked.shape[0]} sentence...")
            # all_syntactic_masks.append(generate_mask(sentence), axis=0)
            all_syntactic_masks[count], all_padding_masks[count] = self.generate_mask(sentence)
            # print("Size of the array in bytes:", all_syntactic_masks.nbytes)
            if verbose == 1 and count %1000 == 0:
                print("finished generating mask for 1 sentence") 
            count += 1 
            # c += 1 
        print("exited loop")
        print("converted mask to numpy array") 
        # count = 0
        # for sentence in encoded_texts_masked:
        #     if count == 3: 
        #         break 
        #     print(f'count: {count}, decoded sentence: {decode(sentence)}')
        #     count += 1 
    
        # #decode for masking 
        # all_syntactic_masks = [] 
        # for sentence in encoded_texts_masked: 
        #     decoded_sentence = decode(sentence) 
        #     temp = "[CLS] " + decoded_sentence + " [SEP]" 
        #     syntactic_mask = generate_mask(temp)
        #     all_syntactic_masks.append(syntactic_mask) 
        # print(f'all_syntatic_masks shape: {len(all_syntactic_masks)}')
        # # count = 1 
        # # for sentence in encoded_texts_masked:
        # #     if count == 3: 
        # #         break 
        # #     print(f'count: {count}, decoded sentence: {decode(sentence)}')
        # #     count += 1 
    
        #rreturn encoded_texts_masked, y_labels, sample_weights, all_syntactic_masks
        return encoded_texts_masked, y_labels, all_syntactic_masks, all_padding_masks, sample_weights

    def generate_dataset(self): 
        train_df = self.get_data_from_text_files("train")
        test_df = self.get_data_from_text_files("test")
        
        all_data = pd.concat([train_df, test_df])   
        
        vectorize_layer = self.get_vectorize_layer(
            all_data.review.values.tolist(),
            self.config.VOCAB_SIZE,
            self.config.MAX_SEQ,
            special_tokens=["[mask]"],
        )
        # mask_token_id = self.vectorize_layer(["[mask]"]).numpy()[0][0]
    
        x_train = self.encode(train_df.review.values)  # encode reviews with vectorizer
        y_train = train_df.sentiment.values
        train_classifier_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(1000)
            .batch(self.config.BATCH_SIZE)
        )
    
        x_test = self.encode(test_df.review.values)
        y_test = test_df.sentiment.values
        test_classifier_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
            self.config.BATCH_SIZE
        )
    
        test_raw_classifier_ds = tf.data.Dataset.from_tensor_slices(
            (test_df.review.values, y_test)
        ).batch(self.config.BATCH_SIZE)
    
        # Prepare data for masked language model
        x_all_review = self.encode(all_data.review.values)
        
        print("starting generating masks...")
        
        x_masked_train, y_masked_labels, syntactic_masks, padding_masks, sample_weights = self.get_masked_input_and_labels(
            x_all_review, verbose=1
        )
        
        print("finished generating mask")

        def generator_inputs():
            for i in range(x_masked_train.shape[0]):
                x_data = x_masked_train[i]
                y_data = y_masked_labels[i]
                syntactic_mask = syntactic_masks[i] 
                padding_mask = padding_masks[i]
                sample_weight = sample_weights[i]
        
                x_data_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32) 
                y_data_tensor = tf.convert_to_tensor(y_data, dtype=tf.float32) 
                syntactic_mask_tensor = tf.convert_to_tensor(syntactic_mask, dtype=tf.bool) 
                padding_mask_tensor = tf.convert_to_tensor(padding_mask, dtype=tf.bool)
                sample_weights_tensor = tf.convert_to_tensor(sample_weight, dtype=tf.float32) 
                
                yield x_data_tensor, y_data_tensor, syntactic_mask_tensor, padding_mask_tensor, sample_weights_tensor
    
        print("started forming dataset") 
        mlm_ds = tf.data.Dataset.from_generator(generator_inputs, 
                                                output_signature=(tf.TensorSpec(shape=(self.config.MAX_SEQ,), dtype=tf.int32), 
                                                 tf.TensorSpec(shape=(self.config.MAX_SEQ,), dtype=tf.int32), 
                                                 tf.TensorSpec(shape=(self.config.MAX_SEQ,self.config.MAX_SEQ), dtype=tf.bool),
                                                 tf.TensorSpec(shape=(self.config.MAX_SEQ,self.config.MAX_SEQ), dtype=tf.bool),
                                                 tf.TensorSpec(shape=(self.config.MAX_SEQ,), dtype=tf.int32)))
        print("finished forming dataset") 

        return mlm_ds

def get_dataset_class(config):
    dataset = Dataset(config)
    return dataset 



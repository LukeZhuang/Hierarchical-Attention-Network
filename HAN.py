import tensorflow as tf
import numpy as np
from tensorflow.python.layers.base import Layer


class HAN(Layer):
    def __init__(self,num_classes,word2vec_size,lamb=0.0,disTill=1.0):
        self.num_classes=num_classes
        self.word2vec_size=word2vec_size
        self.lamb=lamb
        self.disTill=disTill
        
        # Part I: word level attention
        self.gru_hidden_size_low=128
        self.attention_hidden_low=128
        self.cell_fw_low=tf.contrib.rnn.GRUCell(self.gru_hidden_size_low)
        self.cell_bw_low=tf.contrib.rnn.GRUCell(self.gru_hidden_size_low)
        self.word_attention_net_low=tf.layers.Dense(units=self.attention_hidden_low,activation=tf.nn.tanh,name='word_attention_net_low')
        
        # Part II: sentence level attention
        self.gru_hidden_size_high=128
        self.attention_hidden_high=128
        self.cell_fw_high=tf.contrib.rnn.GRUCell(self.gru_hidden_size_high)
        self.cell_bw_high=tf.contrib.rnn.GRUCell(self.gru_hidden_size_high)
        self.word_attention_net_high=tf.layers.Dense(units=self.attention_hidden_high,activation=tf.nn.tanh,name='word_attention_net_high')
        
    def mask_attention(self,attention_matrix,index_length):
        s0=tf.shape(attention_matrix)[0]
        s1=tf.shape(attention_matrix)[1]
        index=tf.tile(tf.reshape(tf.range(s1),(1,-1)),multiples=[s0,1])
        mask=tf.cast(index<tf.reshape(index_length,(-1,1)),tf.float32)
        return mask*attention_matrix     
        
    def __call__(self,X_sentence,y,sentence_length,document_length,dropout,is_training):
        y_onehot=tf.one_hot(y,self.num_classes)
        batch_size=tf.shape(X_sentence)[0]
        document_size=tf.shape(X_sentence)[1]
        sentence_size=tf.shape(X_sentence)[2]
        
        # Part I: word level attention
        X_sentence_r=tf.reshape(X_sentence,(batch_size*document_size,sentence_size,self.word2vec_size)) # shape=(batch_size*doc_size,sen_size,vocab)
        with tf.variable_scope("word_level") as vs:
            # shape=(batch_size*doc_size,sen_size,gru_size)
            low_output=tf.nn.bidirectional_dynamic_rnn(self.cell_fw_low,self.cell_bw_low,
                                                       dtype=tf.float32,
                                                       inputs=X_sentence_r,
                                                       sequence_length=sentence_length,
                                                       time_major=False)
        low_output=tf.concat(low_output[0],axis=2)  # shape=(batch_size*doc_size,sen_size,2*gru_size)
        uw=tf.get_variable("uw", dtype=tf.float32, 
                           shape=(self.attention_hidden_low,1),
                           initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        low_output_r=tf.reshape(low_output,(-1,2*self.gru_hidden_size_low))  # shape=(batch_size*doc_size*sen_size,2*gru_size)
        low_output_r=self.word_attention_net_low(low_output_r)  # shape=(batch_size*doc_size*sen_size,attention_hidden_low)
        low_output_r=tf.layers.dropout(inputs=low_output_r,rate=dropout,training=is_training)
        score_low=tf.matmul(low_output_r,uw)  # shape=(batch_size*doc_size*sen_size,1)
        score_low=tf.reshape(score_low,(batch_size*document_size,sentence_size))  # shape=(batch_size*doc_size,sen_size)
        attention_low=tf.nn.softmax(score_low/self.disTill,dim=1)
        sentence_vector=(tf.reshape(attention_low,(batch_size*document_size,sentence_size,1))*low_output)  # shape: same as low_output
        sentence_vector=tf.reduce_sum(sentence_vector,axis=1)  # shape=(batch_size*doc_size,2*gru_size)
        sentence_vector=tf.reshape(sentence_vector,(batch_size,document_size,2*self.gru_hidden_size_low))  # shape=(batch_size,doc_size,2*gru_size)
        
        
        # Part II: sentence level attention
        with tf.variable_scope("sentence_level") as vs:
            high_output=tf.nn.bidirectional_dynamic_rnn(self.cell_fw_high,self.cell_bw_high,
                                                       dtype=tf.float32,
                                                       inputs=sentence_vector,
                                                       sequence_length=document_length,
                                                       time_major=False)
        high_output=tf.concat(high_output[0],axis=2)  # shape=(batch_size,doc_size,2*gru_size)

        us=tf.get_variable("us", dtype=tf.float32, 
                           shape=(self.attention_hidden_high,1),
                           initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        high_output_r=tf.reshape(high_output,(-1,2*self.gru_hidden_size_high))  # shape=(batch_size*doc_size,2*gru_size)
        high_output_r=self.word_attention_net_high(high_output_r)  # shape=(batch_size*doc_size,attention)
        high_output_r=tf.layers.dropout(inputs=high_output_r,rate=dropout,training=is_training)
        score_high=tf.matmul(high_output_r,us)  # shape=(batch_size*doc_size,1)
        score_high=tf.reshape(score_high,(batch_size,document_size))  # shape=(batch_size,doc_size)
        attention_high=tf.nn.softmax(score_high/self.disTill,dim=1)  # shape=(batch_size,doc_size)
        output=(tf.reshape(attention_high,(batch_size,document_size,1))*high_output)  # shape=(batch_size,doc_size,2*gru_size)
        output=tf.reduce_sum(output,axis=1)  # shape=(batch_size,2*gru_size)
        
        # Part III: predict
        score=tf.layers.dense(output,units=self.num_classes,name='predict_net')

        loss=tf.losses.softmax_cross_entropy(onehot_labels=y_onehot,logits=score)
#         masked_attention_low=self.mask_attention(attention_low,sentence_length)  # does not count in the filled 0s after sentence
#         masked_attention_high=self.mask_attention(attention_high,document_length)
#         loss+=self.lamb1*tf.reduce_mean(tf.reduce_sum(tf.square(masked_attention_low),axis=1))
#         loss+=self.lamb2*tf.reduce_mean(tf.reduce_sum(tf.square(masked_attention_high),axis=1))
        l_variables=self.word_attention_net_low.variables
        h_variables=self.word_attention_net_high.variables
        loss+=self.lamb*(tf.reduce_sum(tf.square(l_variables[0]))+tf.reduce_sum(tf.square(l_variables[1]))+
                         tf.reduce_sum(tf.square(h_variables[0]))+tf.reduce_sum(tf.square(h_variables[0]))+
                         tf.reduce_sum(tf.square(uw))+tf.reduce_sum(tf.square(us)))
        
        predict=tf.cast(tf.argmax(score,axis=1),dtype=tf.int32)
        accuracy=tf.reduce_mean(tf.cast(tf.equal(y,predict),dtype=tf.float32))
        
        return loss,predict,accuracy,attention_low,attention_high
    
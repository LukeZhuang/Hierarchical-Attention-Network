import numpy as np
import random

class Dataset():
    def __init__(self,data,word2vec_model,word2vec_size,cut_long_sentence):
        self.cursor=0
        self.data=data
        self.word2vec_model=word2vec_model
        self.word2vec_size=word2vec_size
        self.cut_long_sentence=cut_long_sentence
        
    def initialize(self):
        self.cursor=0
    
    def start_epoch(self):
        self.cursor=0
        random.shuffle(self.data)
        
    def fill_np(self,data):
        '''
        from: https://stackoverflow.com/questions/32037893/numpy-fix-array-with-rows-of-different-lengths-by-filling-the-empty-elements-wi
        '''
        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:,None]

        # Setup output array and put elements from data into masked positions
        out = np.zeros(mask.shape, dtype=data.dtype)
        out[mask] = np.concatenate(data)
        return out
        
    def next_batch(self,batch_size=64):
        batch=self.data[self.cursor:self.cursor+batch_size]
        labels=np.array([d[1] for d in batch])-1
        self.cursor+=batch_size
        document_sizes=np.array([len(d[0]) for d in batch])
        document_size=np.max(document_sizes)
        
        sentence_sizes=np.array([[min(len(s),self.cut_long_sentence) for s in d[0]] for d in batch])
#         sentence_size = max(map(max, sentence_sizes))
        sentence_sizes=self.fill_np(sentence_sizes)
        sentence_size=np.max(sentence_sizes)
        
        # shape=(batch_size, document_len, sentence_len, word2vec_len)
        output=np.zeros((batch_size,document_size,sentence_size,self.word2vec_size))
        
        for (id_d,d) in enumerate(batch):
            for (id_s,s) in enumerate(d[0]):
                for (id_w,w) in enumerate(s):
                    if id_w>=self.cut_long_sentence:  # cut too long sentences
                        break
                    output[id_d,id_s,id_w,:]=self.word2vec_model[w]
        
        return output,document_sizes,sentence_sizes,labels
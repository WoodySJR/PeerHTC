from mxnet.contrib import text
import pandas as pd
from mxnet import nd, autograd, init, gluon
import collections
from sklearn.model_selection import train_test_split
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils, nn, rnn
import numpy as np
import d2lzh as d2l
from tqdm import tqdm
import mxnet as mx
import time
import gluonnlp as nlp


class PeerHTC(nn.Block):
    def __init__(self, embed_size, num_hiddens, num_layers, **kwargs):
        super(HiAGM, self).__init__(**kwargs)
        
        # parameters
        ## embed_size: dim of word embeddings
        ## num_hiddens: dim of hidden states of each direction in lstm
        ## num_layers: number of layers in lstm
        ## dv: dim of original label embeddings
        ## dh: dim of hidden states of each direction in structure encoder
        
        # activations
        self.tanh = nn.Activation('tanh')
        self.relu = nn.Activation('relu')
        self.sigmoid = nn.Activation('sigmoid')
        
        # structure encoder
        ## original label embedding (trainable)
        self.v1 = self.params.get('v1', shape = (dv, c1))
        self.v2 = self.params.get('v2', shape = (dv, c2))
        self.v3 = self.params.get('v3', shape = (dv, c3))
        
        ## bottom-up frequency matrix (initialized properly and trainable)
        self.fre32 = self.params.get('f32', shape = (c3, c2))
        self.fre21 = self.params.get('f21', shape = (c2, c1))
        
        ## top-down manner

        self.wi1 = self.params.get('wi1', shape = (dh, dv))
        self.ui1 = self.params.get('ui1', shape = (dh, dh))
        self.bi1 = self.params.get('bi1', shape = (dh, 1))
        
        self.wf1 = self.params.get('wf1', shape = (dh, dv))
        self.uf1 = self.params.get('uf1', shape = (dh, dh))
        self.bf1 = self.params.get('bf1', shape = (dh, 1))
        
        self.wo1 = self.params.get('wo1', shape = (dh, dv))
        self.uo1 = self.params.get('uo1', shape = (dh, dh))
        self.bo1 = self.params.get('bo1', shape = (dh, 1))
        
        self.wu1 = self.params.get('wu1', shape = (dh, dv))
        self.uu1 = self.params.get('uu1', shape = (dh, dh))
        self.bu1 = self.params.get('bu1', shape = (dh, 1))
        
        ## bottom-up manner
        
        self.wi2 = self.params.get('wi2', shape = (dh, dv))
        self.ui2 = self.params.get('ui2', shape = (dh, dh))
        self.bi2 = self.params.get('bi2', shape = (dh, 1))
        
        self.wf2 = self.params.get('wf2', shape = (dh, dv))
        self.uf2 = self.params.get('uf2', shape = (dh, dh))
        self.bf2 = self.params.get('bf2', shape = (dh, 1))
        
        self.wo2 = self.params.get('wo2', shape = (dh, dv))
        self.uo2 = self.params.get('uo2', shape = (dh, dh))
        self.bo2 = self.params.get('bo2', shape = (dh, 1))
        
        self.wu2 = self.params.get('wu2', shape = (dh, dv))
        self.uu2 = self.params.get('uu2', shape = (dh, dh))
        self.bu2 = self.params.get('bu2', shape = (dh, 1))
        
        # whole-hierarchy GCN
        self.Ad = self.params.get('Ad', shape = (K, K))
        self.W = self.params.get('W', shape = (dv, dv))
        
        # levelwise GCN
        #self.Ad1 = self.params.get('Ad1', shape = (c1, c1))
        #self.W1 = self.params.get('W1', shape = (dv, dv))
        #self.Ad2 = self.params.get('Ad2', shape = (c2, c2))
        #self.W2 = self.params.get('W2', shape = (dv, dv))
        #self.Ad3 = self.params.get('Ad3', shape = (c3, c3))
        #self.W3 = self.params.get('W3', shape = (dv, dv))
        
        self.project = self.params.get('project', shape = (fea_dim, 3*dv))
        self.project_2 = self.params.get('project_2', shape = (768, feature_dim))
        
        # document representation (with LSTM or BERT)
        #self.embedding = nn.Embedding(len(word_emb), embed_size)
        #self.encoder = rnn.LSTM(num_hiddens, num_layers = num_layers, bidirectional = True, input_size = embed_size)
        
        self.encoder,_ = nlp.model.get_model(name='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased',
                                            pretrained=True, ctx=mx.gpu(),
                                            use_pooler=False, use_classifier=False,
                                             use_token_type_embed=False, use_decoder=False,
                                            dropout=dropout_rate)
        
        # global classifier (based on multi-label attention)
        self.wg1 = self.params.get('wg1', shape = (2*num_hiddens*(c1+c2+c3), d1))
        self.bg1 = self.params.get('bg1', shape = (1, d1))
        self.wg2 = self.params.get('wg2', shape = (d1,d2))
        self.bg2 = self.params.get('bg2', shape = (1, d2))
        self.wg3 = self.params.get('wg3', shape = (d2, d3))
        self.bg3 = self.params.get('bg3', shape = (1, d3))
        self.wfinal = self.params.get('wfinal', shape = (d3, c1+c2+c3))
        self.bfinal = self.params.get('bfinal', shape = (1, c1+c2+c3))

    def forward(self, inputs):
        
        # structure encoder
        ## top-down manner
        ## 1
        i11 = self.sigmoid(nd.dot(self.wi1.data(), self.v1.data()) + self.bi1.data())
        o11 = self.sigmoid(nd.dot(self.wo1.data(), self.v1.data()) + self.bo1.data())
        u11 = self.tanh(nd.dot(self.wu1.data(), self.v1.data()) + self.bu1.data())
        c11 = i11 * u11
        h11 = o11 * self.tanh(c11)
        
        ## 1->2
        h21_tilde = nd.dot(h11, w12)
        i21 = self.sigmoid(nd.dot(self.wi1.data(), self.v2.data()) + nd.dot(self.ui1.data(), h21_tilde) + self.bi1.data())
        f21 = self.sigmoid(nd.dot(self.wf1.data(), self.v2.data()) + nd.dot(self.uf1.data(), nd.dot(h11, w12)) + self.bf1.data())
        o21 = self.sigmoid(nd.dot(self.wo1.data(), self.v2.data()) + nd.dot(self.uo1.data(), h21_tilde) + self.bo1.data())
        u21 = self.tanh(nd.dot(self.wu1.data(), self.v2.data()) + nd.dot(self.uu1.data(), h21_tilde) + self.bu1.data())
        c21 = i21 * u21 + f21 * nd.dot(c11, w12)
        h21 = o21 * self.tanh(c21)
        
        ## 2->3
        h31_tilde = nd.dot(h21, w23)
        i31 = self.sigmoid(nd.dot(self.wi1.data(), self.v3.data()) + nd.dot(self.ui1.data(), h31_tilde) + self.bi1.data())
        f31 = self.sigmoid(nd.dot(self.wf1.data(), self.v3.data()) + nd.dot(self.uf1.data(), nd.dot(h21, w23)) + self.bf1.data())
        o31 = self.sigmoid(nd.dot(self.wo1.data(), self.v3.data()) + nd.dot(self.uo1.data(), h31_tilde) + self.bo1.data())
        u31 = self.tanh(nd.dot(self.wu1.data(), self.v3.data()) + nd.dot(self.uu1.data(), h31_tilde) + self.bu1.data())
        c31 = i31 * u31 + f31 * nd.dot(h21, w23)
        h31 = o31 * self.tanh(c31)
        
        ## bottom-up manner
        ## 3
        i32 = self.sigmoid(nd.dot(self.wi2.data(), self.v3.data()) + self.bi2.data())
        o32 = self.sigmoid(nd.dot(self.wo2.data(), self.v3.data()) + self.bo2.data())
        u32 = self.tanh(nd.dot(self.wu2.data(), self.v3.data()) + self.bu2.data())
        c32 = i32 * u32
        h32 = o32 * self.tanh(c32)
        
        ## 3->2
        h22_tilde = nd.dot(h32, self.fre32.data())
        i22 = self.sigmoid(nd.dot(self.wi2.data(), self.v2.data()) + nd.dot(self.ui2.data(), h22_tilde) + self.bi2.data())
        f22 = self.sigmoid(nd.dot(self.wf2.data(), nd.dot(self.v2.data(), w23)) + nd.dot(self.uf2.data(), h32) + self.bf2.data())
        o22 = self.sigmoid(nd.dot(self.wo2.data(), self.v2.data()) + nd.dot(self.uo2.data(), h22_tilde) + self.bo2.data())
        u22 = self.tanh(nd.dot(self.wu2.data(), self.v2.data()) + nd.dot(self.uu2.data(), h22_tilde) + self.bu2.data())
        c22 = i22 * u22 + nd.dot(f22*c32, w23.transpose((1,0)))
        h22 = o22 * self.tanh(c22)
        
        ## 2->1
        h12_tilde = nd.dot(h22, self.fre21.data())
        i12 = self.sigmoid(nd.dot(self.wi2.data(), self.v1.data()) + nd.dot(self.ui2.data(), h12_tilde) + self.bi2.data())
        f12 = self.sigmoid(nd.dot(self.wf2.data(), nd.dot(self.v1.data(), w12)) + nd.dot(self.uf2.data(), h22) + self.bf2.data())
        o12 = self.sigmoid(nd.dot(self.wo2.data(), self.v1.data()) + nd.dot(self.uo2.data(), h12_tilde) + self.bo2.data())
        u12 = self.tanh(nd.dot(self.wu2.data(), self.v1.data()) + nd.dot(self.uu2.data(), h12_tilde) + self.bu2.data())
        c12 = i12 * u12 + nd.dot(f12*c22, w12.transpose((1,0)))
        h12 = o12 * self.tanh(c12)
        
        ## concatenation among top-down and bottom-up directions
        h1 = nd.concat(h11, h12, dim = 0)
        h2 = nd.concat(h21, h22, dim = 0)
        h3 = nd.concat(h31, h32, dim = 0)
        h = nd.concat(h1,h2,h3, dim = 1)
        
        # peer-label learning with GCN
        label_embedding = nd.concat(self.v1.data(),self.v2.data(),self.v3.data(),dim=1).transpose((1,0))
        
        ## level-wise
        #hh1 = self.relu(nd.dot(nd.dot(self.Ad.data(), label_embedding[0:c1,:]),self.W.data()))
        #hh2 = self.relu(nd.dot(nd.dot(self.Ad2.data(), label_embedding[c1:(c1+c2),:]),self.W2.data()))
        #hh3 = self.relu(nd.dot(nd.dot(self.Ad3.data(), label_embedding[(c1+c2):,:]),self.W3.data()))
        #hh = nd.concat(hh1,hh2,hh3,dim=0)
        
        ## whole-hierarchy
        hh = self.relu(nd.dot(nd.dot(self.Ad.data(), label_embedding),self.W.data()))
        h = nd.dot(self.project.data(), nd.concat(h,hh.transpose((1,0)),dim=0))
        
        # document representation (with LSTM or BERT)
        #embeddings = self.embedding(inputs.T)
        #outputs = self.encoder(embeddings) # shape: (#words, batch size, 2*num_hiddens)
        #outputs = outputs.transpose((1,0,2))
        N = inputs.shape[0] # batch size
        token_types = nd.zeros((N,max_len),ctx=mx.gpu())
        outputs = self.encoder(inputs,token_types)
        outputs = nd.batch_dot(outputs, self.project_2.data()*nd.ones((N,768,feature_dim),ctx=mx.gpu()))
        
        # global classifier 
        ## 2 fully-connected layers
        dot = nd.batch_dot(outputs, nd.ones((N, fea_dim, c1+c2+c3), ctx = mx.gpu()) * h)
        att = dot.exp() / dot.exp().sum(axis = 1, keepdims = True)
        fea = nd.batch_dot(outputs.transpose((0,2,1)), att)
        fea = fea.reshape((N, -1))
        
        fea1 = self.tanh(nd.dot(fea, self.wg1.data()) + self.bg1.data())
        fea2 = self.tanh(nd.dot(fea1,self.wg2.data()) + self.bg2.data())
        fea3 = self.tanh(nd.dot(fea2, self.wg3.data()) + self.bg3.data())
        out = self.sigmoid(nd.dot(fea3, self.wfinal.data()) + self.bfinal.data())
       
        return(out)
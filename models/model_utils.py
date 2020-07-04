from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
import time



# self-attention layer
class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, projected_size, attenType='scaled'):
        super(SelfAttentionLayer, self).__init__()
        self.W_K = nn.Linear(hidden_dim, projected_size)
        self.W_Q = nn.Linear(hidden_dim, projected_size)
        self.W_V = nn.Linear(hidden_dim, projected_size)
        
        self.hidden_dim = hidden_dim
        self.projected_size = projected_size
        self.attenType = attenType

    def forward(self, K, Q, V):
        '''
        K [batch_size, num_frames, hidden_dim] // key K
        Q [batch_size, num_frames, hidden_dim] // query, Q
        V [batch_size, num_frames, hidden_dim] // value, V
        return [batch_size, num_frames, hidden_dim] :  value after self-attn, V'
        '''
        assert K.size(0) == Q.size(0)
        assert Q.size(0) == V.size(0)
        
        batch_size, num_frames, hidden_dim = K.size()
        V_out = Variable(torch.ones((batch_size, num_frames, hidden_dim)), requires_grad=True)        
        
#         # compute attention score [batch_size, num_frames, projected_size]
#         if hidden_dim != self.projected_size:
#             print ('hidden_dim != projected_size, need to project first!')
        K = F.dropout(self.W_K(K))
        Q = F.dropout(self.W_Q(Q))
        V = F.dropout(self.W_V(V))

        K = K.permute(0, 2, 1) # [batch_size, hidden_dim, num_frames] 
        
        if self.attenType=='scaled':
            C = torch.bmm(Q, K).view(batch_size, num_frames, num_frames) / np.sqrt(hidden_dim)
        else:
            C = torch.bmm(Q, K).view(batch_size, num_frames, num_frames)
            
        # normalise with softmax
        C = F.softmax(C, dim=2)
        
        V_out = torch.bmm(C, V)
        
        '''
        for index in range(num_frames): 
            #(batch_size, num_frames, hidden_dim)
            v = V[:, index, :].unsqueeze(1).expand(batch_size, num_frames, hidden_dim)
            c = C[:, index, :].unsqueeze(2).expand(batch_size, num_frames, hidden_dim)
            v_out = torch.sum(v * c, 1).unsqueeze(1)
            
            #print('v.shape', v.shape)
            #print('c.shape', c.shape)
            #print('v_out.shape', v_out.shape)
            
            if index == 0: 
                V_out = v_out
            else:
                V_out = torch.cat((V_out, v_out), 1)
            #print('V_out', V_out)
        '''
        
        return V_out
    
        
        
# attention layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim_K, hidden_dim_q, projected_size, attenType='additive'):
        super(AttentionLayer, self).__init__()
        self.W_K = nn.Linear(hidden_dim_K, projected_size)
        self.W_q = nn.Linear(hidden_dim_q, projected_size)
        self.W_qK = nn.Linear(hidden_dim_q, hidden_dim_K)
        self.W_x = nn.Linear(projected_size, 1, False)
        
        self.hidden_dim_K = hidden_dim_K
        self.hidden_dim_q = hidden_dim_q
        self.projected_size = projected_size
        self.attenType = attenType

    def forward(self, K, q):
        '''
        K [batch_size, num_frames, hidden_dim] // key K, local embeddings
        h [batch_size, hidden_dim] // query q, global embeddings
        return [batch_size, num_frames] : attention score a
        '''
        assert K.size(0) == q.size(0)
        batch_size, num_frames, hidden_dim_K = K.size()
        hidden_dim = q.size(1)
        
        # compute attention score [batch_size, num_frames, projected_size]
        if self.attenType=='additive':
            if hidden_dim != self.projected_size:
                print ('hidden_dim != self.projected_size, use new_additive instead!')
            assert hidden_dim == self.projected_size
            q = q.unsqueeze(1).expand(batch_size, num_frames, hidden_dim)
            x = F.tanh(K + q)
            # project to one dim
            x = F.dropout(self.W_x(x))
        elif self.attenType=='simple':
            x = F.tanh(F.dropout(self.W_K(K)))
            # project to one dim
            x = F.dropout(self.W_x(x))
        elif self.attenType=='dot':
            if hidden_dim != self.projected_size:
                print ('hidden_dim != hidden_dim_K, need to project first!')
                q = F.dropout(self.W_q(q))
                K = F.dropout(self.W_K(K))
            q = q.unsqueeze(1) # [batch_size, 1, hidden_dim]
            K = K.permute(0, 2, 1) # [batch_size, hidden_dim, num_frames] 
            x = torch.bmm(q, K).view(batch_size, num_frames, 1)
        elif self.attenType=='scaled_dot':
            if hidden_dim != self.projected_size:
                print ('hidden_dim != hidden_dim_K, need to project first!')
                q = F.dropout(self.W_q(q))
                K = F.dropout(self.W_K(K))
            q = q.unsqueeze(1) # [batch_size, 1, hidden_dim]
            K = K.permute(0, 2, 1) # [batch_size, hidden_dim, num_frames] 
            x = torch.bmm(q, K).view(batch_size, num_frames, 1) / np.sqrt(hidden_dim_K)
        elif self.attenType=='bilinear':
            q = self.W_qK(q).unsqueeze(1)
            K = K.permute(0, 2, 1) # [batch_size, hidden_dim, num_frames] 
            x = torch.bmm(q, K).view(batch_size, num_frames, 1)
        elif self.attenType=='location_based':
            q = q.unsqueeze(1).expand(batch_size, num_frames, hidden_dim)
            x = F.tanh(F.dropout(self.W_q(q)))
            # project to one dim
            x = F.dropout(self.W_x(x))
        elif self.attenType=='new_additive':
            q = q.unsqueeze(1).expand(batch_size, num_frames, hidden_dim)
            x = F.tanh(F.dropout(self.W_K(K)) + F.dropout(self.W_q(q)))
            # project to one dim
            x = F.dropout(self.W_x(x))
        # fall back to simple attention
        else:
            x = F.tanh(F.dropout(self.W_K(K)))
            # project to one dim
            x = F.dropout(self.W_x(x))
        
        # normalise with softmax
        a = F.softmax(x.squeeze(2), dim=1)

        return a
        
        
def _smallest(matrix, k, only_first_row=False):
    if only_first_row:
        flatten = matrix[:1, :].flatten()
    else:
        flatten = matrix.flatten()
    args = np.argpartition(flatten, k)[:k]
    args = args[np.argsort(flatten[args])]
    return np.unravel_index(args, matrix.shape), flatten[args]



class AttenVisualEncoder(nn.Module):
    def __init__(self, opt):
        super(AttenVisualEncoder, self).__init__()
        # embedding (input) layer options
        self.feat_size = opt.feat_size
        self.embed_dim = opt.word_embed_dim
        # rnn layer options
        self.rnn_type = opt.rnn_type
        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim
        self.dropout = opt.visual_dropout
        self.story_size = opt.story_size
        self.with_position = opt.with_position
        
        self.attenType = opt.attenType
        self.use_self = opt.use_self
        
        # visual embedding layer
        self.vis_emb_global = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                        nn.BatchNorm1d(self.embed_dim),
                                        nn.ReLU(True))
        self.vis_emb_det = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                        nn.BatchNorm1d(self.embed_dim),
                                        nn.ReLU(True))
        
        # visual rnn layer
        self.hin_dropout_layer = nn.Dropout(self.dropout)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                              dropout=self.dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                               dropout=self.dropout, batch_first=True, bidirectional=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.relu = nn.ReLU()

        if self.with_position:
            self.position_embed = nn.Embedding(self.story_size, self.embed_dim)

        self.atten_layer = AttentionLayer(self.embed_dim, self.embed_dim, self.hidden_dim, self.attenType)
        
        print ('BaseModel: self.use_self', self.use_self)
        if self.use_self:
            self.selfAtten_layer = SelfAttentionLayer(self.embed_dim, self.hidden_dim)
 
            
    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return Variable(weight.new(self.num_layers * times, batch_size, dim).zero_())
        else:
            return (Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()),
                    Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()))

    def forward(self, input, hidden=None):
        """
        inputs:
        - feature_fc: (batch_size, 5, feat_size)
        - feature_det: (batch_size, 5, 64, feat_size)
        - hidden 	(num_layers * num_dirs, batch_size, hidden_dim // 2)
        return:
        - out 		(batch_size, 5, rnn_size), serve as context
        """
        feature_fc, feature_det = input
        
        batch_size, seq_length, box_num, feat_size = feature_det.size(0), feature_det.size(1), feature_det.size(2), feature_det.size(3)
        
        feature_fc = feature_fc.view(batch_size*seq_length, feat_size)
        feature_det = feature_det.view(batch_size*seq_length*box_num, feat_size)
        
        feature_img = Variable(torch.ones((batch_size, seq_length, self.embed_dim)), requires_grad=True)

        # visual embeded
        embedding_fc = self.vis_emb_global(feature_fc).contiguous().view(batch_size, seq_length, self.embed_dim)
        embedding_det = self.vis_emb_det(feature_det).contiguous().view(batch_size, seq_length, box_num, self.embed_dim)
        
        # self atten on objects
#         print ('BaseModel: self.use_self', self.use_self)
        if self.use_self:
            embedding_det = embedding_det.view(batch_size, seq_length * box_num, self.embed_dim)
            embedding_det = self.selfAtten_layer(embedding_det, embedding_det, embedding_det)
            embedding_det = embedding_det.contiguous().view(batch_size, seq_length, box_num, self.embed_dim)
            
        img_vec_list = []
        for index in range(seq_length):
            #(batch_size, feat_size)
            emb_fc = embedding_fc[:, index, :]
            #(batch_size, 64, feat_size)
            emb_det = embedding_det[:, index, :, :]
            
            atten_emb = self.atten_layer(emb_det, emb_fc) # (batch_size, 64)
            atten_map = atten_emb.unsqueeze(2).expand(batch_size, box_num, self.embed_dim) #(batch_size, 64, feat_size)
            
            #print('atten_map.shape', atten_map.shape)
            #print('emb_det.shape', emb_det.shape)

            img_vec = torch.sum(emb_det * atten_map, 1).unsqueeze(1)
            #print('img_vec.shape', img_vec.shape)
            
            if index == 0:
                feature_img = img_vec
            else:
                feature_img = torch.cat((feature_img, img_vec), 1)
            #print('feature_img', feature_img)
                    
        emb = feature_img.view(batch_size, seq_length, -1)  # (Na, album_size, embedding_size)
        
        # visual rnn layer
        if hidden is None:
            hidden = self.init_hidden(batch_size, bi=True, dim=self.hidden_dim // 2)
        rnn_input = self.hin_dropout_layer(emb)  # apply dropout
        houts, hidden = self.rnn(rnn_input, hidden)

        # residual layer
        out = emb + self.project_layer(houts)
        out = self.relu(out)  # (batch_size, 5, embed_dim)

        if self.with_position:
            for i in xrange(self.story_size):
                position = Variable(input.data.new(batch_size).long().fill_(i))
                out[:, i, :] = out[:, i, :] + self.position_embed(position)

        return out, hidden

    
    
class VisualEncoder(nn.Module):
    def __init__(self, opt):
        super(VisualEncoder, self).__init__()
        # embedding (input) layer options
        self.feat_size = opt.feat_size
        self.embed_dim = opt.word_embed_dim
        # rnn layer options
        self.rnn_type = opt.rnn_type
        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim
        self.dropout = opt.visual_dropout
        self.story_size = opt.story_size
        self.with_position = opt.with_position

        # visual embedding layer
        self.visual_emb = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                        nn.BatchNorm1d(self.embed_dim),
                                        nn.ReLU(True))
        
        # visual rnn layer
        self.hin_dropout_layer = nn.Dropout(self.dropout)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                              dropout=self.dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                               dropout=self.dropout, batch_first=True, bidirectional=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.relu = nn.ReLU()

        if self.with_position:
            self.position_embed = nn.Embedding(self.story_size, self.embed_dim)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return Variable(weight.new(self.num_layers * times, batch_size, dim).zero_())
        else:
            return (Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()),
                    Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()))

    def forward(self, input, hidden=None):
        """
        inputs:
        - input  	(batch_size, 5, feat_size)
        - hidden 	(num_layers * num_dirs, batch_size, hidden_dim // 2)
        return:
        - out 		(batch_size, 5, rnn_size), serve as context
        """
        
        batch_size, seq_length = input.size(0), input.size(1)

        # visual embeded
        emb = self.visual_emb(input.view(-1, self.feat_size))
        emb = emb.view(batch_size, seq_length, -1)  # (Na, album_size, embedding_size)

        # visual rnn layer
        if hidden is None:
            hidden = self.init_hidden(batch_size, bi=True, dim=self.hidden_dim // 2)
        rnn_input = self.hin_dropout_layer(emb)  # apply dropout
        houts, hidden = self.rnn(rnn_input, hidden)

        # residual layer
        out = emb + self.project_layer(houts)
        out = self.relu(out)  # (batch_size, 5, embed_dim)

        if self.with_position:
            for i in xrange(self.story_size):
                position = Variable(input.data.new(batch_size).long().fill_(i))
                out[:, i, :] = out[:, i, :] + self.position_embed(position)

        return out, hidden

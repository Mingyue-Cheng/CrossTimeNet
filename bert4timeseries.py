from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TStokenizer.model import VQVAE
from transformers import (
    BertForMaskedLM,
    BertConfig,
    BertTokenizer
)
from args import args
import random
import os

    
class bert4ts(nn.Module):
    def __init__(self, task_type):
        super(bert4ts, self).__init__()
        self.data_name = ['ecg-data','eeg-data','har-data']#args.data_path
        self.datasetid = {
            'ecg-data':0,
            'eeg-data':1,
            'har-data':2
        }
        self.emb_layer = nn.ModuleDict({})
        self.vqvae_model = nn.ModuleDict({})
        self.head = nn.ModuleDict({})
        config = BertConfig.from_pretrained(args.local_model_path,output_hidden_states=True)
        self.seq_len =  [5000,3000,128] #args.seq_len
        self.feat_dim = [12,2,9] # args.feat_dim
        self.num_classes = [27,8,6]# args.class_num
        self.d_model = config.hidden_size
        self.dropout = 0.1
        self.n_embd_num = [512, 512 , 512]
        self.wave_length = [40,25,2]   #args.wave_length
        self.patch_num = [self.seq_len[i] // self.wave_length[i] for i in range(len(self.seq_len))]#self.seq_len // self.wave_length
        self.custom_embeddings_dim = [64]*3
        self.task_type = task_type
        self.device = args.device        
        for name,seq_len,feat_dim,custom_embeddings_dim,n_embed,wave_length in zip(self.data_name,self.seq_len,self.feat_dim,self.custom_embeddings_dim,self.n_embd_num,self.wave_length):
            self.vqvae_model[name] = VQVAE(data_shape=(seq_len, feat_dim), hidden_dim=custom_embeddings_dim, n_embed=n_embed , wave_length=wave_length,block_num=4)

        self.n_embed = sum(self.n_embd_num)
        self.parms = args.params
        local_model_path = args.local_model_path #"./bert"
        
        self.mask = nn.Parameter(torch.zeros(config.hidden_size))
        self.n_embed += 1
        self.mask_token = self.n_embed - 1
        nn.init.uniform_(self.mask, -1.0 / self.n_embed, 1.0 / self.n_embed)
        
        if self.parms: #如果有参数
            self.encoder = BertForMaskedLM.from_pretrained(local_model_path,output_attentions=True, output_hidden_states=True)
        else:
            self.encoder = BertForMaskedLM(config) # bert large
            

        if self.task_type != 'word_mapping': # random
            self.encoder.resize_token_embeddings(self.n_embed)
            self.emb_layer = nn.Embedding(self.n_embed, self.d_model)
        else: # word mapping
            weight = self.encoder.get_input_embeddings().weight
            sample = random.sample(list(range(len(weight))),self.n_embed)
            weight = weight[sample]
            self.emb_layer = nn.Embedding(self.n_embed, self.d_model).from_pretrained(weight)
        # Last Layer init
        self.encoder.config.vocab_size = self.n_embed
        new_output = nn.Linear(config.hidden_size, self.n_embed, bias=False)
        self.encoder.set_output_embeddings(new_output)

        for idx,i in enumerate(self.data_name):
            self.head[i] = nn.Sequential(*[nn.GELU(),nn.LayerNorm(self.d_model * self.patch_num[idx]),nn.Linear(self.d_model * self.patch_num[idx], self.num_classes[idx])])
    
    @staticmethod
    def init_weights_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def init_vqvae(self, vqvae_model_path):
        vqvae_state_dict = torch.load(vqvae_model_path, map_location=self.device)
        if 'ecg' in vqvae_model_path:
            name = 'ecg-data'
        elif 'eeg' in vqvae_model_path:
            name = 'eeg-data'
        elif 'har' in vqvae_model_path:
            name = 'har-data'
        self.vqvae_model[name].load_state_dict(vqvae_state_dict)
        self.vqvae_model[name].eval()


         
    def forward(self, x_enc, pretrain=False,dataid=None):
        B, L, M = x_enc.shape
        with torch.no_grad():
            _, _, labels = self.vqvae_model[self.data_name[dataid]](x_enc)
            if dataid > 0:
                offset = sum(self.n_embd1[:dataid])
            else:
                offset = 0
            labels = labels + offset
        


        if pretrain:
            mask = torch.rand(labels.shape,device=labels.device)<args.mask_ratio
            label_mask = labels.clone().detach()
            label_mask[mask] = self.mask_token
            outputs = self.emb_layer(label_mask)
            labels[~mask] = -100
            outputs = self.encoder(inputs_embeds=outputs, labels=labels)


        else:
            outputs = self.emb_layer(labels)
            if args.frozen:
                self.encoder.eval()
                with torch.no_grad():
                    outputs = self.encoder(inputs_embeds=outputs).hidden_states[-1]
            else:
                outputs = self.encoder(inputs_embeds=outputs).hidden_states[-1]
            outputs = self.head[self.data_name[dataid]](outputs.reshape(B,-1)) 

        return outputs
    
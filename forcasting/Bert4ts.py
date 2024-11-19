import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig
import random
from args import args
from ecg_tokenizer.model import VQVAE
from RevIN import RevIN


class Bert4ts(nn.Module):
    def __init__(self):
        super(Bert4ts, self).__init__()
        config = BertConfig.from_pretrained(args.local_model_path, output_hidden_states=True)
        self.d_model = config.hidden_size
        
        self.data_name = ['ETTh1', 'ETTm1','ETTh2', 'ETTm2', 'weather','exchange_rate']
        self.datasetid = {name: idx for idx, name in enumerate(self.data_name)}
        
        # 初始化tokenizer
        self.tokenizer = self._initialize_tokenizer()
        # 初始化掩码token
        self._initialize_mask_token(config)
        self.revin_layer = RevIN(args.feat_dim).to(args.device)
        
        # 初始化BERT模型
        self.model = self._initialize_model(config)
        if args.layers:
            self.model.bert.encoder.layer = nn.ModuleList(self.model.bert.encoder.layer[:3])
            # print("bert = {}".format(self.model))
        # 初始化嵌入层
        if args.rand:
            self.emb_layer = nn.Embedding(self.n_embed, self.d_model).to(args.device)
            self.model.config.vocab_size = self.n_embed
            print('rand')
        else:
            self.emb_layer = self._initialize_embedding_layer()
        # 初始化输出层
        self._initialize_output_layer()
        # print("bert = {}".format(self.model))
        
        self.model.to(args.device)

        # 预训练相关层
        self.pretrain_layers = nn.ModuleDict({
            'model': self.model,
            'emb_layer': self.emb_layer
        })
        

        # 初始化vqvae
        # self.init_vqvae()
        # 初始化预测头
        # self.init_forecasting()

    def _initialize_tokenizer(self):
        if args.shuffle:
            tokenizer = nn.ModuleDict({})
            seq_len = [336, 336, 336,336, 336,336]
            feat_dim = [7, 7,7,7, 21,8]
            wave_length = [7, 7, 7, 7, 7,7]
            self.n_embed = [args.n_embed]*6
            self.patch_num = [seq_len[i] // wave_length[i] for i in range(len(seq_len))]
            custom_embeddings_dim = [64] * 6
            
            for name, seq_len, feat_dim, custom_dim, n_embed, wave_length in zip(
                    self.data_name, seq_len, feat_dim, custom_embeddings_dim, self.n_embed, wave_length):
                tokenizer[name] = VQVAE(
                    data_shape=(seq_len, feat_dim),
                    hidden_dim=custom_dim,
                    n_embed=n_embed,
                    wave_length=wave_length,
                    block_num=2
                ).to(args.device)
            self.n_embed = sum(self.n_embed)
            return tokenizer
        else:
            self.custom_embeddings_dim = 64
            self.n_embed = args.n_embed
            self.patch_num = args.seq_len // args.wave_length
            self.channels = args.feat_dim
            return VQVAE(
                data_shape=(args.seq_len, args.feat_dim),
                hidden_dim=self.custom_embeddings_dim,
                n_embed=self.n_embed,
                wave_length=args.wave_length,
                block_num=2,
            ).to(args.device)

    def _initialize_mask_token(self, config):
        self.mask = nn.Parameter(torch.zeros(config.hidden_size))
        self.n_embed += 1
        self.mask_token = self.n_embed - 1
        nn.init.uniform_(self.mask, -1.0 / self.n_embed, 1.0 / self.n_embed)

    def _initialize_model(self, config):
        if args.params:
            print("params")
            return BertForMaskedLM.from_pretrained(
                args.local_model_path, output_attentions=True, output_hidden_states=True
            )
        else:
            return BertForMaskedLM(config)

    def _initialize_embedding_layer(self):
        weight = self.model.get_input_embeddings().weight
        random.seed(args.seed)
        sample_indices = random.sample(range(len(weight)), self.n_embed + 1)
        weight = weight[sample_indices]
        self.model.config.vocab_size = self.n_embed
        return nn.Embedding(self.n_embed, self.d_model).from_pretrained(weight).to(args.device)

    def _initialize_output_layer(self):
        new_output = nn.Linear(self.d_model, self.n_embed, bias=False).to(args.device)
        self.model.set_output_embeddings(new_output)

    def init_vqvae(self, vqvae_model_path):
        vqvae_state_dict = torch.load(vqvae_model_path, map_location=args.device)
        # 根据模型路径确定数据集名称
        if args.shuffle:
            # name = next((name for name in self.data_name if name in vqvae_model_path), None)
            name = next((name for name in self.data_name if name in vqvae_model_path), None)
            if name:
                self.tokenizer[name].load_state_dict(vqvae_state_dict)
            else:
                print("No matching dataset name found in the path.")
        else:
            self.tokenizer.load_state_dict(vqvae_state_dict)
        self.tokenizer.eval()

    def init_forecasting(self):
        # 初始化预测头
        self.head = nn.ModuleDict() if args.shuffle else None
        self.flatten = nn.Flatten().to(args.device)
        self.forecasting_head = nn.Linear(
            self.patch_num * self.d_model, args.pred_len * args.feat_dim
        ).to(args.device)
        

    @staticmethod
    def init_weights_kaiming(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            m.bias.data.fill_(0.01)

    def forward(self, x_enc, pretrain=False, dataid=None):
        if pretrain:
            outputs = self._pretrain_forward(x_enc, dataid)
        else:
            outputs = self._forecasting_forward(x_enc)
        return outputs

    def _pretrain_forward(self, x_enc, dataid=None):
        # x_enc = self.revin_layer(x_enc, "norm")
        B, L, M = x_enc.shape
        with torch.no_grad():
            if args.shuffle:
                _, _, labels = self.tokenizer[self.data_name[dataid]](x_enc)
                offset = dataid * args.n_embed
                labels = labels + offset
            else:
                _, _, labels = self.tokenizer(x_enc)
        mask = torch.rand(labels.shape, device=labels.device) < args.mask_ratio
        label_mask = labels.clone().detach()
        label_mask[mask] = self.mask_token
        
        outputs = self.emb_layer(label_mask).to(args.device)
        # print(outputs.shape)
        labels[~mask] = -100
        outputs = self.model(inputs_embeds=outputs, labels=labels)
        return outputs

    def _forecasting_forward(self, x_enc, dataid=None):
        if not hasattr(self, 'flatten'):
            self.init_forecasting()
        x_enc = x_enc.to(args.device)
        x_enc = self.revin_layer(x_enc, "norm")
        B, L, M = x_enc.shape
        with torch.no_grad():
            if args.shuffle:
                _, _, labels = self.tokenizer[self.data_name[dataid]](x_enc)
                offset = dataid * args.n_embed
                labels = labels + offset
            else:
                _, _, labels = self.tokenizer(x_enc)
        outputs = self.emb_layer(labels)
        if args.frozen:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(inputs_embeds=outputs).hidden_states[-1]
        else:
            outputs = self.model(inputs_embeds=outputs).hidden_states[-1]
        if args.shuffle:
            outputs = self.head[self.data_name[dataid]](outputs)
        else:
            outputs = self.flatten(outputs)
            outputs = self.forecasting_head(outputs)
        outputs = outputs.view(-1, args.pred_len, args.feat_dim)
        outputs = self.revin_layer(outputs, "denorm")
        return outputs

    def load_pretrained_weights(self, pretrained_model_path):
        if not args.zero:
            pretrained_state_dict = torch.load(pretrained_model_path)
            self.model.load_state_dict(pretrained_state_dict, strict=False)
            print('model loaded.')
        if not args.pretrain:
            self.init_forecasting()

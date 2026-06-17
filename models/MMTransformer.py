import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.positional_encoding import PositionalEncoding

from typing import Any, Dict, Literal, Tuple
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
import numpy as np


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_properties: int,
            d_model: int = 512,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            num_layers: int = 4,
            dropout: float = 0.0,
            activation: str = "gelu",
            norm_first: bool = True,
            bias=True,
            fullrep=False,
    ):
        super().__init__()
        self.activation = {"gelu": nn.GELU(approximate="tanh"), "relu": nn.ReLU()}[activation]

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=self.activation,
                batch_first=True,
                norm_first=norm_first,
                bias=bias
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.num_properties = num_properties
        self.d_model = d_model


    def forward(self, x: torch.FloatTensor, padding_mask: torch.BoolTensor = None):
        """
        Args:       token_ids:      (B, d_model) emedded input properties
                    padding_mask:   (B, L) mask where True means padding // contrast with huggingface token mask
        Returns:    encoded:        (B, L, d_model)
        """
        x = self.transformer(x, src_key_padding_mask=~padding_mask)  # (B, L, d_model)
        return x[:, 0, :].unsqueeze(1)


class MMTransformerAR(nn.Module):
    def __init__(self,
                 tokenizer,
                 vocab_size,
                 latent_dim,
                 d_model,
                 nhead,
                 dim_feedforward,
                 num_layers,
                 dec_layers,
                 activation,
                 bias,
                 norm_first,
                 pad_token_id,
                 dropout=0.0,
                 alpha=2.0,
                 beta=1.0,
                 gamma=0.01,
                 temperature=0.05,
                 num_properties=38,
                 fullrep=True,
                 L2=False,
                 loss_type=None,
                 inverse=False,
                 property=False,
                 deepp=False,
                 attn_pool=False):
        super().__init__()

        # Mdoality specific embedding to indicate the type of each input token
        self.prop_shared_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.token_shared_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_spec_emb = nn.Parameter(torch.randn(1, 1, d_model))

        # Encoder related
        self.L2 = True
        self.loss_type = loss_type.lower()
        self.prop_embedding = nn.Embedding(num_properties, d_model) #nn.Embedding(num_properties, d_model - 1) #
        self.prop_spec_embedding = nn.Embedding(num_properties, d_model)

        self.encoder = TransformerEncoder(vocab_size=num_properties,
                                          num_properties=num_properties,
                                          d_model=d_model,
                                          nhead=nhead,
                                          dim_feedforward=dim_feedforward,
                                          num_layers=num_layers,
                                          dropout=dropout,
                                          activation=activation,
                                          bias=bias,
                                          norm_first=norm_first,
                                          fullrep=fullrep,
                                          )
        self.post_quant_conv = torch.nn.Linear(d_model, latent_dim)

        # Decoder related
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(vocab_size, d_model)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                               nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               activation=activation,
                                               bias=bias,
                                               norm_first=norm_first)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        self.max_sequence_length = 160
        self.word_dropout = 0.0

        self.position_encoding = PositionalEncoding(d_model, max_len=self.max_sequence_length, dropout=self.word_dropout)
        self.log_softmax = nn.LogSoftmax()

        # for high_fidelity
        self.l1_crit = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.pad_idx = pad_token_id
        self.sos_idx = tokenizer.bos_token_id
        self.eos_idx = tokenizer.eos_token_id

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.num_properties = num_properties

        self.outputs2vocab = nn.Linear(d_model, vocab_size)

        self.inverse = inverse
        self.property = property
        self.deepp = deepp

        if self.L2 is not True:
            self.Tanh = nn.Tanh()
            self.temperature = 4.0

        if self.inverse:
            print("Inverse model (no MM)")
        else:
            print("MM model (no Inverse)")
            if self.deepp:
                self.predict_properties = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                                        nn.GELU(),
                                                        nn.Linear(latent_dim, num_properties) )
            else:
                self.predict_properties = torch.nn.Linear(latent_dim, num_properties)
            
        prop_ = torch.cat((torch.ones(1), torch.ones(self.num_properties), torch.zeros(self.max_sequence_length)), dim=0).unsqueeze(0)
        token_ = torch.cat((torch.ones(1), torch.zeros(self.num_properties), torch.ones(self.max_sequence_length)), dim=0).unsqueeze(0)
        joint_ = torch.cat((torch.ones(1), torch.ones(self.num_properties), torch.ones(self.max_sequence_length)), dim=0).unsqueeze(0)

        mask_emb = torch.cat((prop_, token_, joint_), dim=0)
        self.register_buffer("mask_emb", mask_emb, persistent=True)
        self.register_buffer("emb_mask_probs", torch.tensor([1/3, 1/3, 1/3], dtype=torch.float))

        print("PSMILES decoder depth:", dec_layers)
        print("Deep Property decoder:", self.deepp)

        print('Alpha:', self.alpha)
        print('Beta:', self.beta)
        print('Gamma:', self.gamma)
        print('temperature:', self.temperature)


    def encode(self, properties, token_ids, drop_rate=0.0):

        # Property embeddings & their masks
        batch_size = properties.shape[0]

        prop_ids = torch.arange(self.num_properties, device=properties.device).unsqueeze(0)
        prop_emb = self.prop_embedding(prop_ids)
        prop_emb = torch.repeat_interleave(prop_emb, batch_size, dim=0)
        prop_emb = torch.unsqueeze(properties, 2) * prop_emb

        prop_spec_emb = self.prop_spec_embedding(prop_ids)
        prop_emb = prop_emb + prop_spec_emb + self.prop_shared_emb.to(properties.device)

        prop_mask = torch.bernoulli( (1.0 - drop_rate) * torch.ones((batch_size, self.num_properties), device=properties.device)).bool()


        # PSMILES (Tokenized) embeeding & their masks
        token_ids = token_ids.to(properties.device)
        token_embedding = self.embedding(token_ids).permute(1, 0, 2)
        token_embedding = self.position_encoding(token_embedding).permute(1, 0, 2)
        token_embedding = token_embedding + self.token_shared_emb.to(token_ids.device)

        padding_mask = ~self.create_pad_mask(token_ids)
        token_mask = torch.bernoulli( (1.0 - drop_rate) * torch.ones((batch_size, self.max_sequence_length), device=token_ids.device)).bool()
        soeos_mask = (token_ids == self.sos_idx) + (token_ids == self.eos_idx)
        token_mask = padding_mask * token_mask + soeos_mask


        # Combine Property & PSMILES embeddings and feed them to encoder, along with mask
        x = torch.cat([torch.repeat_interleave(self.cls_spec_emb.to(properties.device), batch_size, dim=0), prop_emb, token_embedding], dim=1)
        mask = torch.cat([torch.ones((batch_size, 1), dtype=bool, device=properties.device), prop_mask, token_mask], dim=1).bool()

        z = self.encoder(x, mask)
        z = self.post_quant_conv(z) 

        if self.L2:
            z = F.normalize(z, p=2.0, dim=-1)
        else:
            z = self.Tanh(z)

        return z


    def encode_lambda(self, properties, token_ids, drop_rate=0.0):

        # Property embeddings & their masks
        batch_size = properties.shape[0]

        prop_ids = torch.arange(self.num_properties, device=properties.device).unsqueeze(0)
        prop_emb = self.prop_embedding(prop_ids)
        prop_emb = torch.repeat_interleave(prop_emb, batch_size, dim=0)
        prop_emb = torch.unsqueeze(properties, 2) * prop_emb

        prop_spec_emb = self.prop_spec_embedding(prop_ids)
        prop_emb = prop_emb + prop_spec_emb + self.prop_shared_emb.to(properties.device)

        prop_mask = torch.bernoulli( (1.0 - drop_rate) * torch.ones((batch_size, self.num_properties), device=properties.device)).bool()


        # PSMILES (Tokenized) embeeding & their masks
        token_ids = token_ids.to(properties.device)
        token_embedding = self.embedding(token_ids).permute(1, 0, 2)
        token_embedding = self.position_encoding(token_embedding).permute(1, 0, 2)
        token_embedding = token_embedding + self.token_shared_emb.to(token_ids.device)

        padding_mask = ~self.create_pad_mask(token_ids)
        token_mask = torch.bernoulli( (1.0 - drop_rate) * torch.ones((batch_size, self.max_sequence_length), device=token_ids.device)).bool()
        soeos_mask = (token_ids == self.sos_idx) + (token_ids == self.eos_idx)
        token_mask = padding_mask * token_mask + soeos_mask


        # Combine Property & PSMILES embeddings and feed them to encoder, along with mask
        x = torch.cat([torch.repeat_interleave(self.cls_spec_emb.to(properties.device), batch_size, dim=0), prop_emb, token_embedding], dim=1)
        mask = torch.cat([torch.ones((batch_size, 1), dtype=bool, device=properties.device), prop_mask, token_mask], dim=1)

        mask_emb = self.mask_emb.to(properties.device)
        emb_mask_indices = torch.multinomial(self.emb_mask_probs, batch_size, replacement=True)
        emb_mask = mask_emb[emb_mask_indices].to(properties.device).bool()

        mask = mask * emb_mask

        z = self.encoder(x, mask)
        z = self.post_quant_conv(z) 

        if self.L2:
            z = F.normalize(z, p=2.0, dim=-1)
        else:
            z = self.Tanh(z)

        return z


    def encode_one(self, properties, token_ids, drop_rate=0.0):

        # Property embeddings & their masks
        batch_size = properties.shape[0]

        prop_ids = torch.arange(self.num_properties, device=properties.device).unsqueeze(0)
        prop_emb = self.prop_embedding(prop_ids)
        prop_emb = torch.repeat_interleave(prop_emb, batch_size, dim=0)
        prop_emb = torch.unsqueeze(properties, 2) * prop_emb

        prop_spec_emb = self.prop_spec_embedding(prop_ids)
        prop_emb = prop_emb + prop_spec_emb + self.prop_shared_emb.to(properties.device)

        prop_mask = torch.bernoulli( (1.0 - drop_rate) * torch.ones((batch_size, self.num_properties), device=properties.device)).bool()


        # PSMILES (Tokenized) embeeding & their masks
        token_ids = token_ids.to(properties.device)  
        token_embedding = self.embedding(token_ids).permute(1, 0, 2)
        token_embedding = self.position_encoding(token_embedding).permute(1, 0, 2)
        token_embedding = token_embedding + self.token_shared_emb.to(token_ids.device)

        padding_mask = ~self.create_pad_mask(token_ids)
        token_mask = torch.bernoulli( (1.0 - drop_rate) * torch.ones((batch_size, self.max_sequence_length), device=token_ids.device)).bool()
        soeos_mask = (token_ids == self.sos_idx) + (token_ids == self.eos_idx)
        token_mask = padding_mask * token_mask + soeos_mask


        # Combine Property & PSMILES embeddings and feed them to encoder, along with mask
        x = torch.cat([torch.repeat_interleave(self.cls_spec_emb.to(properties.device), batch_size, dim=0), prop_emb, token_embedding], dim=1)
        mask = torch.cat([torch.ones((batch_size, 1), dtype=bool, device=properties.device), prop_mask, token_mask], dim=1)

        mask_emb = self.mask_emb.to(properties.device)
        emb_mask_indices = torch.tensor(np.random.choice([0, 1], batch_size, replace=True)).to(properties.device)
        emb_mask = mask_emb[emb_mask_indices].to(properties.device).bool()

        mask = mask * emb_mask

        z = self.encoder(x, mask) 
        z = self.post_quant_conv(z)

        if self.L2:
            z = F.normalize(z, p=2.0, dim=-1)
        else:
            z = self.Tanh(z)

        return z
    

    def encode_properties_KS(self, properties, K=29):
        # Property embedding & their masks
        batch_size = properties.shape[0]

        cls_emb = self.cls_spec_emb.to(properties.device).expand(batch_size, -1, -1)
        dummy_emb = torch.zeros((batch_size, self.max_sequence_length, self.d_model), device=properties.device)
        dummy_mask = torch.zeros(batch_size, self.max_sequence_length, dtype=bool, device=properties.device)

        prop_ids = torch.arange(self.num_properties, device=properties.device).unsqueeze(0)
        prop_emb = self.prop_embedding(prop_ids)
        prop_emb = torch.repeat_interleave(prop_emb, batch_size, dim=0)
        prop_emb = torch.unsqueeze(properties, 2) * prop_emb
        prop_spec_emb = self.prop_spec_embedding(prop_ids)

        prop_emb = prop_emb + prop_spec_emb + self.prop_shared_emb.to(properties.device)
        prop_mask = self.make_prop_mask(properties, K)

        x = torch.cat([cls_emb, prop_emb, dummy_emb], dim=1)
        mask = torch.cat([torch.ones((batch_size, 1), dtype=bool, device=properties.device), prop_mask, dummy_mask], dim=1)

        z = self.encoder(x, mask)
        z = self.post_quant_conv(z)

        if self.L2:
            z = F.normalize(z, p=2.0, dim=-1)
        else:
            z = self.Tanh(z)

        return z


    def encode_properties(self, properties, drop_rate=0.0):
        # Property embedding & their masks
        batch_size = properties.shape[0]

        cls_emb = self.cls_spec_emb.to(properties.device).expand(batch_size, -1, -1)
        dummy_emb = torch.zeros((batch_size, self.max_sequence_length, self.d_model), device=properties.device)
        dummy_mask = torch.zeros(batch_size, self.max_sequence_length, dtype=bool, device=properties.device)

        prop_ids = torch.arange(self.num_properties, device=properties.device).unsqueeze(0)
        prop_emb = self.prop_embedding(prop_ids)
        prop_emb = torch.repeat_interleave(prop_emb, batch_size, dim=0)
        prop_emb = torch.unsqueeze(properties, 2) * prop_emb
        prop_spec_emb = self.prop_spec_embedding(prop_ids)

        prop_emb = prop_emb + prop_spec_emb + self.prop_shared_emb.to(properties.device)

        prop_mask = torch.bernoulli( (1.0 - drop_rate) * torch.ones((batch_size, self.num_properties), device=properties.device)).bool()

        x = torch.cat([cls_emb, prop_emb, dummy_emb], dim=1)
        mask = torch.cat([torch.ones((batch_size, 1), dtype=bool, device=properties.device), prop_mask, dummy_mask], dim=1)

        z = self.encoder(x, mask)
        z = self.post_quant_conv(z)

        if self.L2:
            z = F.normalize(z, p=2.0, dim=-1)
        else:
            z = self.Tanh(z)

        return z



    def encode_tokens(self, token_ids, drop_rate=0.0):
        # PSMILES (Tokenized) embeeding & their masks
        batch_size = token_ids.shape[0]

        cls_emb = self.cls_spec_emb.to(token_ids.device).expand(batch_size, -1, -1)
        dummy_emb = torch.zeros((batch_size, self.num_properties, self.d_model), device=token_ids.device)
        dummy_mask = torch.zeros(batch_size, self.num_properties, dtype=bool, device=token_ids.device)

        token_embedding = self.embedding(token_ids).permute(1, 0, 2)
        token_embedding = self.position_encoding(token_embedding).permute(1, 0, 2)
        token_embedding = token_embedding + self.token_shared_emb.to(token_ids.device)

        padding_mask = ~self.create_pad_mask(token_ids)
        token_mask = torch.bernoulli( (1.0 - drop_rate) * torch.ones((batch_size, self.max_sequence_length), device=token_ids.device)).bool()
        soeos_mask = (token_ids == self.sos_idx) + (token_ids == self.eos_idx)
        token_mask = padding_mask * token_mask + soeos_mask

        # Combine Property & PSMILES embeddings and feed them to encoder, along with mask
        x = torch.cat([torch.repeat_interleave(self.cls_spec_emb.to(token_ids.device), batch_size, dim=0), dummy_emb, token_embedding], dim=1)
        mask = torch.cat([torch.ones((batch_size, 1), dtype=bool, device=token_ids.device), dummy_mask, token_mask], dim=1)

        z = self.encoder(x, mask)
        z = self.post_quant_conv(z)

        if self.L2:
            z = F.normalize(z, p=2.0, dim=-1)
        else:
            z = self.Tanh(z)

        return z



    # Memory shape: [L, B, D]
    def batch_decode(self, memory):
        batch_size = memory.size(1)
        device = memory.device

        init_input = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)

        for step in range(self.max_sequence_length):
            # input embedding + position encoding
            tgt_embedding = self.embedding(init_input).permute(1, 0, 2)
            tgt_embedding = self.position_encoding(tgt_embedding) 

            # decoding mask and decoder forward
            tgt_mask = self.get_tgt_mask(init_input.size(1)).to(device)
            padding_mask = self.create_pad_mask(init_input)

            transformer_output = self.decoder(tgt_embedding, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)  
            last_hidden = transformer_output[-1, :, :]  

            logp = F.log_softmax(self.outputs2vocab(last_hidden), dim=-1) 

            next_item = torch.argmax(logp, dim=-1).unsqueeze(1)  

            init_input = torch.cat((init_input, next_item), dim=1)

        inference = [seq[1:] for seq in init_input.tolist()]

        return inference

    # [B, 29]
    def forward(self, properties=None, drop_rate=0.5, K=29, token_ids=None, mode='train'):
        assert mode in ['train', 'infer_psmiles', 'infer_properties'], "Invalid mode. Choose from 'train', 'infer_psmiles', or 'infer_properties'."

        if mode == 'infer_psmiles':
            # zs = self.encode_properties(properties, drop_rate=drop_rate)
            zs = self.encode_properties_KS(properties, K=K)
            memory = zs.permute(1, 0, 2)
            return self.batch_decode(memory)

        elif mode == 'infer_properties':
            zs = self.encode_tokens(token_ids, drop_rate=drop_rate)
            predict_prop = self.predict_properties(zs[:, 0, :])
            return predict_prop
        

        if self.property:
            zs = self.encode_tokens(token_ids=token_ids, drop_rate=drop_rate)

            if 'cwa' in self.loss_type: zf = self.encode_tokens(token_ids=token_ids)
            else: zf = None
            predict_prop = self.predict_properties(zs[:, 0, :])

            return None, predict_prop, zf, zs

        elif self.inverse:
            zs = self.encode_properties(properties, drop_rate=drop_rate)

            if 'cwa' in self.loss_type: zf = self.encode_properties(properties)
            else: zf = None
            predict_prop = None

            memory = zs.permute(1, 0 ,2)

        else:

            if 'cwa' in self.loss_type:
                zf = self.encode_lambda(properties=properties, token_ids=token_ids, drop_rate=drop_rate)
                zs = self.encode(properties=properties, token_ids=token_ids, drop_rate=0.0)
            
            else:
                zf = self.encode_lambda(properties=properties, token_ids=token_ids, drop_rate=drop_rate)
                zs = zf

            predict_prop = self.predict_properties(zf[:, 0, :])

            memory = zf.permute(1, 0 ,2)

        tf_out = self.teacher_forcing(memory, token_ids)
        logits = self.TF_2_logit(tf_out)

        return logits, predict_prop, zf, zs


    def compute_loss_with_logits(self, input_ids, properties, pad_token_id, eos_token_id, logits, predict_prop, zf, zs):
        B, L, V = logits.shape

        # (1) Cross entropy loss
        target = input_ids[:, 1:]
        pred_logits = logits[:, :-1, :]

        ce_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)), 
            target.reshape(-1), 
            ignore_index=pad_token_id,
            reduction="mean"
        )

        # (2) MSE loss
        if self.inverse:
            mse_loss = torch.tensor(0.0).cuda()
        else:
            mse_loss = torch.mean(torch.sum((properties - predict_prop)**2, dim=-1), dim=0)

        # (3) CwA loss
        if "cwa" in self.loss_type:
            zf = zf[:, 0, :]  
            zs = zs[:, 0, :]   

            # If representation is L2 normalized, similarity = dot product.
            # If not, similarity = the negative l2 distance .
            if self.L2:
                logits = torch.matmul(zs, zf.T)
            else:
                zf_tiled = torch.stack([zf] * B, dim=0)
                zs_tiled = torch.stack([zs] * B, dim=1)
                logits = - torch.sum((zf_tiled - zs_tiled) ** 2, axis=-1)

            logits = (1.0 / self.temperature) * logits
            
            labels = torch.arange(B, device=zs.device).long()

            contrast_loss = self.cross_entropy_loss(logits, labels)
        else:
            contrast_loss = torch.tensor(0.).cuda()

        eos_mask = (target == eos_token_id)  
        eos_pos = eos_mask.float().argmax(dim=1)  
        eos_logits = pred_logits[torch.arange(target.size(0)), eos_pos] 
        eos_loss = F.cross_entropy(eos_logits, eos_token_id * torch.ones_like(eos_pos))

        loss = ce_loss + self.alpha * mse_loss + self.beta * contrast_loss + self.gamma * eos_loss

        return loss, ce_loss, mse_loss, contrast_loss, eos_loss

    def teacher_forcing(self, memory, input_idx):
        input_idx = input_idx.to(memory.device)  
        naive_embedding = self.embedding(input_idx).permute(1, 0, 2)
        input_embedding = self.position_encoding(naive_embedding)
        ipt_mask = self.get_tgt_mask(self.max_sequence_length).to(memory.device)
        padding_mask = self.create_pad_mask(input_idx)

        transformer_output = self.decoder(input_embedding, memory, tgt_mask=ipt_mask, tgt_key_padding_mask=padding_mask)
        return transformer_output

    def TF_2_logit(self, transformer_output):
        transformer_output_ = transformer_output.permute(1, 0, 2)
        logit = self.outputs2vocab(transformer_output_.reshape(-1, self.d_model))
        logit = logit.reshape(-1, self.max_sequence_length, self.vocab_size)  #
        return logit

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1)  
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) 
        mask = mask.masked_fill(mask == 1, float(0.0)) 
        return mask
    
    def make_prop_mask(self, properties, K):
        device = properties.device
        scores = torch.rand(properties.shape, device=device)
        idx = scores.topk(K, dim=1).indices 
        mask = torch.zeros(properties.shape, device=device, dtype=torch.bool)
        mask.scatter_(1, idx, True)    

        return mask

    def create_pad_mask(self, matrix: torch.tensor) -> torch.tensor:
        return (matrix == self.pad_idx)


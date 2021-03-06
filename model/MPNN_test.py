from model.Ttransformer.positional_encoding import *


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, X):
        # X: [batch, window, n_multiv]
        X = self.linear(X)  # X: [batch, n_multiv, 1]
        return X


class test_model(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int
                 , num_nodes: int, window: int, num_layers, dropout: float, dec_seq_len, batch_size):
        super(test_model, self).__init__()
        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.conv1 = GCNConv(self.in_channels, self.hidden_size)
        self.conv2 = GCNConv(self.hidden_size, self.hidden_size)

        self.bn1 = nn.BatchNorm2d(self.hidden_size)
        self.bn2 = nn.BatchNorm2d(self.hidden_size)

        self.src_mask = None
        feature_size = 256
        # d_model 词嵌入维度

        self.d_model = feature_size
        self.dropout = dropout
        self.maX_len = 100

        self.local = context_embedding(self.d_model, self.d_model, 1)

        # ----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(self.d_model)

        # ----------whether transformer encoder layer with batch_norm---------#
        self.input_project = nn.Linear(self.hidden_size, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=dec_seq_len)

        self.tmp_out = nn.Linear(256, 1)
        self.src_key_padding_mask = None
        self.transformer = nn.Transformer(d_model=self.d_model,
                                          nhead=8,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=dec_seq_len,
                                          dropout=dropout, )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tmp_out2 = nn.Linear(256, 12)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, X, edge_indeX, edge_attr):
        lst = []  # X [256bz 12node 11feature 64timestep]
        X = X.permute(0, 3, 1, 2)  # X [256bz  64timestep 12node  11feature ]
        X = self.relu(self.conv1(X, edge_indeX, edge_attr))
        X = X.permute(0, 3, 2, 1)
        X = self.bn1(X)
        X = X.permute(0, 3, 2, 1)
        X = self.dropout(X)
        lst.append(X)

        X = self.relu(self.conv2(X, edge_indeX, edge_attr))
        X = X.permute(0, 3, 2, 1)
        X = self.bn2(X)
        X = X.permute(0, 3, 2, 1)
        X = self.dropout(X)
        lst.append(X)
        # X [256 64ts 12node 32fe]
        X = torch.transpose(X, 0, 1)
        X = X.contiguous().view(self.window, -1, X.shape[-1])  # [64 256*12 32] [64 3072 32 ]
        src = X
        # src = src.permute(1, 0, 2)  # [time_steps, batch_size,feature_dim] [12 192 32]
        tgt = src[-3:, :, :]  # src [64seqlen 3072bz 32feature]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化掩码张量
        mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)
        # src [12 192 32]
        src = self.input_project(src) * math.sqrt(self.d_model)  # [12ts 192bz 256fd]
        tgt = self.input_project(tgt) * math.sqrt(self.d_model)
        src = self.local(src.permute(1, 2, 0))  # [192 256 12]
        src = src.permute(2, 0, 1)  # [12(seq_len) 192(bz) 256]
        # torch.Size([168, 16, 256])->torch.Size([168, 16, pos_encoder.featuresize(256)])
        src = self.pos_encoder(src)  # torch.Size([168,16,64 ])
        tgt = self.pos_encoder(tgt)  # torch,Size(3 16 64)

        X = self.transformer(src=src,
                             tgt=tgt,
                             tgt_mask=mask)
        # X = X.view(-1, self.batch_size, 256)
        # X = torch.transpose(X, 0, 1).contiguous().view(self.batch_size, -1)
        X = X.view(3, self.batch_size, self.num_nodes, -1)
        X = X.transpose(0, 1)

        # X[seqlen bz features]
        transformer_out = self.tmp_out(X)
        transformer_out = transformer_out[:, -1, :, :]
        # transformer_out = self.relu(transformer_out)
        # transformer_out = self.tmp_out2(transformer_out)
        return transformer_out


class test_model2(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int
                 , num_nodes: int, window: int, num_layers, dropout: float, dec_seq_len, batch_size):
        super(test_model2, self).__init__()
        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size

        self.src_mask = None

        self.conv1 = GCNConv(self.in_channels, self.hidden_size)
        self.conv2 = GCNConv(self.hidden_size, self.hidden_size)

        self.bn1 = nn.BatchNorm2d(self.hidden_size)
        self.bn2 = nn.BatchNorm2d(self.hidden_size)

        self.src_mask = None
        feature_size = 256
        # d_model 词嵌入维度

        self.d_model = feature_size
        self.dropout = dropout
        self.maX_len = 100

        self.local = context_embedding(self.d_model, self.d_model, 1)

        # ----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(self.d_model)

        # ----------whether transformer encoder layer with batch_norm---------#
        self.input_project = nn.Linear(self.hidden_size, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=dec_seq_len)
        self.decoder = nn.Linear(feature_size, 1)
        self.tmp_out = nn.Linear(196608, 256)
        self.src_key_padding_mask = None
        self.transformer = nn.Transformer(d_model=self.d_model,
                                          nhead=8,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=dec_seq_len,
                                          dropout=dropout, )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tmp_out2 = nn.Linear(256, 12)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, X, edge_indeX, edge_attr):
        lst = []  # X [256bz 12node 11feature 64timestep]
        X = X.permute(0, 3, 1, 2)  # X [256bz  64timestep 12node  11feature ]
        X = self.relu(self.conv1(X, edge_indeX, edge_attr))
        X = X.permute(0, 3, 2, 1)
        X = self.bn1(X)
        X = X.permute(0, 3, 2, 1)
        X = self.dropout(X)
        lst.append(X)

        X = self.relu(self.conv2(X, edge_indeX, edge_attr))
        X = X.permute(0, 3, 2, 1)
        X = self.bn2(X)
        X = X.permute(0, 3, 2, 1)
        X = self.dropout(X)
        lst.append(X)
        # X [256 64ts 12node 32fe]
        X = torch.transpose(X, 0, 1)
        X = X.contiguous().view(self.window, -1, X.shape[-1])  # [64 256*12 32] [64 3072 32 ]
        src = X
        # src = src.permute(1, 0, 2)  # [time_steps, batch_size,feature_dim] [12 192 32]
        tgt = src[-3:, :, :]  # src [64seqlen 3072bz 32feature]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化掩码张量
        mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        # src [12 192 32]
        src = self.input_project(src) * math.sqrt(self.d_model)  # [12ts 192bz 256fd]
        tgt = self.input_project(tgt) * math.sqrt(self.d_model)
        src = self.local(src.permute(1, 2, 0))  # [192 256 12]
        src = src.permute(2, 0, 1)  # [12(seq_len) 192(bz) 256]
        # torch.Size([168, 16, 256])->torch.Size([168, 16, pos_encoder.featuresize(256)])
        src = self.pos_encoder(src)  # torch.Size([168,16,64 ])
        tgt = self.pos_encoder(tgt)  # torch,Size(3 16 64)
        X = self.transformer_encoder(src=src, mask=mask)
        X = self.transformer(src=src,
                             tgt=tgt,
                             tgt_mask=mask)
        X = X.view(-1, self.batch_size, 256)

        X = torch.transpose(X, 0, 1).contiguous().view(self.batch_size, -1)
        transformer_out = self.tmp_out(X)
        transformer_out = self.relu(transformer_out)
        transformer_out = self.tmp_out2(transformer_out)
        return transformer_out


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=32, periods=periods,
                            batch_size=batch_size)  # node_features=2, periods=12
        # Equals single-shot prediction

        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        # x.shape = (batch_size,node_num,feature_size,time_steps)
        h = self.tgnn(x, edge_index, edge_attr)  # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h)
        h = self.linear(h)
        return h

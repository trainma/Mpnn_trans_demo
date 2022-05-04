from torch_geometric.nn import GCNConv

from model.Ttransformer.positional_encoding import *


class AR(nn.Module):
    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, X):
        # X: [batch, window, n_multiv]
        X = self.linear(X)  # X: [batch, n_multiv, 1]
        return X


##
class CustomGraphConv(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, dropout: float):
        super(CustomGraphConv, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels

        self.conv1 = GCNConv(self.in_channels, self.hidden_size)
        self.conv2 = GCNConv(self.hidden_size, self.hidden_size)
        self.bn1 = nn.BatchNorm2d(self.hidden_size)
        self.bn2 = nn.BatchNorm2d(self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, X, edge_indeX, edge_attr):
        lst = []  # X [256bz 12node 11feature 64timestep]
        X = X.permute(0, 3, 1, 2)  # X [256bz  64timestep 12node  11feature ]
        skip = X.clone()
        X = self.relu(self.conv1(X, edge_indeX, edge_attr))
        X = X.permute(0, 3, 2, 1)
        X = self.bn1(X)
        X = X.permute(0, 3, 2, 1)
        X = self.dropout(X)
        lst.append(X)  # X.shape[16 64 12 64]

        X = self.relu(self.conv2(X, edge_indeX, edge_attr))
        X = X.permute(0, 3, 2, 1)
        X = self.bn2(X)
        X = X.permute(0, 3, 2, 1)
        X = self.dropout(X)  # X.shape[16 64 12 64]
        lst.append(X)
        out = torch.cat(lst, dim=-1)

        return X, out, skip


class Graph_transformer(nn.Module):
    def __init__(self, hidden_size: int, d_model: int, feature_dim: int, label_len: int
                 , num_nodes: int, window: int, num_layers, dropout: float, dec_seq_len, batch_size):
        super(Graph_transformer, self).__init__()
        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.label_len = label_len
        self.src_mask = None
        self.d_model = d_model
        self.maX_len = 100

        self.local = context_embedding(self.d_model, self.d_model, 1)
        self.pos_encoder = PositionalEncoding(self.d_model)

        self.input_project = nn.Linear(self.hidden_size + self.feature_dim, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=dec_seq_len)

        self.tmp_out = nn.Linear(256, 1)
        self.src_key_padding_mask = None
        self.transformer = nn.Transformer(d_model=self.d_model,
                                          nhead=8,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=dec_seq_len,
                                          dropout=dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        tgt = src[-self.label_len:, :, :]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)
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
        X = X.view(self.label_len, self.batch_size, self.num_nodes, -1)
        X = X.transpose(0, 1)

        # transformer_out = self.tmp_out(X)
        # transformer_out = transformer_out[:, -1, :, :]
        # transformer_out = self.tmp_out(X)
        transformer_out = X[:, :8, :, :]
        return transformer_out


class Multi_graph_trans(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int, d_model: int, feature_dim: int,
                 label_len: int
                 , num_nodes: int, window: int, num_layers, dropout: float, dec_seq_len, batch_size):
        super(Multi_graph_trans, self).__init__()

        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.d_model = d_model
        self.geo_conv = CustomGraphConv(in_channels=self.in_channels, hidden_size=self.hidden_size,
                                        dropout=self.dropout)
        self.poi_conv = CustomGraphConv(in_channels=self.in_channels, hidden_size=self.hidden_size,
                                        dropout=self.dropout)
        self.ST_conv = CustomGraphConv(in_channels=self.in_channels, hidden_size=self.hidden_size,
                                       dropout=self.dropout)

        self.geo_trans = Graph_transformer(hidden_size=self.hidden_size, d_model=d_model, num_nodes=self.num_nodes,
                                           feature_dim=feature_dim, label_len=label_len,
                                           window=self.window, num_layers=num_layers, dropout=self.dropout,
                                           dec_seq_len=dec_seq_len, batch_size=self.batch_size)
        self.poi_trans = Graph_transformer(hidden_size=self.hidden_size, d_model=d_model, num_nodes=self.num_nodes,
                                           feature_dim=feature_dim, label_len=label_len,
                                           window=self.window, num_layers=num_layers, dropout=self.dropout,
                                           dec_seq_len=dec_seq_len, batch_size=self.batch_size)
        self.ST_trans = Graph_transformer(hidden_size=self.hidden_size, d_model=d_model, num_nodes=self.num_nodes,
                                          feature_dim=feature_dim, label_len=label_len,
                                          window=self.window, num_layers=num_layers, dropout=self.dropout,
                                          dec_seq_len=dec_seq_len, batch_size=self.batch_size)
        self.linear = nn.Linear(self.d_model * 3, 1)

    def forward(self, X, geo_graph_edge_index, geo_graph_edge_attr,
                poi_graph_edge_index, poi_graph_edge_attr,
                ST_graph_edge_index, ST_graph_edge_attr):
        X1 = self.geo_conv(X, geo_graph_edge_index, geo_graph_edge_attr)[0]
        X2 = self.geo_conv(X, poi_graph_edge_index, poi_graph_edge_attr)[0]
        X3 = self.geo_conv(X, ST_graph_edge_index, ST_graph_edge_attr)[0]

        skip1 = self.geo_conv(X, geo_graph_edge_index, geo_graph_edge_attr)[2]
        skip2 = self.geo_conv(X, poi_graph_edge_index, poi_graph_edge_attr)[2]
        skip3 = self.geo_conv(X, ST_graph_edge_index, ST_graph_edge_attr)[2]

        X1 = X1.transpose(0, 1).contiguous().view(self.window, -1, X1.shape[-1])
        X2 = X2.transpose(0, 1).contiguous().view(self.window, -1, X2.shape[-1])
        X3 = X3.transpose(0, 1).contiguous().view(self.window, -1, X3.shape[-1])

        skip1 = skip1.transpose(0, 1).contiguous().view(self.window, -1, skip1.shape[-1])
        skip2 = skip2.transpose(0, 1).contiguous().view(self.window, -1, skip2.shape[-1])
        skip3 = skip3.transpose(0, 1).contiguous().view(self.window, -1, skip3.shape[-1])

        X1 = torch.cat((X1, skip1), dim=-1)
        X2 = torch.cat((X2, skip2), dim=-1)
        X3 = torch.cat((X3, skip3), dim=-1)

        Y1 = self.geo_trans(X1)
        Y2 = self.geo_trans(X2)
        Y3 = self.geo_trans(X3)

        Y = torch.cat((Y1, Y2, Y3), dim=-1)
        out = self.linear(Y)
        return out


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
        tgt = src[-24:, :, :]  # src [64seqlen 3072bz 32feature]
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
        X = X.view(3, self.batch_size, self.num_nodes, -1)
        X = X.transpose(0, 1)

        # X[seqlen bz features]
        transformer_out = self.tmp_out(X)
        transformer_out = transformer_out[:, -8:, :, :]
        return transformer_out

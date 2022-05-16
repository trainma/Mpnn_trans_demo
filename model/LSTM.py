import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, nfeat, nhid, n_nodes, window, dropout, batch_size, recur):
        super().__init__()
        self.nhid = nhid
        self.n_nodes = n_nodes
        self.nout = n_nodes
        self.window = window
        self.nb_layers = 2

        self.nfeat = nfeat
        self.recur = recur
        self.batch_size = batch_size
        self.lstm = nn.LSTM(nfeat, self.nhid, num_layers=self.nb_layers)

        self.linear = nn.Linear(nhid, self.nout)
        self.cell = (nn.Parameter(nn.init.xavier_uniform(
            torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),
            requires_grad=True))

        # self.hidden_cell = (torch.zeros(2,self.batch_size,self.nhid).to(device),torch.zeros(2,self.batch_size,self.nhid).to(device))
        # nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))

    def forward(self, adj, features):
        # adj is 0 here
        # print(features.shape)
        features = features.view(self.window, -1, self.n_nodes)  # .view(-1, self.window, self.n_nodes, self.nfeat)
        # print(features.shape)
        # print("----")

        # ------------------
        if (self.recur):
            # print(features.shape)
            # self.hidden_cell =
            try:
                lstm_out, (hc, self.cell) = self.lstm(features, (
                    torch.zeros(self.nb_layers, self.batch_size, self.nhid).cuda(), self.cell))
                # = (hc,cn)
            except:
                # hc = self.hidden_cell[0][:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
                hc = torch.zeros(self.nb_layers, features.shape[1], self.nhid).cuda()
                cn = self.cell[:, 0:features.shape[1], :].contiguous().view(2, features.shape[1], self.nhid)
                lstm_out, (hc, cn) = self.lstm(features, (hc, cn))
        else:
            # ------------------
            lstm_out, (hc, cn) = self.lstm(features)  # , self.hidden_cell)#self.hidden_cell

        predictions = self.linear(lstm_out)  # .view(self.window,-1,self.n_nodes)#.view(self.batch_size,self.nhid))#)
        # print(predictions.shape)
        return predictions[-1].view(-1)

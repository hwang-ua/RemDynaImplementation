import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
import os
import time
import numpy
from utils.distance_matrix_func import one_hot_feature

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ANN(nn.Module):
    def __init__(self, num_input, num_node, num_output):
        super(ANN, self).__init__()

        self.layers = []
        self.layers.append(nn.Linear(num_input, num_node[0]))
        nn.init.kaiming_normal_(self.layers[-1].weight)
        self.layers.append(nn.ReLU(True))

        for i in range(len(num_node) - 1):
            self.layers.append(nn.Linear(num_node[i], num_node[i+1]))
            nn.init.kaiming_normal_(self.layers[-1].weight)

            self.layers.append(nn.ReLU(True))

        self.layers.append(nn.Linear(num_node[-1], num_output))
        nn.init.normal_(self.layers[-1].weight, std=0.01)

        self.nnet = nn.Sequential(*self.layers)

    def forward(self, x):
        v = self.nnet(x)
        return v

    def get_value(self, x):
        with torch.no_grad():
            eval = self.forward(x)
        return eval

    def print_net(self):
        print("=========== NN structure ==========")
        print(self.nnet)
        print("===================================")

class RecvState():
    def __init__(self, num_input, num_node, num_output, learning_rate, weight_decay):

        self.net = ANN(num_input, num_node, num_output)
        self.net = self.net.to(device)
        self.criterion = self.mse_loss

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        print(self.net.print_net())

    def mse_loss(self, y_hat, y):
        ls = ((y_hat - y) ** 2).sum(dim=1).mean()
        return ls

    def training(self, rep, s, num_epochs, batch_size=1, print_loss = True):

        self.net.print_net()
        len_rep = rep.shape[1]
        data = numpy.concatenate((rep, s), axis=1)
        data = torch.from_numpy(data).float().to(device)
        data = Data.TensorDataset(data)
        loader = Data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        loss_change = []
        for epoch in range(num_epochs):
            allloss = []
            start_time = time.time()
            for step, [batch_x] in enumerate(loader):
                # ===================forward=====================
                output = self.net(batch_x[:, :len_rep])
                loss = self.criterion(output, batch_x[:, len_rep:])
                # print(batch_x[:50, len_rep:])
                # exit()
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                allloss.append(loss.data.item())

                del step
                del batch_x
                del output
                del loss

            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}, time:{:.4f}'
                  .format(epoch + 1, num_epochs, numpy.mean(numpy.array(allloss)), time.time() - start_time))
            loss_change.append(numpy.mean(numpy.array(allloss)))

            del allloss

        return numpy.array(loss_change)



    def test(self, rep, s):
        x = torch.from_numpy(rep).float().to(device)
        s = torch.from_numpy(s).float().to(device)
        output = self.net.get_value(x)
        loss = self.criterion(output, s)
        return output.detach().cpu().numpy(), loss

    def test2(self, rep):
        with torch.no_grad():
            x = torch.from_numpy(rep).float().to(device)
            output = self.net.get_value(x)
        return output.detach().cpu().numpy()

    def saving(self, path, file_name):
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
            print("make new dir", path)
        while os.path.exists(path+file_name):
            file_name += "-"
        torch.save(self.net.state_dict(), path+file_name+".pth")
        print("model saved in:", path+file_name+".pth")
        return

    def loading(self, path, file_name):
        print("*** encoder: load saved model ***")
        m = torch.load(path + file_name + ".pth", map_location='cpu')
        for k in m.keys():
            print(k, m[k].size())
        print()

        self.net.load_state_dict(torch.load(path + file_name + ".pth", map_location='cpu'))
        self.net.to(device)
        return

    def print_nn(self):
        self.net.print_net()


"""
Public function
"""

def reduce(func, seq):
    first = seq[0]
    for i in seq[1:]:
        first = func(first, i)
    return first

def op_mul(a, b):
    return a*b

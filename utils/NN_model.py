import torch
from torch import nn
import torch.utils.data as Data
import time
import numpy as np
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class NN(nn.Module):
    def __init__(self, num_input, num_node, num_output, gate="Relu"):
        super(NN, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(num_input, num_node[0]))
        # nn.init.xavier_normal_(self.layers[-1].weight)
        nn.init.kaiming_normal_(self.layers[-1].weight)
        if gate == "Relu":
            self.layers.append(nn.ReLU(True))
        elif gate == "Selu":
            self.layers.append(nn.SELU(True))

        for i in range(len(num_node) - 1):
            self.layers.append(nn.Linear(num_node[i], num_node[i+1]))
            # nn.init.xavier_normal_(self.layers[-1].weight)
            nn.init.kaiming_normal_(self.layers[-1].weight)

            if gate == "Relu":
                self.layers.append(nn.ReLU(True))
            elif gate == "Selu":
                self.layers.append(nn.SELU(True))

        self.layers.append(nn.Linear(num_node[-1], num_output))
        nn.init.normal_(self.layers[-1].weight, std=0.01)
        # self.layers.append(nn.Sigmoid())

        self.nnet = nn.Sequential(*self.layers)

    def forward(self, x):
        v = self.nnet(x)
        return v

    def test(self, x):
        with torch.no_grad():
            eval = self.nnet(x)
        return eval


class NNModel:
    def __init__(self, num_input, num_node, num_output, learning_rate, gate="relu"):
        self.net = NN(num_input, num_node, num_output, gate=gate)
        self.net = self.net.to(device)
        # self.opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.opt = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def training(self, x, len_x, num_epochs, batch_size):

        self.net.print_net()

        data_tensor = torch.from_numpy(x).float().to(device)
        data = Data.TensorDataset(data_tensor)
        loader = Data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        all_epoch_loss = []
        for epoch in range(num_epochs):
            all_loss = []

            start_time = time.time()
            for step, [batch_x] in enumerate(loader):
                y_hat = self.net(batch_x[:, :len_x])
                self.opt.zero_grad()

                training_loss = self.loss(y_hat, batch_x[:, len_x:])
                training_loss.backward()
                self.opt.step()

                all_loss.append(training_loss.data.item())

            print('epoch [{}/{}], loss:{:.4f}, time:{:.4f}'
                      .format(epoch + 1, num_epochs, np.mean(np.array(all_loss)), time.time() - start_time))
            all_epoch_loss.append(np.mean(np.array(all_loss)))

        return np.array(all_epoch_loss)

    def test(self, x, len_x):
        x = torch.from_numpy(x).float().to(device)
        y_hat = self.net.test(x[:, :len_x])
        test_loss = self.loss(y_hat, x[:, len_x:])
        return y_hat.detach().cpu().numpy(), test_loss.data.item()

    def predict(self, x):
        x = torch.from_numpy(x).float().to(device)
        y_hat = self.net.test(x)
        return y_hat.detach().cpu().numpy()

    # save model
    def saving(self, path, file_name):
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
            print("make new dir", path)
        while os.path.exists(path + file_name):
            file_name += "-"
        torch.save(self.net.state_dict(), path + file_name + ".pth")
        print("model saved in:", path + file_name + ".pth")
        return

    # load model
    def loading(self, path, file_name):
        m = torch.load(path + file_name + ".pth", map_location='cpu')
        print("===== loading =====")
        for k in m.keys():
            print(k, m[k].size())
        print()
        self.net.load_state_dict(torch.load(path + file_name + ".pth", map_location='cpu'))
        self.net.to(device)
        return

    def print_net(self):
        print("=========== NN structure ==========")
        print(self.net)
        print("===================================")

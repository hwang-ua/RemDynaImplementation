import torch
from torch import nn
import torch.utils.data as Data
import os
import time
import numpy
from utils.distance_matrix_func import *

class AutoEncoder(nn.Module):
    def __init__(self, num_input, num_node, num_feature, num_output, num_dec_node='none', dropout=0):
        super(AutoEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.en_layers = []

        # Input -> Representation

        if num_node != []:
            self.en_layers.append(nn.Linear(num_input, num_node[0]))
            nn.init.kaiming_normal_(self.en_layers[0].weight)
            self.en_layers.append(nn.ReLU(True))

            for i in range(len(num_node) - 1):
                self.en_layers.append(nn.Linear(num_node[i], num_node[i + 1]))
                nn.init.kaiming_normal_(self.en_layers[-1].weight)
                self.en_layers.append(nn.ReLU(True))
            self.en_layers.append(self.dropout)
            self.en_layers.append(nn.Linear(num_node[-1], num_feature))
            nn.init.normal_(self.en_layers[-1].weight, std=0.01)

        else:
            self.en_layers.append(nn.Linear(num_input, num_feature))
            nn.init.kaiming_normal_(self.en_layers[0].weight)
            self.en_layers.append(nn.ReLU(True))

        self.encoder = nn.Sequential(*self.en_layers)

        # Representation -> Predecessor Feature

        if num_dec_node == 'none':
            num_dn = numpy.flip(num_node)
        else:
            num_dn = num_dec_node

        self.de_layers = []

        if num_dn != []:

            self.de_layers.append(nn.Linear(num_feature, num_dn[0]))
            nn.init.kaiming_normal_(self.de_layers[-1].weight)

            self.de_layers.append(nn.ReLU(True))

            for j in range(len(num_dn) - 1):
                self.de_layers.append(nn.Linear(num_dn[j], num_dn[j + 1]))
                nn.init.kaiming_normal_(self.de_layers[-1].weight)

                self.de_layers.append(nn.ReLU(True))

            self.de_layers.append(nn.Linear(num_dn[-1], num_output))
            nn.init.kaiming_normal_(self.de_layers[-1].weight)

        else:
            print("Successor Feature: ELSE block")
            self.de_layers.append(nn.Linear(num_feature, num_output))
            nn.init.normal_(self.de_layers[-1].weight, std=0.01)

        self.decoder = nn.Sequential(*self.de_layers)

    def forward(self, x, decoder = False):
        r = self.encoder(x)
        if decoder:
            return r, self.decoder(r)
        return r, None

    def get_feature(self, x):
        return self.encoder(x)

    def test(self, x):
        # with torch.no_grad():
        #     phi = self.encoder(x)
        #     rec = self.decoder(phi)
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            phi = self.encoder(x)
            rec = self.decoder(phi)
        return phi, rec

    def print_net(self):
        print("=========== NN structure ==========")
        print(self.encoder)
        print(self.decoder)
        print("===================================")


class AETraining():
    def __init__(self, num_input, num_node, num_feature, num_output, learning_rate, learning_rate_rcvs,
                 weight_decay=0, num_dn=None, num_rec_node=[2], optimizer="adam", dropout=0, beta=0, delta=1,
                 legal = False, continuous=False, num_tiling=None, num_tile=None, constraint=True):

        self.net = AutoEncoder(num_input, num_node, num_feature, num_output, num_dec_node=num_dn, dropout=dropout)
        self.beta = beta
        self.delta = delta
        self.rep_crit_mse = self.mse_loss # nn.MSELoss() #
        self.rep_crit_cst = self.cst_loss
        self.rcvs_crit = nn.MSELoss() # self.loss #
        print("Beta:", self.beta)
        print("Delta:", self.delta)
        print("Learning rate", learning_rate, learning_rate_rcvs)

        if optimizer == "adam":
            self.rep_opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "AMSGrad":
            self.rep_opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                            amsgrad=True)

        self.legal = legal
        self.continuous = continuous
        self.num_input = num_input
        if self.continuous:
            self.tc = TileCoding(1, num_tiling, num_tile)
            self.num_tiling = num_tiling
            self.num_tile = num_tile

        self.constraint = constraint
        print("Use constraint:", self.constraint)

    def mse_loss(self, y_hat, y):
        # ls = ((y_hat - y) ** 2).sum(dim=1).mean()
        ls = ((y_hat - y) ** 2).mean()
        #print("mse", ls)
        return ls

    def cst_loss(self, rnd1, rnd2):
        ls = self.beta * (((rnd1 * rnd2).sum(dim=1) ** 2).mean()
                          - self.delta * (rnd1 ** 2).mean() - self.delta * (rnd2 ** 2).mean() + self.delta**2*32)
        # ls = self.beta * (((rnd1 * rnd2).sum(dim=1) ** 2).mean()
        #                   - self.delta * (rnd1 ** 2).sum(dim=1).mean() - self.delta * (rnd2 ** 2).sum(dim=1).mean() + self.delta ** 2 * 32)
        # print("cst", ((rnd1 * rnd2).sum(dim=1) ** 2).mean(), (rnd1 ** 2).sum(dim=1).mean(), (rnd2 ** 2).sum(dim=1).mean())
        return ls

    def training(self, x, len_x, num_epochs, num_epochs_rcvs, batch_size):
        self.net.print_net()

        data_tensor = torch.from_numpy(x).float()
        data = Data.TensorDataset(data_tensor)
        loader = Data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        rep_loss_change = []
        rcvs_loss_change = []

        for epoch in range(max(num_epochs, num_epochs_rcvs)):
            rep_allloss = []
            rcvs_allloss = []

            start_time = time.time()
            for step, [batch_x] in enumerate(loader):
                if self.constraint:
                    if self.legal:
                        rnd_idx = numpy.random.randint(0, len(x), size=len(batch_x))
                        rnd = data_tensor[rnd_idx, :len_x]

                    elif self.continuous and not self.legal:

                        # rnd_x = numpy.random.random(size=len(batch_x)).reshape((-1, 1))
                        # rnd_y = numpy.random.random(size=len(batch_x)).reshape((-1, 1))
                        rnd_x = numpy.random.random(size=len(batch_x) * (len(batch_x) - 1)).reshape((-1, 1))
                        rnd_y = numpy.random.random(size=len(batch_x) * (len(batch_x) - 1)).reshape((-1, 1))
                        states = numpy.concatenate((rnd_x, rnd_y), axis=1)

                        if self.num_input == 2:
                            rnd = states
                        else:
                            rnd = []
                            for t in states:
                                xidx = self.tc.get_index(t[0].reshape((-1, 1)))
                                xtf = np.zeros(self.num_tile * self.num_tiling)
                                xtf[xidx] = 1

                                yidx = self.tc.get_index(t[1].reshape((-1, 1)))
                                ytf = np.zeros(self.num_tile * self.num_tiling)
                                ytf[yidx] = 1

                                rnd.append(np.concatenate((xtf, ytf)))

                            rnd = numpy.array(rnd)

                        rnd = torch.from_numpy(rnd).float()

                    elif not self.continuous and not self.legal:
                        rnd_x = numpy.random.randint(0, 15, size=len(batch_x)).reshape((-1, 1))
                        rnd_y = numpy.random.randint(0, 15, size=len(batch_x)).reshape((-1, 1))
                        rnd = numpy.concatenate((rnd_x, rnd_y), axis=1)
                        rnd = one_hot_feature(rnd, 15, 15)
                        rnd = torch.from_numpy(rnd).float()

                rep, rec = self.net(batch_x[:, :len_x], decoder=True)

                if self.constraint:
                    rnd_rep, rnd_rep_rec = self.net(rnd)
                    rep_cst = self.rep_crit_cst(rep, rnd_rep)

                    self.rep_opt.zero_grad()
                    rep_cst.backward(retain_graph=True)
                    self.rep_opt.step()

                rep_mse = self.rep_crit_mse(rec, batch_x[:, len_x:])
                self.rep_opt.zero_grad()
                rep_mse.backward()
                self.rep_opt.step()

                rep_allloss.append(rep_mse.data.item())

                del step, batch_x
                if self.constraint:
                    del rnd, rnd_rep, rep_cst
                del rep, rec

            # ===================log========================
            # l1, l2 = numpy.mean(numpy.array(rep_allloss), axis=0)
            # print('epoch [{}/{}], rep_loss:{:.4f}, rcvs_loss:{:.4f}, time:{:.4f}'
            #       .format(epoch + 1, num_epochs, numpy.mean(numpy.array(rep_allloss)), numpy.mean(numpy.array(rcvs_allloss)), time.time() - start_time))
            print('epoch [{}/{}], time:{:.4f}'.format(epoch + 1, num_epochs, time.time() - start_time))
            rep_loss_change.append(numpy.mean(numpy.array(rep_allloss), axis=0))
            rcvs_loss_change.append(numpy.mean(numpy.array(rcvs_allloss)))

            del rep_allloss, rcvs_allloss

        return numpy.array(rep_loss_change), numpy.array(rcvs_loss_change)

    # # return representation
    # def encode(self, x):
    #     x = torch.from_numpy(x).float()
    #     enc = self.net.get_feature(x).detach().numpy()
    #     return enc

    # return representation, reconstruction, loss

    def test(self, x, len_x):
        x = torch.from_numpy(x).float()
        phi, rec = self.net.test(x[:, :len_x])
        return phi.detach().numpy(), rec.detach().numpy()

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
        self.net.load_state_dict(torch.load(path + file_name + ".pth"))
        return

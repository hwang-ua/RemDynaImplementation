"""
New code
"""

import torch
from torch import nn
import torch.utils.data as Data
import os
import time
import numpy
from utils.distance_matrix_func import *
import random

class MyDataset(Data.Dataset):
    def __init__(self, data):
        self.data_here = data

    def __getitem__(self, index):
        data = self.data_here[index]

        # Your transformations here (or set it in CIFAR10)

        return data, index

    def __len__(self):
        return len(self.data_here)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# class RcvsNN(nn.Module):
#     def __init__(self, num_input, num_node, num_output):
#         super(RcvsNN, self).__init__()
#
#         self.layers = []
#         self.layers.append(nn.Linear(num_input, num_node[0]))
#         nn.init.kaiming_normal_(self.layers[-1].weight)
#         self.layers.append(nn.ReLU(True))
#
#         for i in range(len(num_node) - 1):
#             self.layers.append(nn.Linear(num_node[i], num_node[i+1]))
#             nn.init.kaiming_normal_(self.layers[-1].weight)
#
#             self.layers.append(nn.ReLU(True))
#
#         self.layers.append(nn.Linear(num_node[-1], num_output))
#         nn.init.normal_(self.layers[-1].weight, std=0.01)
#
#         self.nnet = nn.Sequential(*self.layers)
#
#     def forward(self, x):
#         v = self.nnet(x)
#         return v
#
#     def test(self, x):
#         with torch.no_grad():
#             eval = self.nnet(x)
#         return eval
#
#     def print_net(self):
#         print("=========== NN structure ==========")
#         print(self.nnet)
#         print("===================================")

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
            # nn.init.xavier_normal_(self.en_layers[0].weight)
            # self.en_layers.append(nn.Tanh())
            # self.en_layers.append(nn.BatchNorm1d(num_features=num_node[0]))

            for i in range(len(num_node) - 1):
                self.en_layers.append(nn.Linear(num_node[i], num_node[i + 1]))
                nn.init.kaiming_normal_(self.en_layers[-1].weight)
                self.en_layers.append(nn.ReLU(True))
                # nn.init.xavier_normal_(self.en_layers[-1].weight)
                # self.en_layers.append(nn.Tanh())
                # self.en_layers.append(nn.BatchNorm1d(num_features=num_node[i + 1]))
            # self.en_layers.append(nn.Dropout(p=0.0))
            self.en_layers.append(nn.Linear(num_node[-1], num_feature))
            nn.init.normal_(self.en_layers[-1].weight, std=0.01)
            # self.en_layers.append(nn.Sigmoid())
            # self.en_layers.append(nn.Tanh())
        else:
            self.en_layers.append(nn.Linear(num_input, num_feature))
            nn.init.kaiming_normal_(self.en_layers[0].weight)
            self.en_layers.append(nn.ReLU(True))
            # nn.init.xavier_normal_(self.en_layers[0].weight)
            # self.en_layers.append(nn.Tanh())
            # self.en_layers.append(nn.BatchNorm1d(num_features=num_feature))

        self.encoder = nn.Sequential(*self.en_layers)

        # Representation -> Successor Feature

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
            nn.init.normal_(self.de_layers[-1].weight, std=0.01)
            self.decoder = nn.Sequential(*self.de_layers)

        else:
            print("Successor Feature: ELSE block")
            self.de_layers.append(nn.Linear(num_feature, num_output))
            nn.init.normal_(self.de_layers[-1].weight, std=0.01)
            self.decoder = nn.Sequential(*self.de_layers)

    def forward(self, x, decoder = False):
        r = self.encoder(x)
        # y = self.decoder(r)
        # return r, y
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
                 weight_decay=0, num_dn=None, num_rec_node=[2], optimizer="adam", dropout=0, beta=0, delta=1, weight_reward = None,
                 legal = False, continuous=False, num_tiling=None, num_tile=None, constraint=True, input_range=[0, 1],
                 image_catch = False):

        self.image_catch = image_catch
        self.net = AutoEncoder(num_input, num_node, num_feature, num_output, num_dec_node=num_dn, dropout=dropout)
        # self.rcvs_net = RcvsNN(num_feature, num_rec_node, num_input)
        self.net = self.net.to(device)
        # self.rcvs_net = self.rcvs_net.to(device)

        self.weight_reward = weight_reward
        self.beta = beta
        self.delta = delta
        self.rep_crit_mse = self.mse_loss # nn.MSELoss() #
        # self.rep_crit_cst = self.cst_loss
        self.rep_crit_cst1 = self.cst_loss1
        self.rep_crit_cst2 = self.cst_loss2

        self.rcvs_crit = nn.MSELoss() # self.loss #
        self.input_range = input_range
        print("Beta:", self.beta)
        print("Delta:", self.delta)
        print("Learning rate:", learning_rate, learning_rate_rcvs)
        print("Input range:", self.input_range)

        if optimizer == "adam":
            self.rep_opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            # self.rcvs_opt = torch.optim.Adam(self.rcvs_net.parameters(), lr=learning_rate_rcvs, weight_decay=weight_decay)
        elif optimizer == "AMSGrad":
            self.rep_opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                            amsgrad=True)
            # self.rcvs_opt = torch.optim.Adam(self.rcvs_net.parameters(), lr=learning_rate_rcvs, weight_decay=weight_decay,
            #                                  amsgrad=True)

        self.legal = legal
        self.continuous = continuous
        self.num_input = num_input
        self.num_output = num_output
        if self.continuous:
            self.tc = TileCoding(1, num_tiling, num_tile)
            # self.tc = TileCoding(2, num_tiling, num_tile)
            self.num_tiling = num_tiling
            self.num_tile = num_tile

        self.constraint = constraint
        print("Use constraint:", self.constraint)

        self.net.print_net()

    def loss(self, y_hat, y, rnd1, rnd2):
        # prop = ((y_hat - y) ** 2).mean() / ((rnd1 * rnd2).sum(dim=1) ** 2).mean() * self.beta
        ls = ((y_hat - y) ** 2).mean() \
             + self.beta * (((rnd1 * rnd2).sum(dim=1) ** 2).mean()
                            - self.delta * (rnd1 ** 2).sum(dim=1).mean() - self.delta * (rnd2 ** 2).sum(dim=1).mean()
                            + self.delta**2*32)

        # print(((y_hat - y) ** 2).mean().data.item(),
        #       ((rnd1 * rnd2).sum(dim=1) ** 2).mean().data.item(),
        #       (rnd1 ** 2).mean().data.item(),
        #       (rnd2 ** 2).mean().data.item())
        # print("= ", ls, self.beta)
        return ls

    def mse_loss(self, y_hat, y):
        if self.weight_reward:
            ls = ((y_hat[:, :-1] - y[:, :-1]) ** 2).sum(dim=1).mean()
            ls_r = ((y_hat[:, -1:] - y[:, -1:]) ** 2).sum(dim=1).mean()
            ls += self.weight_reward * ls_r
        else:
            ls = ((y_hat - y) ** 2).sum(dim=1).mean()
        return ls

    # def cst_loss(self, rnd1, rnd2):
    #     ls = self.beta * (((rnd1 * rnd2).sum(dim=1) ** 2).mean()
    #                       - self.delta * (rnd1 ** 2).mean() - self.delta * (rnd2 ** 2).mean() + self.delta**2*32)
    #     # ls = self.beta * (((rnd1 * rnd2).sum(dim=1) ** 2).mean()
    #     #                   - self.delta * (rnd1 ** 2).sum(dim=1).mean() - self.delta * (rnd2 ** 2).sum(dim=1).mean() + self.delta ** 2 * 32)
    #     # print("cst", ((rnd1 * rnd2).sum(dim=1) ** 2).mean(), (rnd1 ** 2).sum(dim=1).mean(), (rnd2 ** 2).sum(dim=1).mean())
    #     return ls

    def cst_loss1(self, rnd1, rnd2):
        ls = self.beta * ((rnd1 * rnd2).sum(dim=1) ** 2).mean()
        return ls

    def cst_loss2(self, rnd):
        # ls = -1 * self.beta * self.delta * (rnd ** 2).sum(dim=1).mean()
        ls = self.beta * self.delta * ((rnd ** 2).sum(dim=1).mean() - 1) ** 2
        return ls

    #supervised loss
    def training(self, x, len_x, num_epochs, num_epochs_rcvs, batch_size):
    # #TD loss
    # def training(self, x, len_x, num_epochs, num_epochs_rcvs, batch_size, gamma_list, feature_size):
    #     self.net.print_net()
        # self.rcvs_net.print_net()

        # #supervised loss
        data_tensor = torch.from_numpy(x).float()#.to(device)
        data = Data.TensorDataset(data_tensor)
        loader = Data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        # #TD loss
        # data_tensor_backup = torch.from_numpy(x).float().to(device)
        # x = x[:-1]
        # data_tensor = torch.from_numpy(x).float().to(device)
        # data = Data.TensorDataset(data_tensor)
        # data = MyDataset(data)
        # loader = Data.DataLoader(
        #     dataset=data,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=0
        # )

        self.rep_loss_change = []
        rcvs_loss_change = []

        for epoch in range(max(num_epochs, num_epochs_rcvs)):
            rep_allloss = []
            # rcvs_allloss = []

            start_time = time.time()
            # for _ in [1]:
            #     step = 0
            #     batch_x = data_tensor[numpy.random.randint(0, len(x), size=batch_size)]

            #supervised loss
            # for step, [batch_x] in list(enumerate(loader))[:1]:
            for step, [batch_x] in enumerate(loader):

            # #TD loss
            # for (stuff, sample) in enumerate(loader):
            #     [batch_x] = sample[0]
            #     batch_x_idx = sample[1]
            #     batch_x_idx += 1
            #     batch_x_sup = data_tensor_backup[batch_x_idx,:2]
            #     batch_x_sup_features = data_tensor_backup[batch_x_idx,2:]
                batch_x = batch_x.to(device)
                if self.constraint:
                    if self.legal:
                        rnd_idx = numpy.random.randint(0, len(x), size=len(batch_x))
                        rnd = data_tensor[rnd_idx, :len_x].to(device)

                        # ind_batch = np.arange(len(batch_x))
                        # rnd_idx1, rnd_idx2 = numpy.meshgrid(ind_batch, ind_batch)
                        # rnd_idx1, rnd_idx2 = rnd_idx1.reshape(-1), \
                        #                      rnd_idx2.reshape(-1)
                        # rnd_idx1, rnd_idx2 = rnd_idx1[np.where(rnd_idx1 != rnd_idx2)[0]], \
                        #                      rnd_idx2[np.where(rnd_idx1 != rnd_idx2)[0]]
                        # rnd1, rnd2 = batch_x[rnd_idx1, :len_x], \
                        #              batch_x[rnd_idx2, :len_x]


                    elif self.image_catch and not self.legal:
                        # states = numpy.zeros((len(batch_x), len_x))
                        # for idx in range(len(batch_x)):
                        #     one_pos = random.sample(range(len_x), 2)
                        #     states[idx][one_pos] = 1
                        states = numpy.random.randint(0, 2, size=len_x * len(batch_x)).reshape((len(batch_x), len_x))
                        rnd = torch.from_numpy(states).float().to(device)

                    elif self.continuous and not self.legal:
                        states = numpy.random.random(size=len(batch_x)*2).reshape((-1, 2))
                        if self.num_input == 2:
                            if self.input_range[0] == 0:
                                rnd = states
                            elif self.input_range[0] == -1:
                                rnd = states * 2 - 1
                            else:
                                print("NN: Unknown range for xy input")
                                exit(-1)
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

                                # idx = self.tc.get_index(t)
                                # tf = np.zeros(self.num_tile**2 * self.num_tiling)
                                # tf[idx] = 1
                                # rnd.append(tf)
                            rnd = numpy.array(rnd)

                        rnd = torch.from_numpy(rnd).float().to(device)
                        # rnd1 = torch.from_numpy(rnd[:len(rnd)//2]).float().to(device)
                        # rnd2 = torch.from_numpy(rnd[len(rnd)//2:]).float().to(device)

                    elif not self.continuous and not self.legal:
                        rnd_x = numpy.random.randint(0, 15, size=len(batch_x)).reshape((-1, 1))
                        rnd_y = numpy.random.randint(0, 15, size=len(batch_x)).reshape((-1, 1))
                        rnd = numpy.concatenate((rnd_x, rnd_y), axis=1)
                        rnd = one_hot_feature(rnd, 15, 15)
                        rnd = torch.from_numpy(rnd).float().to(device)
                        # rnd = numpy.random.randint(2, size= len(batch_x)*len_x).reshape((len(batch_x), len_x))
                        # rnd = torch.from_numpy(rnd).float().to(device)

                self.rep_opt.zero_grad()

                #supervised loss
                rep, rec = self.net(batch_x[:, :len_x], decoder=True)
                # rep_mse = self.rep_crit_mse(rec[:, :self.num_output //2], batch_x[:, len_x:len_x+self.num_output//2])
                # rep_mse.backward(retain_graph=True)
                # rep_mse = self.rep_crit_mse(rec[:, self.num_output //2:], batch_x[:, len_x+self.num_output//2:])
                # rep_mse.backward(retain_graph=True)
                rep_mse = self.rep_crit_mse(rec, batch_x[:, len_x:])
                if self.constraint:
                    rep_mse.backward(retain_graph=True)
                else:
                    rep_mse.backward()
                    self.rep_opt.step()

                # #TD loss
                # rep, rec = self.net(batch_x[:, :len_x], decoder=True)
                # rep_sdash, rec_sdash = self.net(batch_x_sup, decoder=True)
                # rec_sdash = rec_sdash.detach()
                # rec_sdash[:,:feature_size] *= gamma_list[0]
                # rec_sdash[:,feature_size:] *= gamma_list[1]
                # rec_sdash[:,:feature_size] += batch_x_sup_features
                # rec_sdash[:,feature_size:] += batch_x_sup_features
                # # rec_sdash[:,:feature_size] *= (1.0/(1.0-gamma_list[0]))
                # # rec_sdash[:,feature_size:] *= (1.0/(1.0-gamma_list[1]))
                # rep_mse = self.rep_crit_mse(rec[:, :self.num_output //2], rec_sdash[:, :self.num_output //2])
                # rep_mse.backward(retain_graph=True)
                # rep_mse = self.rep_crit_mse(rec[:, self.num_output //2:], rec_sdash[:, self.num_output //2:])
                # rep_mse.backward(retain_graph=True)


                # self.rep_opt.step()

                if self.constraint:
                    # self.rep_opt.zero_grad()

                    rnd_rep, rnd_rep_rec = self.net(rnd, decoder=True)
                    # rnd_rep1, _ = self.net(rnd1, decoder=False)
                    # rnd_rep2, _ = self.net(rnd2, decoder=False)

                    rep_cst = self.rep_crit_cst1(rep, rnd_rep)
                    # rep_cst = self.rep_crit_cst1(rnd_rep1, rnd_rep2)
                    rep_cst.backward(retain_graph=True)

                    rep_cst2 = self.rep_crit_cst2(rep)
                    # rep_cst2 = self.rep_crit_cst2(rnd_rep1)
                    rep_cst2.backward(retain_graph=True)

                    rep_cst3 = self.rep_crit_cst2(rnd_rep)
                    # rep_cst3 = self.rep_crit_cst2(rnd_rep2)
                    rep_cst3.backward()#retain_graph=True)
                    # print((rep_cst-rep_cst2-rep_cst3).data.item())

                    self.rep_opt.step()

                # # self.rep_opt.zero_grad()
                # cons2_rnd = numpy.random.random(size=len(batch_x) * 2).reshape((-1, 2))
                # cons2_rnd = torch.from_numpy(cons2_rnd).float().to(device)
                # # rep, rec = self.net(batch_x[:, :len_x], decoder=False)
                # cons2_rnd_rep, cons2_rnd_rec = self.net(cons2_rnd, decoder=False)
                # cons2_mse = self.beta * 0.1 * (cons2_rnd_rep ** 2).sum(dim=1).mean()
                # cons2_mse.backward()
                # self.rep_opt.step()

                # print("mse: {:8.0f}, {:8.0f}".format(rep_mse.detach().numpy(), rep_cst.detach().numpy()))

                # rep_cst.backward()
                # stt = self.rcvs_net(rep)
                # rcvs_loss = self.rcvs_crit(stt, batch_x[:, :len_x])
                # self.rcvs_opt.zero_grad()
                # rcvs_loss.backward()
                # self.rcvs_opt.step()

                if self.constraint:
                    rep_allloss.append([rep_mse.data.item(), rep_cst.data.item(), rep_cst2.data.item(), rep_cst3.data.item()])#, cons2_mse.data.item()])
                else:
                    rep_allloss.append([rep_mse.data.item(), 0, 0, 0])  # , cons2_mse.data.item()])

            # ===================log========================
            l1, l2, l3, l4 = numpy.mean(numpy.array(rep_allloss), axis=0)
            print('epoch [{}/{}], rep_loss:{:.4f},{:.4f},{:.4f},{:.4f}, time:{:.4f}'
                  .format(epoch + 1, num_epochs, l1, l2, l3, l4, time.time() - start_time))
            # print('epoch [{}/{}], time:{:.4f}'.format(epoch + 1, num_epochs, time.time() - start_time))
            self.rep_loss_change.append(numpy.mean(numpy.array(rep_allloss), axis=0))
            # rcvs_loss_change.append(numpy.mean(numpy.array(rcvs_allloss)))

            if epoch % 100 == 0:
                self.save_checkpoint(epoch)

            # del rep_allloss, rcvs_allloss

        return numpy.array(self.rep_loss_change), numpy.array(rcvs_loss_change)

    # # return representation
    # def encode(self, x):
    #     x = torch.from_numpy(x).float()
    #     enc = self.net.get_feature(x).detach().numpy()
    #     return enc

    # return representation, reconstruction, loss

    def test(self, x, len_x):
        x = torch.from_numpy(x).float().to(device)
        phi, rec = self.net.test(x[:, :len_x])
        # rcvs = self.rcvs_net.test(phi)
        # rcvs_loss = self.rcvs_crit(rcvs, x[:, :len_x])
        return phi.detach().cpu().numpy(), rec.detach().cpu().numpy(), None, None#, rcvs.detach().cpu().numpy(), rcvs_loss.data.item()

    # save model
    def saving(self, path, file_name):
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
            print("make new dir", path)
        while os.path.exists(path + file_name):
            file_name += "-"
        torch.save(self.net.state_dict(), path + file_name + ".pth")
        # torch.save(self.rcvs_net.state_dict(), path + file_name + "_rcvs.pth")
        print("model saved in:", path + file_name + ".pth")
        return

    def save_checkpoint(self, num_epoch):
        self.saving("./", "_checkpoint_legal"+str(self.legal))
        numpy.save("_loss_legal"+str(self.legal), numpy .array(self.rep_loss_change))
        print("checkpoint saved", num_epoch)

    def saving_entire(self, path, file_name):
        torch.save(self.net, path + file_name + "_entire.pth")
        print("model saved in:", path + file_name + "_entire.pth")


    # load model
    def loading(self, path, file_name):
        print("*** encoder: load saved model ***")
        m = torch.load(path + file_name + ".pth", map_location='cpu')
        for k in m.keys():
            print(k, m[k].size())
        print()

        self.net.load_state_dict(torch.load(path + file_name + ".pth", map_location='cpu'))
        self.net.to(device)
        # self.rcvs_net.load_state_dict(torch.load(path + file_name + "_rcvs.pth", map_location='cpu'))
        # self.rcvs_net.to(device)
        return

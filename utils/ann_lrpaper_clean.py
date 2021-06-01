import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
import os
import time
import numpy
from utils.distance_matrix_func import one_hot_feature

class ANN(nn.Module):
    def __init__(self, num_input, num_node, num_output):
        super(ANN, self).__init__()

        # Neural net structure

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

class NN():
    def __init__(self, num_input, num_node, num_output, learning_rate, weight_decay, mode = "xy"):

        self.net = ANN(num_input, num_node, num_output)
        self.mse = self.mse_loss
        self.mode = mode
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    """
    MSE
    """
    def mse_loss(self, output, target):
        loss = ((output - target) ** 2).sum(dim=1).mean()
        return loss

    """
    Train Neural net
    s: [[trajectory number, step number, x, y]...]
    sp: dictionary {(trajectory number,  step number, x, y): [[x_t+i, y_t+i], ...], ...}
    num_epochs: number of epochs for training
    batch_size: batch size
    """
    def training(self, s, sp, num_epochs, batch_size, print_loss = False):

        # probability for choosing the (t+i)_{th} successor
        prob_dist = numpy.array([0.9**i*0.1 for i in range(0, 50)])
        # make the sum of probability to be 1
        prob_dist /= prob_dist.sum()

        # index 0 (trajectory number) and 1 (step number) are not state
        f_len = len(s[0]) - 2
        dataset = s

        # log
        loss_change = []

        for epoch in range(num_epochs):
            # log
            allloss = []
            start_time = time.time()

            # =========== batch for the current epoch =========
            ft_s = numpy.zeros((batch_size, f_len))
            ft_sp = numpy.zeros((batch_size, f_len))
            count = 0
            while count < batch_size:

                # choose u
                s_idx = numpy.random.randint(len(dataset))
                s = dataset[s_idx]

                # choose v
                suc = sp[str(s.astype(numpy.float))]
                idx = numpy.random.choice(len(prob_dist), p=prob_dist)
                # if length of trajectory is less than 50,
                # it is possible to choose an index out of range
                # thus choose again
                while idx >= len(suc):
                    idx = numpy.random.choice(len(prob_dist), p=prob_dist)

                # take x,y from data and save as u
                ft_s[count] = s[-f_len:]
                # take chosen successor and save as v
                ft_sp[count] = suc[idx]

                count += 1

            # cross pairs
            rd_u = []
            rd_v = []
            for i in range(len(ft_s)):
                for j in range(len(ft_s)):
                    if i != j:
                        rd_u.append(ft_s[i])
                        rd_v.append(ft_s[j])
            rd_u = numpy.array(rd_u)
            rd_v = numpy.array(rd_v)

            ft_s = torch.from_numpy(ft_s).float()
            ft_sp = torch.from_numpy(ft_sp).float()
            rd1 = torch.from_numpy(rd_u).float()
            rd2 = torch.from_numpy(rd_v).float()


            # ========== training ==========
            self.optimizer.zero_grad()

            # representations of u and v
            out_s = self.net(ft_s)
            out_sp = self.net(ft_sp)

            # mse loss
            mse = self.mse_loss(out_s, out_sp)
            mse.backward()

            # representations for constraint
            out_r1 = self.net(rd1)
            out_r2 = self.net(rd2)

            # orthogonality constraint
            temp1=(5.0 * ((out_r1 * out_r2).sum(dim=1) ** 2).mean())
            temp1.backward(retain_graph=True)
            temp2=(-5.0 * 0.05 * ((out_r1 ** 2).sum(dim=1).mean()))
            temp2.backward(retain_graph=True)
            temp3 = (-5.0 * 0.05 * ((out_r2**2).sum(dim=1).mean()) + (5.0 * 0.05 ** 2))
            temp3.backward()
            self.optimizer.step()

            allloss.append([mse.data.item(), temp1.data.item(), temp2.data.item(), temp3.data.item()])
            del ft_s, ft_sp
            del suc
            del rd2
            del out_s, out_sp, out_r2
            del mse
            del temp1, temp2, temp3

            # ============ log ==========
            if print_loss:
                l1, l2, l3, l4 = numpy.mean(numpy.array(allloss), axis=0)
                print('epoch [{}/{}], loss:[{:.4f},{:.4f},{:.4f},{:.4f}], time:{:.4f}'
                      .format(epoch + 1, num_epochs, l1, l2, l3, l4, time.time() - start_time))
            loss_change.append(numpy.mean(numpy.array(allloss), axis=0))

            del allloss

        return numpy.array(loss_change)


    def test(self, data):

        x = torch.from_numpy(data).float()
        estimation = self.net.get_value(x)
        loss = None
        return estimation.detach().numpy(), loss

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
        self.net.load_state_dict(torch.load(path + file_name + ".pth"))
        return

    def print_nn(self):
        self.net.print_net()
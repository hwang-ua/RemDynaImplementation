import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
import os
import time
import numpy
import random
from utils.distance_matrix_func import one_hot_feature

class ANN(nn.Module):
    def __init__(self, num_input, num_node, num_output):
        super(ANN, self).__init__()

        self.layers = []
        self.layers.append(nn.Linear(num_input, num_node[0]))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        # nn.init.constant_(self.layers[-1].bias.data, 0)
        self.layers.append(nn.ReLU(True))
        # self.layers.append(nn.SELU(True))

        for i in range(len(num_node) - 1):
            self.layers.append(nn.Linear(num_node[i], num_node[i+1]))
            nn.init.xavier_uniform_(self.layers[-1].weight)
            # nn.init.constant_(self.layers[-1].bias.data, 0)
            self.layers.append(nn.ReLU(True))
            # self.layers.append(nn.SELU(True))

        self.layers.append(nn.Linear(num_node[-1], num_output))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        # nn.init.constant_(self.layers[-1].bias.data, 0)

        self.nnet = nn.Sequential(*self.layers)
        print(self.nnet)

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
    def __init__(self, num_input, num_node, num_output, learning_rate, weight_decay,
                 optimizer="adam", dropout = 0, momentum = 0, mode = "xy", img_catch = False):

        self.img_catch = img_catch
        self.net = ANN(num_input, num_node, num_output)
        self.mse = self.mse_loss
        self.constraint = self.cst_loss
        self.mode = mode
        self.num_input = num_input
        self.num_output = num_output

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay) # adam optimizer

    def mse_loss(self, output, target):
        # loss = ((output - target) ** 2).sum(dim=1).mean()
        loss = 0.5 * torch.mean(torch.sum((output - target) ** 2, 1))
        return loss
    def cst_loss(self, rnd1, rnd2):
        loss = 5.0 * (((rnd1 * rnd2).sum(dim=1) ** 2).mean() - 0.05 * (rnd1 ** 2).mean() - 0.05 * (rnd2 ** 2).mean()
                      + 0.05**2*20)
        return loss

    def training(self, s, sp, size_x, size_y, num_epochs, batch_size=1, legalv=0, print_loss = False):
        """
        s: num_traj, num_step, x, y
        sp: dictionary {(traj, num_step, x, y): [[x_t+i, y_t+i], ...]}
        """
        self.legal = legalv
        prob_dist = numpy.array([0.9**i*0.1 for i in range(0, 50)])
        # prob_dist[0] = 1
        prob_dist /= prob_dist.sum()

        # f_len = 2#len(s[0]) - 2
        f_len = self.num_input
        dataset = s

        self.loss_change = []
        for epoch in range(num_epochs):
            allloss = []
            start_time = time.time()

            ft_s = numpy.zeros((batch_size, f_len))
            ft_sp = numpy.zeros((batch_size, f_len))
            count = 0
            while count < batch_size:
                idx = numpy.random.choice(len(prob_dist), p=prob_dist)
                s_idx = numpy.random.randint(len(dataset)-50)
                #s = dataset[s_idx]
                #ft_s[count] = s[len(prob_dist) - 1 - idx]
                #ft_sp[count] = s[-1]
                ft_s[count] = dataset[s_idx]
                ft_sp[count] = dataset[s_idx + idx]                
                count += 1

            if legalv:
                rd_u = numpy.zeros((batch_size, f_len))
                rd_v = numpy.zeros((batch_size, f_len))
                count = 0
                while count < batch_size:
                    idx_v = numpy.random.choice(len(prob_dist), p=prob_dist)
                    s_idx_u = numpy.random.randint(len(dataset)-50)
                    s_idx_v = numpy.random.randint(len(dataset)-50)
                    u = dataset[s_idx_u: s_idx_u+50]
                    v = dataset[s_idx_v: s_idx_v+50]
                    rd_u[count] = u[len(prob_dist) - 1 - idx_v]
                    rd_v[count] = v[0]#[-1]
                    count += 1
                # rd_u = numpy.array([[x]*batch_size for x in ft_s]).reshape(-1, 2)
                # rd_v = numpy.array([ft_sp] * batch_size).reshape(-1, 2)

            else:
                if self.img_catch:
                    random_idx = numpy.random.randint(len(dataset), size=batch_size)
                    rd_u = numpy.zeros((batch_size, f_len))
                    rd_v = numpy.zeros((batch_size, self.num_input))
                    for idx in range(batch_size):
                        # rd_u[idx] = dataset[random_idx[idx]][-1]
                        rd_u[idx] = dataset[random_idx[idx]]
                        one_pos = random.sample(range(self.num_input), 2)
                        rd_v[idx][one_pos] = 1

                else:
                    u_idx = numpy.random.randint(0, len(dataset), size=batch_size)
                    # rd_u = dataset[u_idx][:, 0]
                    rd_u = dataset[u_idx]
                    rd_v = numpy.random.random(size=len(ft_sp)*2).reshape((-1, 2)) * 2 - 1

            ft_s = torch.from_numpy(ft_s).float()
            ft_sp = torch.from_numpy(ft_sp).float()
            rd1 = torch.from_numpy(rd_u).float()
            rd2 = torch.from_numpy(rd_v).float()

            out_s = self.net(ft_s)
            out_sp = self.net(ft_sp)
            mse = self.mse_loss(out_s, out_sp)

            out_r1 = self.net(rd1)
            out_r2 = self.net(rd2)

            dot_product = torch.bmm(out_r1.view(len(out_r1), 1, self.num_output), out_r2.view(len(out_r2), self.num_output, 1)) ** 2
            dot_product = dot_product.view(len(out_r1))
            orth_loss = dot_product - 0.05 * torch.sum(out_r1 ** 2, 1) - 0.05 * torch.sum(out_r2 ** 2, 1) + self.num_output * 0.05 ** 2
            orth_loss = 5.0 * torch.mean(orth_loss)
            # orth_loss = dot_product - 0.01 * torch.sum(out_r1 ** 2, 1) - 0.01 * torch.sum(out_r2 ** 2, 1) + self.num_output * 0.01 ** 2
            # orth_loss = 0.01 * torch.mean(orth_loss)
            total_loss = mse + orth_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # self.optimizer.zero_grad()
            # temp1=(5.0 * ((out_r1 * out_r2).sum(dim=1) ** 2).mean())
            # temp1.backward(retain_graph=True)
            # temp2=(-5.0*0.05* ((out_r1 ** 2).sum(dim=1).mean()))
            # temp2.backward(retain_graph=True)
            # temp3 = (-5.0*0.05*((out_r2**2).sum(dim=1).mean()) + (5.0 * 0.05 ** 2))
            # temp3.backward()
            # mse.backward()
            # self.optimizer.step()

            # allloss.append([mse.data[0].item(), temp1.data[0].item(), temp2.data[0].item(), temp3.data[0].item()])
            # allloss.append([mse.data.item(), temp1.data.item(), temp2.data.item(), temp3.data.item()])
            allloss.append([mse.data.item(), orth_loss.data.item(), total_loss.data.item(), 0])

            # ===================log========================
            if print_loss:
                l1, l2, l3, l4 = numpy.mean(numpy.array(allloss), axis=0)
                print('epoch [{}/{}], loss:[{:.4f},{:.4f},{:.4f},{:.4f}], time:{:.4f}'
                      .format(epoch + 1, num_epochs, l1, l2, l3, l4, time.time() - start_time))
            self.loss_change.append(numpy.mean(numpy.array(allloss), axis=0))

            del allloss
            # del batch_x

            # # =================== check memory ===================
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(reduce(op_mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
            #     except:
            #         print("Exception")

        return numpy.array(self.loss_change)


    def test(self, data):

        x = torch.from_numpy(data).float()#.to(dtype=torch.float16)
        estimation = self.net.get_value(x)
        loss = None#self.criterion(estimation, y)
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

    def save_checkpoint(self, num_epoch):
        self.saving("./", "_checkpoint_legal"+str(self.legal))
        numpy.save("_loss_legal"+str(self.legal), numpy.array(self.loss_change))
        print("checkpoint saved", num_epoch)

    def loading(self, path, file_name):
        self.net.load_state_dict(torch.load(path + file_name + ".pth"))
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

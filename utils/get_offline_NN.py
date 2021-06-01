import utils.NN_model as nnm
import numpy as np

class GetOfflineNN:
    def __init__(self):
        # self.f_input = 3
        self.f_input = 6
        self.f_output = 4

        # self.b_input = 3
        self.b_input = 6
        self.b_output = 2

        self.node = [512, 256, 128]
        self.lr = 5e-4

        self.num_epochs = 200
        self.batch_size = 256
        self.train_ep = 500

        self.model_path = "prototypes/offline_NN/"
        self.model_name = "offlineNN_oneHotA" \
                          "_node" + str(self.node) + \
                          "_lr" + str(self.lr) + \
                          "_epoch"+str(self.num_epochs) + \
                          "_batch"+str(self.batch_size) + \
                          "_trainEp"+str(self.train_ep)

        self.fnn = nnm.NNModel(self.f_input, self.node, self.f_output, self.lr)
        self.bnn = nnm.NNModel(self.b_input, self.node, self.b_output, self.lr)

        self.fnn.loading(self.model_path, self.model_name + "_forward")
        self.bnn.loading(self.model_path, self.model_name + "_backward")

        return

    def forward_prediction(self, state, action):
        # sp = self.fnn.predict(np.concatenate((state, [action]), axis=0))
        sp = self.fnn.predict(np.concatenate((state, self.one_hot(action)), axis=0))
        sp[:2] = np.clip(sp[:2], 0.0, 1.0)
        return [sp[0], sp[1]], sp[2], sp[3]

    def backward_prediction(self, action, sp):
        # s = self.bnn.predict(np.concatenate(([action], sp), axis=0))
        s = self.bnn.predict(np.concatenate((self.one_hot(action), sp), axis=0))
        s = np.clip(s, 0.0, 1.0)
        return s

    def one_hot(self, action):
        temp = np.zeros(4)
        temp[int(action)] = 1
        return temp


class GetOfflineRepNN:
    def __init__(self):
        self.f_input = 32+4
        self.f_output = 32

        self.b_input = 4+32
        self.b_output = 32

        self.r_input = 32+4 #6 # s, one-hot a
        self.r_output = 2   #2 # r, g

        self.node = [512, 512, 512]
        self.lr = 0.0001

        self.num_epochs = 200
        self.batch_size = 1024
        self.train_ep = 500

        self.gate = {"f": "None",
                     "b": "None",
                     "r": "Relu"}

        self.model_path = "prototypes/offline_NN_separate/"
        self.model_name = "offlineNN_oneHotA" \
                          "_node" + str(self.node) + \
                          self.gate.get("f")+self.gate.get("b")+self.gate.get("r") + \
                          "_lr" + str(self.lr) + \
                          "_epoch"+str(self.num_epochs) + \
                          "_batch"+str(self.batch_size) + \
                          "_trainEp"+str(self.train_ep)

        self.fnn = nnm.NNModel(self.f_input, self.node, self.f_output, self.lr, gate=self.gate.get("f"))
        self.bnn = nnm.NNModel(self.b_input, self.node, self.b_output, self.lr, gate=self.gate.get("b"))
        self.rgnn = nnm.NNModel(self.r_input, self.node, self.r_output, self.lr, gate=self.gate.get("r"))

        print("foward model")
        self.fnn.loading(self.model_path, self.model_name + "_forward")
        print("backward model")
        self.bnn.loading(self.model_path, self.model_name + "_backward")
        print("reward model")
        self.rgnn.loading(self.model_path, self.model_name + "_rg")

        return

    def forward_prediction(self, state, action):
        sp = self.fnn.predict(np.concatenate((state, self.one_hot(action)), axis=0))
        rg = self.rgnn.predict(np.concatenate((state, self.one_hot(action)), axis=0))
        sp = np.clip(sp, 0.0, 1.0)
        return sp, rg[0], rg[1]

    def backward_prediction(self, action, sp):
        # s = self.bnn.predict(np.concatenate(([action], sp), axis=0))
        s = self.bnn.predict(np.concatenate((self.one_hot(action), sp), axis=0))
        s = np.clip(s, 0.0, 1.0)
        return s

    def one_hot(self, action):
        temp = np.zeros(4)
        temp[int(action)] = 1
        return temp
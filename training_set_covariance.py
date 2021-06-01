import numpy as np

class TrainingSetCov:
    def __init__(self, rep_model, num_feature, data_file):
        self.rep_model = rep_model
        self.num_feature = num_feature
        self.data = np.load(data_file)
        self.data = self.data[:5000, :2]
        self.sasprg = np.zeros((self.data.shape[0]-1, num_feature * 2 + 3))

        for i in range(self.data.shape[0] - 1):
            s = self.rep_model.state_representation(self.data[i])
            s /= np.linalg.norm(s)
            sp = self.rep_model.state_representation(self.data[i+1])
            sp /= np.linalg.norm(sp)
            if self._check_goal(self.data[i+1]):
                r = 1
                g = 0
            else:
                r = 0
                g = 0.9
            seq = np.concatenate([s, [0], sp, [r], [g]])
            self.sasprg[i] = seq

        print("data size", self.sasprg.shape)


    def get_representation_cov(self):
        return np.cov(self.sasprg.T)

    def _check_goal(self, xy):
        x, y = xy
        if 0.7 <= x <= 0.75 and 0.95 <= y <= 1.0:
            return True
        else:
            return False

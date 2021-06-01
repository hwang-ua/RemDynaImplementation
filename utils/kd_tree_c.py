import ctypes

kd_tree = ctypes.windll.LoadLibrary("../kdtree-0.5.6/kdtree.so")
print(kd_tree)
# lib.foo_(1, 3)
print('***finish***')

class KD_Tree():
    def __init__(self):
        self.tree =
        return
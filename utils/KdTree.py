import numpy as np

class Node:
    def __init__(self, val):
        self.val = val
        self.rc = None
        self.lc = None
        return

    def add_rc(self, rc):
        self.rc = rc

    def add_lc(self, lc):
        self.lc = lc

    def get_val(self):
        return self.val

    def get_lc(self):
        return self.lc

    def get_rc(self):
        return self.rc

    def is_leaf(self):
        return (self.rc is None) and (self.lc is None)

class CGW_KD_Tree:
    def __init__(self, dim, n=10000):
        self.dim = dim
        # self.indices = np.array([0,1,3,4])
        self.root = None
        self.n = n
        self.b = 0
        self.data_array = np.zeros((self.n, dim))
        return

    def insert(self, val):
        # val = val[self.indices]

        if self.b < self.n:
            self.data_array[self.b] = val
        else:
            np.concatenate((self.data_array, np.array([val])), axis = 0)
        self.b += 1

        node = Node(val)

        if self.root is None:
            self.root = node
            return

        previous = None
        rorl = None
        current = self.root
        done = False
        i = 0
        while not done:
            if current is None:
                if rorl == 'l':
                    previous.add_lc(node)
                elif rorl == 'r':
                    previous.add_rc(node)
                done = True
            elif val[i % self.dim] < current.get_val()[i % self.dim]:
                previous = current
                rorl = 'l'
                current = current.get_lc()
            elif val[i % self.dim] >= current.get_val()[i % self.dim]:
                previous = current
                rorl = 'r'
                current = current.get_rc()
            i += 1
        return

    def rebuild(self, data = None):
        self.root = None
        temp_data = np.copy(self.data_array) if data is None else data
        count = 0
        self.root, ld, rd = self._find_med(count, temp_data)
        parent = self.root
        self._rebuild_lc(parent, count + 1, ld)
        self._rebuild_rc(parent, count + 1, rd)
        return self.root

    def get_tree(self):
        return self.root

    def get_data(self):
        return self.data_array

    # def k_nearest(self, k, target):
    #     heap = []
    #     current = self.root
    #     parents = []
    #     found = False
    #     count = 0
    #     while not found:
    #         if current.get_val()[count % self.dim] <= target[count % self.dim]:
    #             parents.append(current)
    #             current = current.get_lc()
    #         else:
    #             parents.append(current)
    #             current = current.get_rc()
    #         if current is None:
    #             found = True
    #     rect = parents[-1]
    #     return

    def _maintain_heap(self, heap, new):
        return

    def visualization(self, node, pref=''):
        print(pref + str(node.get_val()))
        if node.get_lc() is not None:
            self.visualization(node.get_lc(), ' '*len(pref)+'l---')
        if node.get_rc() is not None:
            self.visualization(node.get_rc(), ' '*len(pref)+'r---')
        return

    def _rebuild_lc(self, parent, count, data):
        if len(data) == 1:
            parent.add_lc(Node(data[0]))
            return
        elif len(data) == 0:
            return
        else:
            node, ld, rd = self._find_med(count, data)
            parent.add_lc(node)
            self._rebuild_lc(node, count + 1, ld)
            self._rebuild_rc(node, count + 1, rd)
            return

    def _rebuild_rc(self, parent, count, data):
        if len(data) == 1:
            parent.add_rc(Node(data[0]))
            return
        elif len(data) == 0:
            return
        else:
            node, ld, rd = self._find_med(count, data)
            parent.add_rc(node)
            self._rebuild_lc(node, count + 1, ld)
            self._rebuild_rc(node, count + 1, rd)
            return

    def _find_med(self, count, data):
        comp = data[:, count % self.dim]
        order = np.argsort(comp, kind='mergesort')
        med_i = order[len(order) // 2]
        node = Node(data[med_i])

        left_data = data[order[:len(order) // 2]]
        right_data = data[order[len(order) // 2 + 1:]]

        return node, np.array(left_data), np.array(right_data)

def test_kdt_1():
    print("------test 1------")
    tree = CGW_KD_Tree(2)
    root = tree.rebuild(data)
    tree.visualization(root)

def test_kdt_2():
    print("------test 2------")
    tree = CGW_KD_Tree(2)
    for d in data:
        tree.insert(d)
    root = tree.get_tree()
    tree.visualization(root)

def test_kdt_3():
    print("------test 3------")
    tree = CGW_KD_Tree(2)
    for d in data:
        tree.insert(d)
    np.random.shuffle(data)
    tree.rebuild(data)
    root = tree.get_tree()
    tree.visualization(root)

# data=np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
# data = np.array([[51, 75], [70, 70], [55, 1], [60, 80], [10, 30], [35, 90], [25, 40], [1, 10], [50, 50]])
data = np.array([(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)])
test_kdt_1()
test_kdt_2()
test_kdt_3()

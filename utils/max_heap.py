import numpy as np
class HeapNode:
    def __init__(self, seq, ind, dist):
        self.seq = seq
        self.ind = ind
        self.dist = dist
        return

    def get_seq(self):
        return self.seq

    def get_dist(self):
        return self.dist

    def get_ind(self):
        return self.ind

class MaxHeap:
    def __init__(self):
        self.heap_list = [0]
        self.current_size = 0

    def perc_up(self,i):
        while i // 2 > 0:
            if self.heap_list[i].get_dist() > self.heap_list[i // 2].get_dist():
                tmp = self.heap_list[i // 2]
                self.heap_list[i // 2] = self.heap_list[i]
                self.heap_list[i] = tmp
            i = i // 2

    def insert(self, seq, ind, dist):
        k = HeapNode(seq, ind, dist)
        self.heap_list.append(k)
        self.current_size += 1
        self.perc_up(self.current_size)

    def perc_down(self,i):
        while (i * 2) <= self.current_size:
            mc = self.max_child(i)
            if self.heap_list[i].get_dist() < self.heap_list[mc].get_dist():
                tmp = self.heap_list[i]
                self.heap_list[i] = self.heap_list[mc]
                self.heap_list[mc] = tmp
            i = mc

    def max_child(self,i):
        if i * 2 + 1 > self.current_size:
            return i * 2
        else:
            if self.heap_list[i*2].get_dist() > self.heap_list[i*2+1].get_dist():
                return i * 2
            else:
                return i * 2 + 1

    def del_max(self):
        retval = self.heap_list[1]
        self.heap_list[1] = self.heap_list[self.current_size]
        self.current_size -= 1
        self.heap_list.pop()
        self.perc_down(1)
        return retval

    def build_heap(self,alist):
        i = len(alist) // 2
        self.current_size = len(alist)
        self.heap_list = [0] + alist[:]
        while (i > 0):
            self.perc_down(i)
            i = i - 1

    def get_size(self):
        return self.current_size

    def get_max_dist(self):
        if self.current_size > 0:
            return self.heap_list[1].get_dist()
        else:
            return np.inf

    def get_heap(self):
        seq, dist, inds = [], [], []
        if self.current_size > 0:
            for hnode in self.heap_list[1:]:
                seq.append(hnode.get_seq())
                inds.append(hnode.get_ind())
                dist.append(hnode.get_dist())
        return np.array(seq), np.array(inds, dtype=int), np.array(dist)

# def test1():
#     bh = MaxHeap()
#     blist = []
#     for i in range(len(alist)):
#         blist.append(HeapNode(alist[i], i, alist[i]+1))
#
#     bh.build_heap(blist)
#     print(bh.get_heap())
#
#     for _ in range(len(blist)):
#         n = bh.del_max()
#         print(n.get_seq(), n.get_dist(), bh.get_max_dist())
#
# def test2():
#     bh = MaxHeap()
#     for i in range(len(alist)):
#         bh.insert(HeapNode(alist[i], i, alist[i]+1))
#     print(bh.get_heap())
#
# # alist = list(np.random.random(10))
# alist = [4,5,3,2,1,-3,2]
# test1()
# test2()

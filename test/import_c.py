




# import ctypes
# import numpy as np
#
# class kdhyperrect(ctypes.Structure):
#     _fields_ = [("dim", ctypes.c_int),
#                 ("min", ctypes.POINTER(ctypes.c_double)),
#                 ("max", ctypes.POINTER(ctypes.c_double))]
#
# class kdnode(ctypes.Structure):
#     pass
# kdnode._fields_ = [("pos", ctypes.POINTER(ctypes.c_double)),
#                    ("dir", ctypes.c_int),
#                    ("data", ctypes.POINTER(ctypes.c_void_p)),
#                    ("left", ctypes.POINTER(kdnode)),
#                    ("right", ctypes.POINTER(kdnode))]
#
# class kdtree(ctypes.Structure):
#     _fields_ = [("dim", ctypes.c_int),
#                 ("root", ctypes.POINTER(kdnode)),
#                 ("rect", ctypes.POINTER(kdhyperrect)),
#                 ("destr", ctypes.CFUNCTYPE(ctypes.c_void_p))]
#
# class res_node(ctypes.Structure):
#     pass
# res_node._fields_ = [("item", ctypes.POINTER(kdnode)),
#                      ("dist_sq", ctypes.c_double),
#                      ("next", ctypes.POINTER(res_node))]
#
# class kdres(ctypes.Structure):
#     _fields_ = [("tree", ctypes.POINTER(kdtree)),
#                 ("rlist", ctypes.POINTER(res_node)),
#                 ("riter", ctypes.POINTER(res_node)),
#                 ("size", ctypes.c_int)]
#
# # lib = ctypes.windll.LoadLibrary("../kdtree-0.5.6/kdtree.so")
# lib = ctypes.CDLL("../kdtree-0.5.6/kdtree.so")
#
# x = np.array([1,2,3,4])
# tree_pt = lib.kd_create(5)
# print(tree_pt)
#
# # lib.kd_insert(tree_pt, ctypes.c_void_p(x.ctypes.data), 0)
# lib.kd_insert(tree_pt, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 0)
# x += 1
# print("target", x)
#
# # res = lib.kd_nearest(tree_pt, ctypes.c_void_p(x.ctypes.data))
# # print(res)
#
# kd_nearest = lib.kd_nearest
# kd_nearest.restype = kdres
# # print(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
# res = kd_nearest(tree_pt, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
# print('size', res.size)
# print('dist', res.rlist.contents.dist_sq)
# # print('rlist->item', res.rlist.contents.item)
# kdn = res.rlist.contents.item
# print('rlist->contents', kdn.contents)
# lib.kd_free(tree_pt)
#
# print('***finish***')
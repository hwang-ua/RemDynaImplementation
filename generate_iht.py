import utils.tiles3 as tc
import pickle as pkl
import os

def generate():
    mem_size = 16
    if os.path.isfile("utils/iht/iht_"+str(mem_size)+".pkl"):
        print("File exists")
        return
    else:
        iht = tc.IHT(mem_size)
        with open("utils/iht/iht_"+str(mem_size)+".pkl", 'wb') as f:
            pkl.dump(iht, f)
        print("IHT saved")

generate()
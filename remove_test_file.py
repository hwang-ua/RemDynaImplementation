import os
os.chdir("temp_continuous/")
dir = os.listdir(".")
for d in dir:
    if not os.path.isfile(d):
        os.chdir(d)
        for f in os.listdir("."):
            if f[:4] == "test":
                os.remove(f)
        os.chdir("../")
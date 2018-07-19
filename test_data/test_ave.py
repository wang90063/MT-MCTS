import glob
import numpy as np
path = '100-20'

allFiles = glob.glob(path + "/*.txt")


data_list=[]
sum_reward=0
for file_ in allFiles:
    data = np.genfromtxt(file_)
    sum_reward+=data[1]
print(sum_reward/len(allFiles))

print()

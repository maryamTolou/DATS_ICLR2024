import glob
import re
import random
import numpy as np

random.seed(1)
upper_bound = 0.1
lower_bound = 1e-4
task_no = 18
folder_path = 'Burger/Data/Burger_full'
mat_files = glob.glob(folder_path + '/*.mat') 
all_nus = []
pattern = r'(?<=data_)[\d\.e+-]+(?=\.mat)'
for i, file_path in enumerate(mat_files):
    match = re.search(pattern, file_path)
    num_str = match.group(0)
    num_float = float(num_str)
    if num_float <=upper_bound and num_float>=lower_bound:
        all_nus.append(num_float)
    """if i % 4 == 0:
        match = re.search(pattern, file_path)
        num_str = match.group(0)
        num_float = float(num_str)
        self.nus.append(num_float)""" 
print(all_nus) 
all_nus.sort()
all_nus_2 = all_nus[1:-1]
train_nus = all_nus_2[::int(len(all_nus_2)/task_no)]
test_nus = random.sample(train_nus, int(0.4*task_no))
train_nus = [nu for nu in train_nus if nu not in test_nus] + [all_nus[0]]+  [all_nus[-1]]
train_nus.sort()
test_nus.sort()

print("train: ", train_nus)
print("test: ", test_nus)
print("done")
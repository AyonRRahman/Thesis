#ground truth depth histogram
import os 
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
def load_gt_depth(filename):
    gt_depth  = cv2.imread(filename)
    gt_depth = cv2.cvtColor(gt_depth, cv2.COLOR_BGR2GRAY)

    return gt_depth

# depth_dir = '/mundus/mrahman527/Thesis/data/scaled_and_cropped_depth'
depth_dir = '/home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D'
folders = os.listdir(depth_dir)
# all_depth = np.zeros((256, 464)).flatten()

# print(all_depth.shape)
all_counts = None
# for folder in folders:
    # depth_folder = f"{depth_dir}/{folder}"
for i, files in enumerate(tqdm(os.listdir(depth_dir))):
    if not '.png' in files:
        continue
    depth = load_gt_depth(os.path.join(depth_dir, files))
    counts, edges = np.histogram(depth,  bins=range(200))
    if all_counts is None:
        all_counts = counts
    else:
        all_counts+=counts
        # all_depth = np.hstack(( all_depth,depth.flatten()))
        # if i==100:
        #     break
    
# print(all_depth.max())
# counts, edges = np.histogram(all_depth,  bins=range(150))
plt.stairs(all_counts[1:], edges[1:], fill=True)
plt.xlabel('Depth(m)')
plt.ylabel('Counts')

plt.savefig('varos_histogram.png')
plt.show()
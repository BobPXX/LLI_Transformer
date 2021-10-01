import numpy as np
import os
NUM_CLASSES=1000
root_dir="/home/pan/Desktop/dataset/imagenet2012/"

num_classes=NUM_CLASSES
train_filename = []
train_labels = []
val_filename = []
val_labels = []

folders=os.listdir(root_dir+'train/')
folders.sort()
for f in range(NUM_CLASSES):
    files=os.listdir(root_dir+'train/'+folders[f]+'/')
    files.sort()
    for file in files:
        train_filename.append(root_dir+'train/'+folders[f]+'/'+file)
        train_labels.append(f)

file=root_dir+'imagenet_labels/ILSVRC2012_validation_ground_truth.txt'
f = open(file, "r")
lines = f.readlines()
val_ground_truth = []
for x in lines:
    val_ground_truth.append(x.split(' ')[0][:-1])
f.close()

file=root_dir+'imagenet_labels/ILSVRC2012_mapping.txt'
f = open(file, "r")
lines = f.readlines()
mapping1 = []
mapping2 = []
for x in lines:
    mapping1.append(x.split(' ')[0])
    mapping2.append(x.split(' ')[1][:-1])
f.close()

images=os.listdir(root_dir+'val/')
images.sort()
for image in images:
    folder_name=mapping2[mapping1.index(val_ground_truth[int(image[-10:-5])-1])]
    f=folders.index(folder_name)
    val_filename.append(root_dir+'val/'+image)
    val_labels.append(f)

np.save(root_dir+'train_filename.npy',train_filename)
np.save(root_dir + 'train_labels.npy', train_labels)
np.save(root_dir+'val_filename.npy',val_filename)
np.save(root_dir + 'val_labels.npy', val_labels)
    

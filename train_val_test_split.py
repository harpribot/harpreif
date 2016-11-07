import sys
import os
import glob
from random import shuffle
import shutil

args = sys.argv

caltech_im_dir = args[1]

train_dir = args[2]
val_dir = args[3]
test_dir = args[4]
transfer_test_dir = args[5]

train_im = []
val_im = []
test_im = []
transfer_test_im = []

objects = os.listdir(caltech_im_dir)
shuffle(objects)

# Use 240 objects to divide into train/val/test
for object in objects[:-16]:
    object_path = caltech_im_dir + '/' + object
    object_ims = glob.glob(object_path + '/*.jpg')
    # shuffle images
    shuffle(object_ims)
    # take the top 80 images
    object_ims = object_ims[:80]
    # Now split the images into 50 for training, 10 for validation, and 20 for testing
    train_im.extend(object_ims[:50])
    val_im.extend(object_ims[50:60])
    test_im.extend(object_ims[60:])


# use the 16 object classes to check for transfer learning
for object in objects[-16:]:
    object_path = caltech_im_dir + '/' + object
    object_ims = glob.glob(object_path + '/*.jpg')
    # shuffle images
    shuffle(object_ims)
    # take 30 images for transfer testing
    object_ims = object_ims[:30]
    transfer_test_im.extend(object_ims)

# Now store the images in the appropriate directory
for im in train_im:
    shutil.move(im, train_dir)

for im in val_im:
    shutil.move(im, val_dir)

for im in test_im:
    shutil.move(im, test_dir)

for im in transfer_test_im:
    shutil.move(im, transfer_test_dir)





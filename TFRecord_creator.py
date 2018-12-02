from random import shuffle
import glob
import tensorflow as tf
import numpy as np
import cv2
import sys

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


label_dic = {

    'auditorium':        0,
    'bedroom':           1,
    'cafeteria':         2,
    'classroom':         3,
    'closet':            4,
    'conference_center': 5,
    'conference_room':   6,
    'corridor':          7,
    'courtyard':         8,
    'dinette':           9,
    'dining_room':      10,
    'dorm_room':        11,
    'home_office':      12,
    'kitchen':          13,
    'living_room':      14,
    'locker_room':      15,
    'office':           16,
    'shower':           17,
    'staircase':        18,
    'swimming_pool':    19

}

shuffle_data = True  # shuffle the addresses before saving
data_path = 'D:\\imagesPlaces205_resize2\\*\\*.jpg'

# read addresses and labels from the 'train' folder
addrs = glob.glob(data_path)
labels = []

for address in addrs:
    for label in label_dic:
        if label in address:
            labels.append(label_dic[label])

# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.00084 * len(addrs))]
train_labels = labels[0:int(0.00084 * len(labels))]

val_addrs = addrs[int(0.00084 * len(addrs)):int(0.00111 * len(addrs))]
val_labels = labels[int(0.00084 * len(addrs)):int(0.00111 * len(addrs))]

test_addrs = addrs[int(0.00111 * len(addrs)):]
test_labels = labels[int(0.00111 * len(labels)):]

train_filename = 'train_prueba.tfrecords'  # address to save the TFRecords file
# # open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

# test_filename = 'test.tfrecords'  # address to save the TFRecords file
# # # open the TFRecords file
# writer = tf.python_io.TFRecordWriter(test_filename)
# for i in range(len(test_addrs)):
#     # print how many images are saved every 1000 images
#     if not i % 1000:
#         print('Test data: {}/{}'.format(i, len(test_addrs)))
#         sys.stdout.flush()
#     # Load the image
#     img = load_image(test_addrs[i])
#     label = test_labels[i]
#     # Create a feature
#     feature = {'test/label': _int64_feature(label),
#                'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())
#
# writer.close()
# sys.stdout.flush()

validation_filename = 'val_prueba.tfrecords'  # address to save the TFRecords file
# # open the TFRecords file
writer = tf.python_io.TFRecordWriter(validation_filename)
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Validation data: {}/{}'.format(i, len(val_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(val_addrs[i])
    label = val_labels[i]
    # Create a feature
    feature = {'val/label': _int64_feature(label),
               'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

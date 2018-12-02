import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Function to preprocess the image data
def parse_fn(example):
    example_fmt = {
        "train/image": tf.FixedLenFeature((), tf.string, ""),
        "train/label": tf.FixedLenFeature((), tf.int64, -1)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.decode_raw(parsed["train/image"], tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.image.resize_images(image, [80, 80])
    image = tf.cast(image, tf.uint8)

    #index = parsed['train/label']
    #index_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    #outputs_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #outputs_array[index.eval(session=tf.Session())] = 1
    #image = tf.reshape(image, [19200])
    #image = tf.as_string(image)
    return parsed['train/label'], image

#Function to get the dataset object from tf.records file
def input_fn():
  dataset = tf.data.TFRecordDataset(".\\train.tfrecords")
  dataset = dataset.shuffle(buffer_size=dataset_params['buffer'])
  dataset = dataset.map(map_func=parse_fn)
  dataset = dataset.batch(batch_size=dataset_params['batch'])
  dataset = dataset.prefetch(buffer_size=dataset_params['prefetch_buffer'])
  return dataset

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


dataset_params = {
    'buffer': 30,
    'batch': 30,
    'prefetch_buffer': 2
}

example_fmt = {
    "train/image": tf.FixedLenFeature([], tf.string),
    "train/label": tf.FixedLenFeature([], tf.int64)
}





with tf.Session() as sess:
    iter = input_fn().make_one_shot_iterator()
    places_batch = sess.run(iter.get_next())

    #COMMENT WHEN PREPROCESSING IS NEEDED
    label_index = 0
    image_index = 1

    # for label in places_batch[label_index]:
    #    print(label)
    placeMatrix = places_batch[1][1]
    placeMatrix = placeMatrix.astype(np.uint8)
    plt.imshow(placeMatrix)
    plt.title(places_batch[0][1])
    plt.show()

    #UNCOMMENT WHEN NO PREPROCESSING IS NEEDED
    # for place in places_batch:
    #     parse_place = tf.parse_single_example(place, example_fmt)
    #     placeImg = tf.decode_raw(parse_place['train/image'], tf.float32)
    #     placeMatrix = placeImg.eval().reshape([80, 80, 3])
    #     placeMatrix = placeMatrix.astype(np.uint8)
    #     plt.imshow(placeMatrix)
    #     plt.show()
    #     time.sleep(1)



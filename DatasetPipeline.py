import tensorflow as tf

class DatasetPipeline:

    def __init__(self, DICT_DATASET_PARAMS):
        self.DICT_DATASET_PARAMS = DICT_DATASET_PARAMS
        self.dataset = self.dataset_generator()
        self.iterator = self.set_iterator_mode(DICT_DATASET_PARAMS["iterator_mode"])

    def dataset_generator(self):
        dataset = tf.data.TFRecordDataset(self.DICT_DATASET_PARAMS["path_file"])
        dataset = dataset.shuffle(buffer_size=self.DICT_DATASET_PARAMS['buffer_size'])
        dataset = dataset.map(map_func=self.image_data_preprocess)
        dataset = dataset.batch(batch_size=self.DICT_DATASET_PARAMS['batch_size'])
        dataset = dataset.prefetch(buffer_size=self.DICT_DATASET_PARAMS['prefetch_number'])
        return dataset

    def image_data_preprocess(self, example):

        image_dict_name = self.DICT_DATASET_PARAMS["dataset_name"] + "/image"
        label_dict_name = self.DICT_DATASET_PARAMS["dataset_name"] + "/label"

        example_fmt = {
            image_dict_name: tf.FixedLenFeature((), tf.string, ""),
            label_dict_name: tf.FixedLenFeature((), tf.int64, -1)
        }

        parsed = tf.parse_single_example(example, example_fmt)
        image = tf.decode_raw(parsed[image_dict_name], tf.float32)
        image = tf.reshape(image, [self.DICT_DATASET_PARAMS["image_original_size"], self.DICT_DATASET_PARAMS["image_original_size"], self.DICT_DATASET_PARAMS["image_dim"]])
        image = tf.image.resize_images(image, [self.DICT_DATASET_PARAMS["image_size"], self.DICT_DATASET_PARAMS["image_size"]])
        image = tf.cast(image, tf.float32) * (1. / 255)
        return parsed[label_dict_name], image

    def set_iterator_mode(self, mode_one_shot):
        return self.dataset.make_one_shot_iterator() if mode_one_shot else self.dataset.make_initializable_iterator()

    def update_batch(self, sess):
        outputs_array = []
        output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        batch = sess.run(self.iterator.get_next())
        for index in range(len(batch[0])):
            output[batch[0][index]] = 1
            outputs_array.append(output)
            output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        return batch[1], outputs_array

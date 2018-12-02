import tensorflow as tf
import NetworkCNN as cnn
import DatasetPipeline as pipeline
import RecorderCSV as rec
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4" #for training on gpu

DICT_CNN = {
    "CNN_2": {"conv_number": 5,
    "image_dim": 3,
    "conv_dimensions": [48, 128, 192, 192, 128],
    "conv_kernels": [11, 5, 3, 3, 3],
    "dense_number": 2,
    "image_size": 224,
    "max_pooling_kernel": 2,
    "dense_dimensions": [3000, 3000],
    "number_classes": 20,
    "learning_rate": 0.001},
}

DICT_DATASET_VAL = {
    "iterator_mode": False,
    "path_file": "../cnn_indoorplaces/dataset/val.tfrecords",
    "buffer_size": 1,
    "batch_size": 128,
    "prefetch_number": 2,
    "dataset_name": "val",
    "image_original_size": 224,
    "image_size": 224,
    "image_dim": 3
}

DICT_DATASET_TRAIN = {
    "iterator_mode": False,
    "path_file": "../cnn_indoorplaces/dataset/train.tfrecords",
    "buffer_size": 1,
    "batch_size": 128,
    "prefetch_number": 2,
    "dataset_name": "train",
    "image_original_size": 224,
    "image_size": 224,
    "image_dim": 3
}

DICT_TRAINING_PARAMS = {
    "epochs": 3000
}

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

with tf.device("/cpu:0"):
    dict_cpu_var = {
        "list_actual_pred": [],
        "list_pred": [],
        "acc_val": 0,
        "loss_val": 0,
        "acc_train": 0,
        "loss_train": 0,
        "number_data": 0
    }

try:

    for CNN_ARQ in DICT_CNN:
        tf.reset_default_graph()
        net = cnn.NetworkCNN(DICT_CNN[CNN_ARQ])
        net.weight_biases_creator()
        net.set_training_algorithms()

        pipe_train = pipeline.DatasetPipeline(DICT_DATASET_TRAIN)
        pipe_val = pipeline.DatasetPipeline(DICT_DATASET_VAL)

        recorder_file_name = "validationRecord_" + str(DICT_CNN[CNN_ARQ]["conv_number"]) + "_" + str(DICT_CNN[CNN_ARQ]["dense_number"]) + "_"
        recorder_acc_loss_val = rec.RecorderCSV(recorder_file_name)
        recorder_acc_loss_val.create_file(["Loss", "Acc", "Epoch"])

        recorder_file_name = "trainingRecord_" + str(DICT_CNN[CNN_ARQ]["conv_number"]) + "_" + str(DICT_CNN[CNN_ARQ]["dense_number"]) + "_"
        recorder_acc_loss_train = rec.RecorderCSV(recorder_file_name)
        recorder_acc_loss_train.create_file(["Loss", "Acc", "Epoch"])

        recorder_file_name = "confussionMatrix_" + str(DICT_CNN[CNN_ARQ]["conv_number"]) + "_" + str(DICT_CNN[CNN_ARQ]["dense_number"]) + "_"
        recorder_conf = rec.RecorderCSV(recorder_file_name)
        recorder_conf.create_file(label_dic.keys())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(pipe_train.iterator.initializer)
            sess.run(pipe_val.iterator.initializer)

            for i in range(DICT_TRAINING_PARAMS["epochs"]):
                net.training_mode = True
                while True:
                    try:
                        batch_updated = pipe_train.update_batch(sess=sess)
                        net.train_network(sess=sess, batch=batch_updated)
                        [loss, acc, pred] = net.update_indicators(sess=sess, batch=batch_updated)
                        dict_cpu_var.update({"number_data": dict_cpu_var["number_data"] + 1})
                        print("----------Train Batch Results---------\nLoss = %f\nAcc = %f\nEpoch = %i\n-----------------------\n" % (loss, acc, i))
                        dict_cpu_var.update({"acc_train": dict_cpu_var["acc_train"] + acc})
                        dict_cpu_var.update({"loss_train": dict_cpu_var["loss_train"] + loss})
                    except tf.errors.OutOfRangeError:
                        sess.run(pipe_train.iterator.initializer)
                        dict_cpu_var.update({"acc_train": dict_cpu_var["acc_train"]/dict_cpu_var["number_data"]})
                        dict_cpu_var.update({"loss_train": dict_cpu_var["loss_train"]/dict_cpu_var["number_data"]})
                        recorder_acc_loss_train.append_into_file([dict_cpu_var["loss_train"], dict_cpu_var["acc_train"], i])
                        print("----------Training Results---------\nLoss = %f\nAcc = %f\nEpoch = %i\n-----------------------\n" % (dict_cpu_var["loss_train"], dict_cpu_var["acc_train"], i))
                        dict_cpu_var.update({"number_data": 0})
                        dict_cpu_var.update({"acc_train": 0})
                        dict_cpu_var.update({"loss_train": 0})
                        net.training_mode = False
                        while True:
                            try:
                                batch_updated = pipe_val.update_batch(sess=sess)
                                [loss, acc, pred] = net.update_indicators(sess=sess, batch=batch_updated)
                                dict_cpu_var.update({"number_data": dict_cpu_var["number_data"] + 1})
                                dict_cpu_var.update({"acc_val": dict_cpu_var["acc_val"] + acc})
                                dict_cpu_var.update({"loss_val": dict_cpu_var["loss_val"] + loss})
                                print("----------Val Batch Results---------\nLoss = %f\nAcc = %f\nEpoch = %i\n-----------------------\n" % (dict_cpu_var["loss_val"], dict_cpu_var["acc_val"], dict_cpu_var["number_data"]))
                                pred_softmax = sess.run(tf.nn.softmax(pred))

                                with tf.device("/cpu:0"):
                                    for correct_label in batch_updated[1]:
                                        dict_cpu_var["list_actual_pred"].append(correct_label.index(max(correct_label)))

                                    for pred_label in pred_softmax:
                                        dict_cpu_var["list_pred"].append(pred_label.tolist().index(max(pred_label)))

                            except tf.errors.OutOfRangeError:
                                sess.run(pipe_val.iterator.initializer)
                                dict_cpu_var.update({"acc_val": dict_cpu_var["acc_val"]/dict_cpu_var["number_data"]})
                                dict_cpu_var.update({"loss_val": dict_cpu_var["loss_val"]/dict_cpu_var["number_data"]})
                                recorder_acc_loss_val.append_into_file([dict_cpu_var["loss_val"], dict_cpu_var["acc_val"], i])
                                print("----------Validation Results---------\nLoss = %f\nAcc = %f\nEpoch = %i\n-----------------------\n" % (dict_cpu_var["loss_val"], dict_cpu_var["acc_val"], dict_cpu_var["number_data"]))
                                conf_matrix = sess.run(tf.confusion_matrix(dict_cpu_var["list_actual_pred"], dict_cpu_var["list_pred"], num_classes=DICT_CNN[CNN_ARQ]["number_classes"]))
                                for conf_item in conf_matrix:
                                    recorder_conf.append_into_file(conf_item)
                                dict_cpu_var.update({"number_data": 0})
                                dict_cpu_var.update({"acc_train": 0})
                                dict_cpu_var.update({"loss_train": 0})
                                dict_cpu_var.update({"list_actual_pred": []})
                                dict_cpu_var.update({"list_pred": []})
                                break
                        break


        print("Training finished successfully")

except KeyboardInterrupt:
    print("Training stoped")
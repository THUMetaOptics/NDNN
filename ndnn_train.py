'''
MAIN TRAINING CODE
'''
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import mask_modulation_model as mmm
import data_generation as dtg
import initialization as init
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix
from pylab import np
#configure a tensorflow session
config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
session = tf.Session(config=config)
gpu_options = tf.GPUOptions(allow_growth=True)

#set environment variables
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"



# import and globalize parameters
tc = init.init_params()
print(tf.config.list_physical_devices('GPU'))


#gpu-computing configuration
def get_default_config(fraction=0.9):
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = fraction
    conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    return conf


# list of monitoring variables
global_step = 0
best_loss = 1e12
best_accuracy = 0

if __name__ == '__main__':
    
    conf = get_default_config()
    sess = tf.InteractiveSession(config=conf)

    with tf.name_scope("datasets"):
        # define data pipeline, initialize iterator
        batch_train = dtg.get_data_batch('training')
        batch_test = dtg.get_data_batch('validation')
        batch_final = dtg.get_data_batch('testing')

        # define the iterator
        iterator = tf.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(batch_train), tf.compat.v1.data.get_output_shapes(batch_train))
        batch = iterator.get_next()
        data_amp, data_phase, sensor_gt, data_label = batch
        onn_field = tf.complex(data_amp * tf.cos(data_phase), data_amp * tf.sin(data_phase))

    # define the model
    for i in range(tc.layer):
        layer=tc.MASK_NUMBER[i]
        onn_measurement, onn_mask_phase, onn_mask_amp, onn_logits= mmm.inferencenl(onn_field,layer)
        onn_predictions = tf.nn.softmax(onn_logits, name = 'predictions')
        onn_loss = mmm.loss_function(onn_measurement, sensor_gt)
        onn_hit = tf.reduce_sum(tf.cast(tf.equal(tf.cast((data_label),dtype=tf.int64), tf.argmax(onn_predictions, axis=1)), dtype=tf.int64))
        with tf.variable_scope("scope_name", reuse=tf.AUTO_REUSE):
            onn_train = tf.train.AdamOptimizer(learning_rate=tc.LEARNING_RATE).minimize(onn_loss)

        #initialization
        init_all = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()
        sess.run(init_all)
        print("Entering training loop")

        for epoch in range(1,tc.MAX_EPOCH+1):
            sess.run(iterator.make_initializer(batch_train))
            train_step = input_count = hit_count_train = total_loss_train = 0
            turn = 0
            while True:
            #define a dead loop that keeps looping until it goes out of range
                try:
                    _,  onn_loss_value, onn_predictions_value, hit_count ,label, phase= sess.run([onn_train, onn_loss,  onn_logits, onn_hit, data_label,data_phase])
                    train_step += 1
                    global_step += 1
                    input_count += tc.BATCH_SIZE
                    hit_count_train += hit_count
                    total_loss_train += onn_loss_value
                except (tf.errors.OutOfRangeError, StopIteration):
                    break
            accuracy_train = hit_count_train/input_count*100
            mean_loss_train = total_loss_train/train_step


            #initialize iterator for validation dataset. No need to shuffle
            sess.run(iterator.make_initializer(batch_test))
            test_step = test_count = hit_count_test = total_loss_test = 0
            while True:
                try:
                    # run test iteration
                    onn_loss_value_test, hit_count, prediction, true= sess.run([onn_loss,  onn_hit, onn_logits, data_label])
                    test_step += 1
                    test_count += tc.BATCH_SIZE
                    hit_count_test += hit_count
                    total_loss_test += onn_loss_value_test
                    confusion_mat = confusion_matrix(true, np.argmax(prediction, axis=1))
                except (tf.errors.OutOfRangeError, StopIteration):
                    break
       
            accuracy_test = hit_count_test/test_count*100
            mean_loss_test = total_loss_test/test_step
        
            #Log/Record
            msg = "epoch " + "mean_train_loss " + "mean_test_loss " + "training_accuracy " + "testing_accuracy " + str(datetime.now())
            print(msg)
            msg = str(epoch) + " " + str(mean_loss_train) + " " + str(mean_loss_test) + " " + str(accuracy_train) + " " + str(accuracy_test)
            print(msg)
            with tf.device('/cpu:0'):
                path_data = 'C:/Users/Administrator/PycharmProjects/pythonProject'
                log_file = open(path_data + 'draw_try_nl.txt', 'a+')
                print(msg,  file=log_file)
                log_file.close()

            if tc.APPLICATION == 'classification':
                save_model = bool(accuracy_test > best_accuracy)
            elif tc.APPLICATION == 'amplitude_imaging':
                save_model = bool(mean_loss_test < best_loss)
            elif tc.APPLICATION == 'phase_imaging':
                save_model = bool(mean_loss_test < best_loss)

            #saving model
            if save_model is True:
                save_path = saver.save(sess, "./MODEL/model{}".format(epoch))
                print("Model saved in path: %s" % save_path)
                best_loss = mean_loss_test
                best_accuracy = accuracy_test
                save_epoch = epoch

 
        print("Training Finished!")
        saver.restore(sess,"./MODEL/model{}".format(save_epoch))
        sess.run(iterator.make_initializer(batch_final))
        test_step = test_count = hit_count_test = total_loss_test = 0
        prediction = np.zeros((1,10))
        label = 0

        while True:
            try:
            # run train iteration
                onn_loss_value_test,  hit_count, label = sess.run([onn_loss,  onn_hit,data_label])
                test_step += 1
                test_count += tc.BATCH_SIZE
                hit_count_test += hit_count
                total_loss_test += onn_loss_value_test
            except (tf.errors.OutOfRangeError, StopIteration):
                break

        accuracy_test = hit_count_test/test_count*100
        mean_loss_test = total_loss_test/test_step


        msg = "epoch " + "mean_train_loss " + "mean_test_loss " + "training_accuracy " + "testing_accuracy " + str(datetime.now())
        print(msg)
        msg = str(epoch) + " " + str(mean_loss_train) + " " + str(mean_loss_test) + " " + str(accuracy_train) + " " + str(accuracy_test)
        print(msg)

print("Finished!")



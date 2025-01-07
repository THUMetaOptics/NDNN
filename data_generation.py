'''
Data Generation
'''
import numpy as np
import data_sensor as dto
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import initialization as init
import os
import numpy as np
from PIL import Image

tc = init.init_params()

#import tensorflow datasets
"""if tc.TRAINING_DATA_TYPE == 'mnist' or tc.TESTING_DATA_TYPE == 'mnist':
    data = tf.keras.datasets.mnist.load_data()

if tc.TRAINING_DATA_TYPE == 'cifar-10' or tc.TESTING_DATA_TYPE == 'cifar-10':
    data = tf.keras.datasets.cifar10.load_data()

if tc.TRAINING_DATA_TYPE == 'fashion-mnist' or tc.TESTING_DATA_TYPE == 'fashion-mnist':
    data = tf.keras.datasets.fashion_mnist.load_data()

train, test = data
data_train, label_train = train
data_test, label_test = test"""

dataset_path = r'C:\Users\Administrator\PycharmProjects\pythonProject\draw\draw\dataset'

# get all class folders
class_folders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

# initialize data and label lists
data0 = []
label0 = []


for label, folder in enumerate(class_folders):
    image_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg')]
    
    # get all image files and add to dataset
    for image_file in image_files:
        image = Image.open(image_file).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image, dtype=np.uint8)
        data0.append(image_array)
        label0.append(label)

# convert to numpy arrays
data = np.array(data0)
label = np.array(label0, dtype=np.uint8)

# shuffle data and label synchronously
indices = np.random.permutation(len(data))
data_shuffled = data[indices]
label_shuffled = label[indices]

# split the data and label
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
data_train = data_shuffled[:split_index]
data_test = data_shuffled[split_index:]
label_train = label_shuffled[:split_index]
label_test = label_shuffled[split_index:]

#create valid dataset and train dataset
global_norm = np.amax([np.amax(np.abs(data_train)), np.amax(np.abs(data_test))])
indices_valid = dto.create_validation(label_train, label_test, tc.validation_ratio)
data_valid = data_train[indices_valid]
label_valid = np.squeeze(label_train[indices_valid])
N_VALID = np.amax(label_valid.shape)
data_train =np.delete(data_train, indices_valid, axis=0)
label_train = np.delete(label_train, indices_valid)

#initialize tensors to store classification information and expectation
gts = tf.zeros((tc.NUM_CLASS, int(tc.out_M), int(tc.out_N)),dtype=tf.float32)
expectation = tf.zeros((tc.M, tc.N),dtype=tf.float32)

#define detect region
for i in range(tc.NUM_CLASS):
    gt_i  = dto.gt_generator_classification(i)
    gt_p = tf.expand_dims(gt_i, 0)
    gts = tf.add(gts, tf.scatter_nd([[i]], gt_p, shape=[tc.NUM_CLASS, int(tc.out_M), int(tc.out_N)]))
    gts_tensor = gts

#prepocess input data
def _preprocess(img, label):
    img_r = tf.reshape(img, [1,tc.DATA_ROW, tc.DATA_COL,1])
    img_r = tf.image.resize(img_r,[tc.RM,tc.RN])
    img_r = tf.squeeze(img_r)
    img_r = tf.divide(img_r,global_norm)
    padx = int((tc.M - tc.RM) / 2)
    pady = int((tc.N - tc.RN) / 2)
    img_pad = tf.pad(img_r, [(padx, padx), (pady, pady)], 'constant')
    label = tf.squeeze(tf.cast(label,dtype=tf.int64))
    gt = tf.squeeze(tf.slice(gts_tensor, [label, 0, 0,], [1, int(tc.out_M), int(tc.out_N)]))
    xi = tf.constant((np.arange(tc.M) - tc.M / 2), shape=[tc.M, 1], dtype=tf.float32)
    x22 = tf.matmul(xi ** 2, tf.ones((1, tc.N), dtype=tf.float32))
    y = tf.constant((np.arange(tc.N) - tc.N / 2), shape=[1, tc.N], dtype=tf.float32)
    y22 = tf.matmul(tf.ones((tc.N, 1), dtype=tf.float32), y ** 2)
    if tc.OBJECT_AMPLITUDE_INPUT is True:
        xishu = tf.ones((tc.M, tc.N), dtype=tf.float32) * 2
        img_amp = 2*tf.cast(img_pad*100* tf.exp(-tf.sqrt(x22 + y22) * tf.sqrt(xishu) / 100),dtype=tf.float32)
        img_phase = tf.ones((tc.M,tc.N),dtype=tf.float32)*np.pi
    elif tc.OBJECT_PHASE_INPUT is True:
        img_amp = tf.ones((tc.M,tc.N),dtype=tf.float32)
        img_phase = 0.999 * np.pi * img_pad/(tf.reduce_max(img_pad))
    return img_amp, img_phase, gt,label


def get_data_batch(request_type):
    if request_type == 'training':
        get_batch = tf.data.Dataset.from_tensor_slices((data_train, label_train))
        get_batch = get_batch.shuffle(tc.NUMBER_TRAINING_ELEMENTS-N_VALID)
        
    elif request_type == 'testing':
        get_batch = tf.data.Dataset.from_tensor_slices((data_test, label_test))
        get_batch = get_batch.shuffle(tc.NUMBER_TEST_ELEMENTS)
        
    elif request_type == 'validation':
        get_batch = tf.data.Dataset.from_tensor_slices((data_valid, label_valid))
        get_batch = get_batch.shuffle(N_VALID)

    #data preprocess, extract data in batch sizes, and define the buffer size
    get_batch = get_batch.map(_preprocess,4)
    get_batch = get_batch.prefetch(buffer_size=tc.BATCH_SIZE)
    get_batch = get_batch.batch(tc.BATCH_SIZE, drop_remainder = True)
    return get_batch




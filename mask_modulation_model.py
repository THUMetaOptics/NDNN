'''
Mask Modulation Model
'''

import tensorflow.compat.v1 as tf
import os
import numpy as np
import tf_OpticsModule_draw as tom
import initialization_draw as init
import data_ops_draw as dto

tc = init.init_params()

def read_mask(path, L, mask_type):
    "read trained masks"
    mask_list = os.listdir(path)
    if mask_type == 'amplitude':
        mask = np.loadtxt(path + "/" + mask_list[L])
    elif mask_type == 'phase':
        mask = np.loadtxt(path + "/" + mask_list[L])
    mask = tf.cast(mask, tf.float32)
    return mask


def phase_NL(x):
    "nonlinear activation function"
    inten = tf.multiply(tf.abs(x), tf.abs(x))
    phase = tf.angle(x)
    act = 2*np.pi/tf.cast(16.25,dtype=tf.float32)*inten     #16.25 is derived from nonlinear refractive index
    phase_act = phase+act
    R = 0.9                                                 #for nonlinear situation: R = 0.9*tf.exp(-inten/n),n is derived from TPA coefficient
    amp = tf.sqrt(0.9*R)
    hidden = tf.complex(amp * tf.cos(phase_act), amp * tf.sin(phase_act))
    return hidden


def mask_init(masknum):
    "create initialized masks"
    if tc.MASK_PHASE_MODULATION is True:
        if tc.MASK_INIT_TYPE == 'const':
            with tf.variable_scope("scope_name", reuse=tf.AUTO_REUSE):
                mask_phase_org = tf.get_variable('mask_phase' + str(masknum),initializer=tf.constant(1.0, shape=[tc.mod_M, tc.mod_N]))
                mask_phase = tf.sigmoid(mask_phase_org)* np.pi
                paddings=int((tc.M *2 -tc.mod_M)/2)
                mask_phase = tf.pad(mask_phase, [(paddings,paddings),(paddings,paddings)], mode='CONSTANT', constant_values=0,name=None)
                mask_amp = tf.ones([ tc.M*2,  tc.N*2])
        elif tc.MASK_INIT_TYPE == 'trained':  # 带入训练好的参数
                mask_phase = read_mask(tc.MASK_PATH, masknum, 'phase')
                mask_amp = tf.ones([tc.M*2,  tc.N*2])
        else:
                mask_phase = tf.zeros([tc.MASK_ROW, tc.MASK_COL])
        return mask_phase, mask_amp

def detector_plane(measurement,cell_locs,thecell):
    "Compute the class probabilities for each cell location in the measurement data."
    kk = 0
    ulcorner = cell_locs[kk,0:2]
    probs = tf.reduce_mean(tf.slice(measurement,[0,ulcorner[0],ulcorner[1]],[tc.BATCH_SIZE,thecell.shape[0],thecell.shape[1]]),axis=[1,2])

    class_probs = tf.expand_dims(probs,-1)
    for kk in range(1,cell_locs.shape[0]):
        ulcorner = cell_locs[kk,0:2]
        probs = tf.reduce_mean(tf.slice(measurement,[0,ulcorner[0],ulcorner[1]],[tc.BATCH_SIZE,thecell.shape[0],thecell.shape[1]]),axis=[1,2])
        probs = tf.expand_dims(probs,-1)
        class_probs = tf.concat([class_probs,probs],axis=1)
    return class_probs


def inference(field,layer):
    with tf.name_scope('first-layer'):
        with tf.name_scope('propagation'):
                "create an input light with a gaussian intensity distribution and a uniform phase"
                xi = tf.constant((np.arange(tc.mod_M) - tc.mod_M / 2), shape=[tc.mod_M, 1], dtype=tf.float32)
                x22 = tf.matmul(xi ** 2, tf.ones((1, tc.mod_N), dtype=tf.float32))
                y = tf.constant((np.arange(tc.mod_N) - tc.mod_N / 2), shape=[1, tc.mod_N], dtype=tf.float32)
                y22 = tf.matmul(tf.ones((tc.mod_N, 1), dtype=tf.float32), y ** 2)
                img_amp = tf.ones((tc.mod_M, tc.mod_N), dtype=tf.float32)
                xishu = tf.ones((tc.mod_M, tc.mod_N), dtype=tf.float32) * 2
                input_amp = tc.input_amp * tf.cast(img_amp * tf.exp(-tf.sqrt(x22 + y22)*tf.sqrt(xishu)/100), dtype=tf.float32)
                input_phase = 0.999 * np.pi * tf.ones((tc.mod_M, tc.mod_N), dtype=tf.float32)
                input_field = tf.complex(input_amp * tf.cos(input_phase), input_amp * tf.sin(input_phase))

                "create a group of gaussian beam"
                input_field_batch = tf.expand_dims(input_field, 0)
                for number in range(1, tc.BATCH_SIZE):
                    input_field_batch = tf.concat([input_field_batch, tf.expand_dims(input_field, 0)], 0)

                "propagate to input layer and carry input information"
                img_p = tom.tf_FSPAS_FFT(input_field_batch, tc.WLENGTH, tc.MASK_MASK_DISTANCE, tc.DX, tc.DY, tc.n_air)
                paddings = int((512-256)/2)
                fields = tf.pad(field, [(0,0), (paddings, paddings), (paddings, paddings)], mode='CONSTANT',constant_values=0, name=None)
                hidden = tf.multiply(img_p, fields)

        with tf.name_scope('mask'):
                "initialize masks"
                mask_phase, mask_amp = mask_init(0)
                mask = tf.complex(mask_amp * tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
                save_mask_phase = tf.expand_dims(mask_phase, 0)
                save_mask_amp = tf.expand_dims(mask_amp, 0)

        with tf.name_scope('propagation'):
                "first propagation and modulation"
                prop = tom.tf_FSPAS_FFT_mid(hidden, tc.WLENGTH, tc.MASK_Si_DISTANCE,tc.Si_thick, tc.DX, tc.DY, tc.n_air,tc.n_si)
                hidden = tf.multiply(prop, mask)

    if layer > 1:
        for layer_number in range(1,layer):
            with tf.name_scope('middle-layer-'+str(layer_number)):
                with tf.name_scope('mask'):
                    mask_phase, mask_amp = mask_init(layer_number)
                    mask = tf.complex(mask_amp * tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
                    save_mask_phase = tf.concat([save_mask_phase, tf.expand_dims(mask_phase, 0)], 0)
                    save_mask_amp = tf.concat([save_mask_amp, tf.expand_dims(mask_amp, 0)], 0)
            with tf.name_scope('propagation'):
                prop = tom.tf_FSPAS_FFT_mid(hidden, tc.WLENGTH, tc.MASK_Si_DISTANCE,tc.Si_thick, tc.DX, tc.DY, tc.n_air,tc.n_si)
                hidden = tf.multiply(prop, mask)

    with tf.name_scope('last-layer'):
        with tf.name_scope('propagation'):
                "propagate to sensor"
                img_p = tom.tf_FSPAS_FFT_last(hidden, tc.WLENGTH, tc.MASK_SENSOR_DISTANCE, tc.DX, tc.DY, tc.n_air)

        with tf.name_scope('sensor'):
                "calcuate the intensity distribution on the sensor plane and classification probability"
                measurement = tf.square(tf.abs(img_p))
                cell_locs, thecell, label_fields = dto.sensorplane_geometry()
                logits = detector_plane(measurement, cell_locs, thecell)
    return measurement, save_mask_phase, save_mask_amp, logits



def inferencenl(field,layer):
    with tf.name_scope('first-layer'):
        with tf.name_scope('propagation'):
                "create an input light with a gaussian intensity distribution and a uniform phase"
                xi = tf.constant((np.arange(tc.mod_M) - tc.mod_M / 2), shape=[tc.mod_M, 1], dtype=tf.float32)
                x22 = tf.matmul(xi ** 2, tf.ones((1, tc.mod_N), dtype=tf.float32))
                y = tf.constant((np.arange(tc.mod_N) - tc.mod_N / 2), shape=[1, tc.mod_N], dtype=tf.float32)
                y22 = tf.matmul(tf.ones((tc.mod_N, 1), dtype=tf.float32), y ** 2)
                img_amp = tf.ones((tc.mod_M, tc.mod_N), dtype=tf.float32)
                xishu = tf.ones((tc.mod_M, tc.mod_N), dtype=tf.float32) * 2
                input_amp = tc.input_amp * tf.cast(img_amp * tf.exp(-tf.sqrt(x22 + y22) * tf.sqrt(xishu) / 100), dtype=tf.float32)
                input_phase = 0.999 * np.pi * tf.ones((tc.mod_M, tc.mod_N), dtype=tf.float32)
                input_field = tf.complex(input_amp * tf.cos(input_phase), input_amp * tf.sin(input_phase))
                "create a group of gaussian beam"
                input_field_batch = tf.expand_dims(input_field, 0)
                for number in range(1, tc.BATCH_SIZE):
                    input_field_batch = tf.concat([input_field_batch, tf.expand_dims(input_field, 0)], 0)
                "propagate to input layer and carry input information"
                img_p = tom.tf_FSPAS_FFT(input_field_batch, tc.WLENGTH, tc.MASK_Si_DISTANCE, tc.DX, tc.DY, 1.0)
                paddings = int((512-256)/2)
                fields = tf.pad(field, [(0,0), (paddings, paddings), (paddings, paddings)], mode='CONSTANT',constant_values=0, name=None)
                hidden = tf.multiply(img_p, fields)

        with tf.name_scope('mask'):
                "initialize masks"
                mask_phase, mask_amp = mask_init(0)
                mask = tf.complex(mask_amp * tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
                save_mask_phase = tf.expand_dims(mask_phase, 0)
                save_mask_amp = tf.expand_dims(mask_amp, 0)

        with tf.name_scope('propagation'):
                "first propagation, nonlinear activation, and modulation"
                prop = tom.tf_FSPAS_FFT(hidden, tc.WLENGTH, tc.MASK_Si_DISTANCE, tc.DX, tc.DY, tc.n_air)
                hidden = phase_NL(prop)
                prop = tom.tf_FSPAS_FFT(hidden, tc.WLENGTH, tc.Si_thick, tc.DX, tc.DY, tc.n_si)
                prop = tom.tf_FSPAS_FFT(prop, tc.WLENGTH, tc.MASK_Si_DISTANCE, tc.DX, tc.DY, tc.n_air)
                hidden = tf.multiply(prop, mask)

    if layer > 1:
        for layer_number in range(1,layer):
            with tf.name_scope('middle-layer-'+str(layer_number)):
                with tf.name_scope('mask'):
                    mask_phase, mask_amp = mask_init(layer_number)
                    mask = tf.complex(mask_amp * tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
                    save_mask_phase = tf.concat([save_mask_phase, tf.expand_dims(mask_phase, 0)], 0)
                    save_mask_amp = tf.concat([save_mask_amp, tf.expand_dims(mask_amp, 0)], 0)
            with tf.name_scope('propagation'):
                prop = tom.tf_FSPAS_FFT(hidden, tc.WLENGTH, tc.MASK_Si_DISTANCE, tc.DX, tc.DY, tc.n_air)
                hidden = phase_NL(prop)
                prop = tom.tf_FSPAS_FFT(hidden, tc.WLENGTH, tc.Si_thick, tc.DX, tc.DY, tc.n_si)
                prop = tom.tf_FSPAS_FFT(prop, tc.WLENGTH, tc.MASK_Si_DISTANCE, tc.DX, tc.DY, tc.n_air)
                hidden = tf.multiply(prop, mask)

    with tf.name_scope('last-layer'):
        with tf.name_scope('propagation'):
                "propagate to sensor"
                img_p = tom.tf_FSPAS_FFT_last(hidden, tc.WLENGTH, tc.MASK_SENSOR_DISTANCE, tc.DX, tc.DY, tc.n_air)

        with tf.name_scope('sensor'):
                "calcuate the intensity distribution on the sensor plane and classification probability"
                measurement = tf.square(tf.abs(img_p))
                cell_locs, thecell, label_fields = dto.sensorplane_geometry()
                logits = detector_plane(measurement, cell_locs, thecell)
    return measurement, save_mask_phase, save_mask_amp, logits


def loss_function(self, ground_truth):
    loss_ = tf.losses.mean_squared_error(labels=self, predictions=ground_truth)
    return loss_


def training(loss):
    "Define the training operation for the model"
    ONN_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mask')
    DNN_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense')
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.GradientTape() as tape:
     gradients = tape.gradient(loss, ONN_variable)
    if tc.OPTIMIZER == 'gradient':
        optimizer = tf.train.GradientDescentOptimizer(tc.LEARNING_RATE)
    elif tc.OPTIMIZER == 'adam':
        optimizer_ONN = tf.train.AdamOptimizer(tc.LEARNING_RATE)
        optimizer_DNN = tf.train.AdamOptimizer(tc.LEARNING_RATE)
    else:
        pass
    train_op = tf.group([optimizer_ONN.minimize(loss, var_list=ONN_variable), optimizer_DNN.minimize(loss, var_list=DNN_variable), extra_update_ops])
    return train_op



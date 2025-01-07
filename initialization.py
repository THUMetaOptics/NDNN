from configobj import ConfigObj
import tensorflow as tf
import numpy as np
def init_params():
    
    '''
    tc : training_configuration
    '''
    # USER-DEFINED GLOBAL PARAMETERS
    tc = ConfigObj()
    tc.APPLICATION = 'classification'
    tc.M, tc.N =  256,256       
    tc.FM, tc.FN = 4*tc.M, 4*tc.N
    tc.WLENGTH = 1.55e-6
    tc.DX, tc.DY = 12.5e-6,12.5e-6                          #pixel size of SLM
    tc.DATA_ROW, tc.DATA_COL = 28, 28                       #original data size
    tc.RM, tc.RN = tc.DATA_ROW*6, tc.DATA_COL*6             #size after padding                     
    tc.mod_M,tc.mod_N = 286,286                             #modulation parameters 286*286
    tc.out_M,tc.out_N = 4*tc.M, 4*tc.N                      #pixels on output plane
    tc.OBJECT_PHASE_INPUT, tc.OBJECT_AMPLITUDE_INPUT = True, False          
    tc.MASK_PHASE_MODULATION, tc.MASK_AMPLITUDE_MODULATION = True, False         
    tc.detect = 20                                          #width/length of detect region
    tc.MASK_INIT_TYPE = 'const'                             #initialization type
    tc.layer = 7                                            #number of sweep parameters
    tc.MASK_NUMBER = [3,4,5,6,7,8,9]                        #layer number
    tc.BATCH_SIZE, tc.TEST_BATCH_SIZE = 16,16                                      
    tc.n_air = 1
    tc.n_si = 3.4757
    tc.MASK_Si_DISTANCE, tc.MASK_SENSOR_DISTANCE = 2.85856e-2, 15e-2                
    tc.Si_thick = 1e-2
    tc.NUM_CLASS = 25                                       
    tc.NUMBER_TRAINING_ELEMENTS, tc.NUMBER_TEST_ELEMENTS = 100000, 25000               
    tc.input_amp=2
    tc.LEARNING_RATE, tc.OPTIMIZER, tc.TV_LOSS_PARAM = 2e-3, 'adam', 0.0            #Learning rate
    tc.MAX_EPOCH = 100                                                              #Epoch
    tc.TFBOARD_PATH, tc.MASK_SAVING_PATH = '.\TFBOARD', '.\MODEL\MASKS'
    tc.validation_ratio = 0.5                               # Ratio of number of elements in validation set to test set
    tc.MASK_PATH = '.\MODEL\MASKS\mask'
    return tc

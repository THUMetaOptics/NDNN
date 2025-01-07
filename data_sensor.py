import numpy as np
import initialization as init

tc = init.init_params()
    
def find_nearest(self,a):
    
    A = np.asarray(self)
    idx = (np.abs(A-a)).argmin()
    return A[idx],idx 

#sensor plane design based on sensor width, number of detetors
def sensorplane_geometry():
    LABEL_ROW = tc.detect
    LABEL_COL = tc.detect
    thecell = np.ones((LABEL_ROW, LABEL_COL))
    cell_locs = np.zeros((tc.NUM_CLASS, 2))
    det_count = 0
    label_fields = np.zeros((tc.NUM_CLASS, tc.out_M, tc.out_N))
    for rr in range(tc.NUM_CLASS):
        qx = int(np.round(tc.out_M/2-120*np.cos(np.deg2rad(18+360/tc.NUM_CLASS*(rr-1)))-LABEL_ROW/2, 0))
        qy = int(np.round(tc.out_N/2+120*np.sin(np.deg2rad(18+360/tc.NUM_CLASS*(rr-1)))-LABEL_ROW/2, 0))
        cell_locs[det_count, :] = [qx, qy]
        label_fields[det_count, qx:qx + LABEL_ROW, qy:qy + LABEL_COL] = 20
        det_count = det_count+1
    cell_locs = cell_locs.astype('int')
    return cell_locs, thecell, label_fields


if tc.APPLICATION == 'classification':
        global cell_locs
        global thecell
        cell_locs, thecell, label_fields = sensorplane_geometry()

#generates a ground truth sensor array for a given class.
def gt_generator_classification(cls):
    gt_sensor = np.zeros((tc.out_M,tc.out_N), dtype=np.float32)
    cellindex = cls
    ulcorner = cell_locs[cellindex,0:2]
    gt_sensor[ulcorner[0]:ulcorner[0]+thecell.shape[0],ulcorner[1]:ulcorner[1]+thecell.shape[1]] = thecell
    return gt_sensor

#Creates a validation set from the training labels based on a given ratio.
def create_validation(label_train, label_test, ratio):
    N_TEST = np.amax(label_test.shape)
    N_TRAIN = np.amax(label_train.shape)
    label_train = np.squeeze(label_train)
    N_VAL = int(N_TEST*ratio)
    N_cls_val = int(np.round(N_VAL/tc.NUM_CLASS))
    ind_val = np.zeros((N_VAL))
    indexes = np.arange(N_TRAIN)

    for cls in range(tc.NUM_CLASS):
        ind_cls = indexes[label_train==cls]
        Ncls = int(np.sum([label_train==cls]))
        #print(Ncls)
        begin = cls*N_cls_val
        rand_ind = ind_cls[np.random.randint(0,Ncls,size=[2*N_cls_val,1])]
        R = np.random.permutation(np.unique(rand_ind))
        ind_val[begin:begin+N_cls_val] = R[0:N_cls_val]
    return ind_val.astype('int32')
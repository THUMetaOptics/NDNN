# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:30:30 2018

@author: deepLearning505
"""

import tensorflow.compat.v1 as tf
import numpy as np
import initialization as init
tc = init.init_params()


def tf_fft_shift_2d(self):
    " shift the zero-frequency component to the center of the spectrum for a 2D tensor."
    B, M, N = self.shape
    if np.mod(M.value, 2) == 0:
        M_half = M.value / 2.0
    else:
        M_half = np.floor(M.value / 2.0) + 1.0
    if np.mod(N.value, 2) == 0:
        N_half = N.value / 2.0
    else:
        N_half = np.floor(N.value / 2.0) + 1.0
    img_1 = tf.slice(self, np.int32([0, 0, 0]), np.int32([B, M_half, N_half]))
    img_2 = tf.slice(self, np.int32([0, 0, N_half]), np.int32([B, M_half, N.value - N_half]))
    img_3 = tf.slice(self, np.int32([0, M_half, 0]), np.int32([B, M.value - M_half, N_half]))
    img_4 = tf.slice(self, np.int32([0, M_half, N_half]), np.int32([B, M.value - M_half, N.value - N_half]))
    return tf.concat([tf.concat([img_4, img_3], 2), tf.concat([img_2, img_1], 2)], 1)


def tf_ifft_shift_2d(self):
    " shift the zero-frequency component to the center of the spectrum for a 2D tensor."
    B, M, N = self.shape
    if np.mod(M.value, 2) == 0:
        M_half = M.value / 2.0
    else:
        M_half = np.floor(M.value / 2.0)
    if np.mod(N.value, 2) == 0:
        N_half = N.value / 2.0
    else:
        N_half = np.floor(N.value / 2.0)
    img_1 = tf.slice(self, np.int32([0, 0, 0]), np.int32([B, M_half, N_half]))
    img_2 = tf.slice(self, np.int32([0, 0, N_half]), np.int32([B, M_half, N.value - N_half]))
    img_3 = tf.slice(self, np.int32([0, M_half, 0]), np.int32([B, M.value - M_half, N_half]))
    img_4 = tf.slice(self, np.int32([0, M_half, N_half]), np.int32([B, M.value - M_half, N.value - N_half]))
    return tf.concat([tf.concat([img_4, img_3], 2), tf.concat([img_2, img_1], 2)], 1)


def tf_shift(self):
    "extracts a specific region from the input tensor, effectively shifting the view"
    img_1 = tf.slice(self, np.array([0, tc.FM / 2- tc.M , tc.FM / 2 - tc.N], dtype=np.int32),np.array([-1, tc.M*2, tc.N*2], dtype=np.int32))
    return img_1

def tf_shift_last(self):
    "extracts a specific region from the input tensor for the last propagation"
    img_1 = tf.slice(self, np.array([0, tc.FM / 2- tc.out_M/2 , tc.FM / 2 - tc.out_N/2 ], dtype=np.int32),np.array([-1, tc.out_M, tc.out_N], dtype=np.int32))
    return img_1



def padding(self):
    "pad the image"
    B, M, N = self.shape
    padx = int((tc.FM - M.value) / 2)
    pady = int((tc.FN - N.value) / 2)
    img_pad = tf.pad(self, [(0,0),(padx, padx), (pady, pady)], 'constant',constant_values=0)
    return img_pad


def tf_FSPAS_FFT(self,wlength,z,dx,dy,ridx):
    "define propagation by angular spectrum theory"
    input = padding(self)
    wlengtheff = wlength/ridx
    B,M,N = input.shape
    dfx = 1/dx/M.value
    dfy = 1/dy/N.value
    fx = tf.constant((np.arange(M.value)-(M.value)/2)*dfx,shape=[M.value,1],dtype=tf.float32)
    fy = tf.constant((np.arange(N.value)-(N.value)/2)*dfy,shape=[1,N.value],dtype=tf.float32)
    fx2 = tf.matmul(fx**2,tf.ones((1,N.value),dtype=tf.float32))
    fy0 = 1 / dx / 2
    fy2 = tf.matmul(tf.ones((M.value,1),dtype=tf.float32),(fy+fy0)**2)
    W = (fx2+fy2)*(wlengtheff**2)
    Hphase = 2*np.pi/wlengtheff*(z*(tf.ones((M.value,N.value))-W)**(0.5))
    sintheta = wlengtheff / 2 / dy
    y0 = z * np.tan(np.arcsin(sintheta))
    inci_f = 2 * np.pi * fy * y0
    Hphase_f = Hphase + inci_f
    HFSP = tf.complex(tf.cos(Hphase_f),tf.sin(Hphase_f))
    ASpectrum = tf.signal.fft2d(input)
    ASpectrum = tf_fft_shift_2d(ASpectrum)
    ASpectrum_z = tf_ifft_shift_2d(tf.multiply(HFSP,ASpectrum))
    output = tf.signal.ifft2d(ASpectrum_z)
    output = tf_shift(output)
    return output

def tf_FSPAS_FFT_mid(self, wlength, z_air, z_si, dx, dy, ridx_air,ridx_si):
    "define once propagation between modulation layers in linear situation"
    y1 = tf_FSPAS_FFT(self,wlength,z_air ,dx,dy,ridx_air)
    y2 = tf_FSPAS_FFT(y1,wlength,z_si ,dx,dy,ridx_si)
    y3 = tf_FSPAS_FFT(y2,wlength,z_air ,dx,dy,ridx_air)
    return y3



def tf_FSPAS_FFT_last(self, wlength, z, dx, dy, ridx):
    "define propagation between last modulation layer and sensor plane"
    input = padding(self)
    wlengtheff = wlength / ridx
    B, M, N = input.shape
    dfx = 1 / dx / M.value
    dfy = 1 / dy / N.value
    fx = tf.constant((np.arange(M.value) - (M.value) / 2) * dfx, shape=[M.value, 1], dtype=tf.float32)
    fy = tf.constant((np.arange(N.value) - (N.value) / 2) * dfy, shape=[1, N.value], dtype=tf.float32)
    fx2 = tf.matmul(fx ** 2, tf.ones((1, N.value), dtype=tf.float32))
    fy0 = 1 / dx / 2
    fy2 = tf.matmul(tf.ones((M.value, 1), dtype=tf.float32), (fy+fy0) ** 2)
    W = (fx2 + fy2) * (wlengtheff ** 2)
    Hphase = 2 * np.pi / wlengtheff * (z * (tf.ones((M.value, N.value)) - W) ** (0.5))
    y0=z*np.tan(np.arcsin(wlengtheff/2/dy))
    inci_f=2*np.pi*fy*y0
    Hphase_f=Hphase+inci_f

    HFSP = tf.complex(tf.cos(Hphase_f), tf.sin(Hphase_f))
    ASpectrum = tf.signal.fft2d(input)
    ASpectrum = tf_fft_shift_2d(ASpectrum)
    ASpectrum_z = tf_ifft_shift_2d(tf.multiply(HFSP, ASpectrum))
    output = tf.signal.ifft2d(ASpectrum_z)
    output = tf_shift_last(output)
    return output


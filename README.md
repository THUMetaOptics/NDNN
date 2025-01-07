# NDNN
This repository contains the code used to build and train the model in the study "Scalable multilayer diffractive neural network with nonlinear activation". We present a scalable, reconfigurable folded multilayer diffractive neural network (DNN) system that utilizes a single spatial light modulator (SLM) activated by the nearly-instantaneous all-optical nonlinearity of a mirror-coated 5-mm-thick silicon (Si) wafer, as schematically illustrated in Fig.1.
![Schematic-05](https://github.com/user-attachments/assets/d30742d7-3938-4abc-aef3-df1da8f2e339)
Figure 1 | Schematic of the scalable, reconfigurable folded multilayer DNN system, consisting of a single SLM and a mirror-coated Si wafer.

We construct a digital DNN model, as shown in Fig. 2. The Gaussian beam is incident on the first block of SLM to carry the input information. After three nonlinear activations and phase modulations, the output is captured. Subsequently, the mean-squared error (MSE) loss is used to quantitative describe the differences between the designed distribution and the actual output. To ensure the result is not affected by the zero-order diffraction, the detect regions are arranged in a ring pattern. The phase profiles are iteratively updated using the backpropagation algorithm, ultimately achieving the desired phase masks.
![Training-05](https://github.com/user-attachments/assets/00ad6f64-4328-4daa-99d1-da6335d13a69)
Figure 2 | Training process of DNN. The black and orange arrows represent the diffractive propagation of the forward model and the backpropagation-based training process, respectively.

A quick guide on the contents of the repository:
+The 'dataset_part.zip' folder contains example images from 25-class quickdraw dataset, with 50 images taken from each class. In the actual simulation, 5000 images are used per class, which can be obtained by downloading the QuickDraw dataset (.bin) and converting the files using 'data.ipynb'
+'initialization.py' includes the intialization of model parameters.
+'data_generation.py' includes dataset loading and preprocessing functions.
+'data_sensor.py' defines the design of the sensor plane.
+'OpticsModule.py' includes the propagation function based on the angular spectrum theory.
+'mask_modulation_model.py' defines both the linear and nonlinear model.
+'ndnn_train.py' contains the training, testing and saving procedures for the model.


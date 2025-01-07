# NDNN
This repository contains the code used for building and training the model in the study "Scalable multilayer diffractive neural network with nonlinear activation". We present a scalable, reconfigurable folded multilayer diffractive neural network (DNN) system that utilizes a single spatial light modulator (SLM) activated by the nearly-instantaneous all-optical nonlinearity of a mirror-coated 5-mm-thick silicon (Si) wafer, as schematically illustrated in Fig.1.
![Schematic-05](https://github.com/user-attachments/assets/d30742d7-3938-4abc-aef3-df1da8f2e339)
Figure 1 | Schematic of the scalable, reconfigurable folded multilayer DNN system, which consists of a single SLM and a mirror-coated Si wafer.

We build a digital DNN model, as shown in Fig. 2. The Gaussian beam is incident on the first block of SLM to carry the input information. After three nonlinear activations and phase modulations, the output is captured. Subsequently, the mean-squared error (MSE) loss is used to quantitative describe the differences between the designed distribution and the actual output. To ensure the result is not affected by the zero-order diffraction, the detect regions are arranged in a ring pattern. 
![Training-05](https://github.com/user-attachments/assets/00ad6f64-4328-4daa-99d1-da6335d13a69)
Figure 2 | Training process of DNN. The black and orange arrows represent the diffractive propagation of the forward model and the backpropagation-based training process, respectively.

A quick guide on the contents of the repository:

Folder 'Data_Collection_Example' and 'Data_Extraction_Example' contain example scripts for instrument control and data collection through the multilayer optical neural network.
Other folders are organized according to the figures of the main text of the paper, each containing the data and the code required to reproduce the plots shown in the main text and associated supplementary figures. Note that some customized dataset used in this study can only be downloaded from Zenodo: https://doi.org/10.5281/zenodo.6888985, due to the size limit of files that are uploadable to GitHub.
Each folder contains a README.txt file that explains the role of each file in the folder in more detail.
For an exmaple of training an optical-neural-network enocder with a digital backend for classification, one can refer to 'Figure_2bcd/QuickDraw_train_onn_encoder.ipynb' as an introduction and 'Figure_4/Full_EBI_Cell_PLOTS.ipynb' for examples with more complex neural-network models.

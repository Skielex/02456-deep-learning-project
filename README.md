# Un- and Semi-supervised Learning for Ultrasonic Images with Convolutional Variational Autoencoders
Repository for project in course 02456 Deep Learning at DTU, fall 2017, by Niels Jeppesen (niejep).

## Prerequisites
- python 3.5+
- keras 2.0+
- tensorflow 1.3+
- scikit-learn 0.19+

## Files
- `autoencoder.py` implements a Variational Autoencoder and a Semi-superviser Variational Autoencoder/Classifier in Keras/Tensorflow.
- `Unsupervivised VAE.ipynb` is a Jupyter Notebook for recreating unsupervised MNIST result from the paper.
- `Semi-supervised VAE.ipynb` is a Jupyter Notebook for recreating semi-supervised MNIST result from the paper.

## Training
It is possible to train the implementations in the notebooks by setting `TRAIN = True` in the beginning of the notebook. By default, training plot are saved to the `output` folder using Tensorboard summaries, and models are saved to the same folder using Keras.

## Loading models
Existing models can be loaded by setting `LOAD_LAST_WEIGHTS = True`. If you don't want to train remember to set `TRAIN = False`. Models are loaded from the folder specified by the `input_dir` variable, which by default points to the `output` folder. You can either load models you've trained yourself, or download pre-trained model listed below.

## Pre-trained models
To avoid having to train all the different models in the notebook the following is provided. Simply extract the `output` folder in the repository root and load the models as described above.
- [Un- and semi-supservised models and training data - 20 epochs](https://dtudk-my.sharepoint.com/personal/niejep_win_dtu_dk/_layouts/15/guestaccess.aspx?docid=08ad1ee819e3147cc82a75538d0a2b814&authkey=Aaq-NtLBwlea5Hcr31_PP74&e=0a57997c44c348b496b538deff85fc7c)
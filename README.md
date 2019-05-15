# Exoplanet Habitability Classification using PHL-EC Data
The notebooks perform all the analysis using both classical ML models and deep learning models. The notebooks are described below.

* `All_features_model.ipynb`: The neural network using a naive approach, using `fastai` and the 1cycle learning policy. This uses three subsets of the data--one with all the features, one with only six features, and another with surface temperature (a key feature for habitability) removed.
* `All_features_model_oversampled.ipynb`: The same as above; however, this time, the classes are balanced by oversampling the mesoplanets and psychroplanets.
* `All_features_model_oversampled_classic.ipynb`: The same as above, but this time, classical ML models are also explored. It turns out they perform very well!
* `All_features_model_v4.ipynb`: The same as above, but this time, the way oversampling is done is fixed. First, the data is split into train and validation sets, and then the training set is oversampled. This prevents the same example appearing in both sets.
* `TabularNN.ipynb`: An attempt to build an Embedding-based neural network using PyTorch, based on fast.ai's architecture for tabular datasets (does not work).
* `Tabular Keras.ipynb`: The above, but in Keras. Works, but LipschitzLR performs poorly. At the end of this notebook is an alternative architecture that's simpler and also works well, but this is continued (and is cleaner) in the next notebook.
* `Alternative network.ipynb`: The alternative network architecture. LipschitzLR outperforms a standard LR with ReLU, LeakyReLU, and PReLU activations.
* `Automated - alternate and original arch.ipynb`: Contains the original architecture and hyper-parameters as well as the alternate architecture above, with automated preprocessing and running.
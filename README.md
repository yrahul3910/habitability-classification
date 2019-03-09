# Exoplanet Habitability Classification using PHL-EC Data
The notebooks perform all the analysis using both classical ML models and deep learning models. The notebooks are described below.

* `All_features_model.ipynb`: The neural network using a naive approach, using `fastai` and the 1cycle learning policy. This uses three subsets of the data--one with all the features, one with only six features, and another with surface temperature (a key feature for habitability) removed.
* `All_features_model_oversampled.ipynb`: The same as above; however, this time, the classes are balanced by oversampling the mesoplanets and psychroplanets.
* `All_features_model_oversampled_classic.ipynb`: The same as above, but this time, classical ML models are also explored. It turns out they perform very well!
* `All_features_model_v4.ipynb`: The same as above, but this time, the way oversampling is done is fixed. First, the data is split into train and validation sets, and then the training set is oversampled. This prevents the same example appearing in both sets.

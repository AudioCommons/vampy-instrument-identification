from __future__ import division

import numpy as np
from keras.backend import image_dim_ordering
from keras.models import model_from_json



class DeepDeploy(object):
    def __init__(self, class_names, model_path, weights_path, feature_extractor):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model_path = model_path
        self.weights_path = weights_path
        if image_dim_ordering() == 'tf':
            self.channel_axis = -1
        else:
            self.channel_axis = -3
        self.feature_extractor = feature_extractor

    def load(self):
        with open(self.model_path) as model_file:
            self.model = model_from_json(model_file.read())
        self.model.load_weights(self.weights_path)

    def predict_from_audio_file(self, audio_path, means=0, stddeviations=1):
        features, start_time, end_time = self.feature_extractor.calculate_from_audio_file(audio_path)
        return self.predict(features, means, stddeviations), start_time, end_time

    # called by Vamp plugins
    def predict_from_spectrogram(self, spectrogram, means=0, stddeviations=1):
        features, start_time, end_time = self.feature_extractor.calculate_from_spectrogram(spectrogram)
        return self.predict(features, means, stddeviations), start_time, end_time

    def predict(self, features, means=0, stddeviations=1):
        scaled_features = np.expand_dims((features - means) / stddeviations, self.channel_axis)
        batch_of_one = np.expand_dims(scaled_features, axis=0)
        class_probabilities = self.model.predict(batch_of_one)
        max_class_idx = np.argmax(class_probabilities)
        return self.class_names[max_class_idx], class_probabilities[0, max_class_idx], class_probabilities

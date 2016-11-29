'''VampyInstrumentIdentification.py.

Created by Johan Pauwels, Centre for Digital Music, Queen Mary University of London.
'''
from vampy import *
import os
os.environ['KERAS_BACKEND'] = 'theano'
from os.path import dirname, join, abspath
import copy
import numpy as np
from deepdeploy import DeepDeploy
from deepdeploy.feature_extraction import LogMelSpectrogramVamp
from keras.backend import backend, image_dim_ordering, set_image_dim_ordering
# try:
# 	import pydevd
# 	pydevd.settrace(suspend=False)
# except ImportError:
# 	pass

class VampyInstrumentIdentification: 
	def __init__(self,inputSampleRate):
		self.vampy_flags =  vf_DEFAULT_V2 #| vf_DEBUG
		self.m_inputSampleRate = inputSampleRate
		self.frames = []
		self.instrument_names = ['Shaker', 'Electronic Beats', 'Drum Kit', 'Synthesizer', 'Female', 'Male', 'Violin', 'Flute', 'Harpsichord', 'Electric Guitar', 'Clarinet', 'Choir', 'Organ', 'Acoustic Guitar', 'Viola', 'French Horn', 'Piano', 'Cello', 'Harp', 'Conga', 'Synthetic Bass', 'Electric Piano', 'Acoustic Bass', 'Electric Bass']
		set_image_dim_ordering('th')
		return None
		
	def initialise(self,channels,stepSize,blockSize):
		if blockSize != 2048:
			# TODO: Warn user of mismatch
			return False
		if stepSize != 512:
			# TODO: Warn user of mismatch
			return False
		if self.m_inputSampleRate != 22050:
			# TODO: Warn user of mismatch
			return False
		model_dir = dirname(abspath(__file__))
		model_path = join(model_dir, 'model_{}.json'.format(image_dim_ordering()))
		weights_path = join(model_dir, 'weights_{}_{}.hdf5'.format(backend(), image_dim_ordering()))
		self.deploy = DeepDeploy(self.instrument_names, model_path, weights_path, LogMelSpectrogramVamp(2048, 512, 22050, 5, 128, 'max-energy'))
		self.deploy.load()
		return True

	def reset(self):
		return None
	
	def getMaker(self):
		return 'Johan Pauwels'
	
	def getName(self):
		return 'Instrument identification'
		
	def getIdentifier(self):
		return 'instrument-identification'

	def getDescription(self):
		return 'Enter a short description of the plugin'

	def getMaxChannelCount(self):
		return 1

	def getInputDomain(self):
		# use TimeDomain or FrequencyDomain
		return FrequencyDomain 

	def getPreferredBlockSize(self):
		return 2048

	def getPreferredStepSize(self):
		return 512

	def getOutputDescriptors(self):
		common = OutputDescriptor()
		common.unit = None
		common.hasFixedBinCount = True
		common.hasKnownExtents = True
		common.minValue = 0
		common.maxValue = 1
		common.isQuantized = False
		common.hasDuration = True
		common.sampleType = VariableSampleRate
		common.sampleRate = 0
		
		predominant_instrument = OutputDescriptor(common)
		predominant_instrument.identifier = 'predominant-instrument'
		predominant_instrument.name = 'Predominant instrument'
		predominant_instrument.description = 'The instrument that is dominant in the audio'
		predominant_instrument.binCount = 1
		
		instrument_probabilities = OutputDescriptor(common)
		instrument_probabilities.identifier = 'instrument-probabilities'
		instrument_probabilities.name = 'Instrument probabilities'
		instrument_probabilities.description = 'Probabilities of all possible instruments'
		instrument_probabilities.binCount = len(self.instrument_names)
		instrument_probabilities.binNames = self.instrument_names
		
		return OutputList(predominant_instrument, instrument_probabilities)

	def getParameterDescriptors(self):
		return ParameterList()

	def setParameter(self,paramid,newval):
		return

	def getParameter(self,paramid):
			return 0.0

	def process(self, inputbuffers, timestamp):
		# accumulate frames
		self.frames.append(copy.deepcopy(inputbuffers))
		return None

	def getRemainingFeatures(self):
		# timesignal = np.array(self.frames).reshape(-1)
		spectrogram = np.squeeze(np.array(self.frames)).T
		(max_instrument_name, max_instrument_prob, instrument_probabilities), start_time, end_time = self.deploy.predict_from_spectrogram(spectrogram)
		
		output = FeatureSet()
		timestamp = RealTime('seconds', start_time)
		duration = RealTime('seconds', end_time-start_time)
		output[0] = Feature(values=max_instrument_prob, label=max_instrument_name, timestamp=timestamp, duration=duration)
		output[1] = Feature(values=instrument_probabilities[0], timestamp=timestamp, duration=duration)
		top_instruments = sorted(zip(self.deploy.class_names, instrument_probabilities[0]), key=lambda l:l[1], reverse=True)[0:5]
		return output

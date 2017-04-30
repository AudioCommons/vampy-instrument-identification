from __future__ import division
import tempfile
import shutil
import subprocess
from os.path import join, basename, splitext
import csv
import numpy as np
from core import FeatureExtractor
from filter_banks import MelFilterBank
from math import ceil

class LogMelSpectrogramVamp(FeatureExtractor):
    def __init__(self, frame_size, step_size, samplerate, audio_duration, num_mel_filters, position='middle'):
        super(LogMelSpectrogramVamp, self).__init__(samplerate, frame_size, step_size)
        self.parameter_string = '{:d}kHz-{:d}ms-{:d}ms-{}s-{}f-{}'.format(int(samplerate/1000), int(round(frame_size/samplerate*1000)), int(round(step_size/samplerate*1000)), audio_duration, num_mel_filters, position)
        num_samples = int(audio_duration * samplerate)
        self.num_rows = num_mel_filters
        self.num_columns = int(ceil(num_samples / step_size))
        self.filterbank = MelFilterBank(samplerate, frame_size//2+1, num_mel_filters)
        self.position = position
        self.dir_path = tempfile.mkdtemp()
        with tempfile.NamedTemporaryFile(dir=self.dir_path, delete=False) as config_file:
            self.config_path = config_file.name
            config_file.write('''\
            @prefix xsd:      <http://www.w3.org/2001/XMLSchema#> .
            @prefix vamp:     <http://purl.org/ontology/vamp/> .
            @prefix :         <#> .
            :transform a vamp:Transform ;
            vamp:plugin <http://vamp-plugins.org/rdf/plugins/vamp-features#frequency> ;
            vamp:sample_rate "{}"^^xsd:int ;
            vamp:step_size "{}"^^xsd:int ;
            vamp:block_size "{}"^^xsd:int ;
            vamp:plugin_version """1""" ;
            vamp:output <http://vamp-plugins.org/rdf/plugins/vamp-features#frequency_output_frames> .
            '''.format(samplerate, step_size, frame_size))
        
    # def __del__(self):
    #     shutil.rmtree(self.dir_path)

    # called by DeepTrain through DeepDeploy
    def calculate_from_audio_file(self, filepath):
        subprocess.check_output(['sonic-annotator', '-t', self.config_path, '-w', 'csv', '--csv-force', '--csv-basedir', self.dir_path, filepath])
        result_path = join(self.dir_path, splitext(basename(filepath))[0]+'_vamp_vamp-features_frequency_frames.csv')
        with open(result_path, 'rb') as result_file:
            values = np.array(map(lambda x: map(float, x), csv.reader(result_file)))
            times = values[:,0]
            # spectrogram = np.vectorize(complex)(values[:,1::2], values[:,2::2])
            spectrogram = (values[:,1::2] + values[:,2::2] * 1j).T
            return self.calculate_from_spectrogram(spectrogram)

    # called by Vamp plugins through DeepDeploy
    def calculate_from_spectrogram(self, spectrogram):
        middle_spectrogram, start_time, end_time = self.select_frames(spectrogram, self.num_columns, self.position)
        middle_melspectrogram = self.filterbank.apply(np.absolute(middle_spectrogram))
        return 20 * np.log10(np.maximum(middle_melspectrogram, 1e-20)), start_time, end_time

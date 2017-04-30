from __future__ import division
import numpy as np

class FeatureExtractor(object):
    def __init__(self, samplerate, frame_size, step_size):
        self.samplerate = samplerate
        self.frame_size = frame_size
        self.step_size = step_size
        self.parameter_string = ''
        self.num_rows = None
        self.num_columns = None

    def select_samples(self, samples, num_samples, position='middle'):
        num_original = len(samples)
        if num_original > num_samples:
            if position == 'start':
                start_sample = 0
                end_sample = num_samples
            elif position == 'middle':
                start_sample = (num_original-num_samples)//2
                end_sample = (num_original+num_samples)//2
            elif position == 'end':
                start_sample = -num_samples
                end_sample = None
            elif position == 'max-energy':
                sliding_windows = np.lib.stride_tricks.as_strided(samples, shape=[len(samples)-num_samples+1, num_samples], strides=[samples.strides[0], samples.strides[0]])
                start_sample = np.argmax(np.sum(sliding_windows**2, axis=1))
                end_sample = start_sample + num_samples
            else:
                raise ValueError('Unknown position selector "{}"'.format(position))
            selected_samples = samples[start_sample:end_sample]
            start_time = start_sample / self.samplerate
            end_time = end_sample / self.samplerate
        else:
            selected_samples = np.hstack((samples, np.zeros((num_samples - num_original,))))
            start_time = 0
            end_time = num_original / self.samplerate
        return selected_samples, start_time, end_time

    def select_frames(self, frames, num_frames, position='middle'):
        num_original = frames.shape[1]
        if num_original > num_frames:
            if position == 'start':
                start_frame = 0
                end_frame = num_frames
            elif position == 'middle':
                start_frame = (num_original-num_frames)//2
                end_frame = (num_original+num_frames)//2
            elif position == 'end':
                start_frame = -num_frames
                end_frame = None
            elif position == 'max-energy':
                sliding_windows = np.lib.stride_tricks.as_strided(frames, shape=[frames.shape[1]-num_frames+1, frames.shape[0], num_frames], strides=(frames.strides[1],)+frames.strides)
                energies = np.sum(sliding_windows**2,axis=(1,2))
                start_frame = np.argmax(energies)
                end_frame = start_frame + num_frames
            else:
                raise ValueError('Unknown position selector "{}"'.format(position))
            selected_frames = frames[:, start_frame:end_frame]
            start_time = start_frame * self.step_size / self.samplerate
            end_time = end_frame * self.step_size / self.samplerate
        else:
            selected_frames = np.hstack((frames, np.zeros((self.frame_size//2+1, num_frames - num_original))))
            start_time = 0
            end_time = num_original * self.step_size / self.samplerate
        return selected_frames, start_time, end_time

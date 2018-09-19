"""Utilities for writing piano-rolls to MIDI files.
"""
import numpy as np
from pypianoroll import Multitrack, Track


def write_midi(filepath, pianorolls, config):
    is_drums = config['is_drums']
    track_names = config['track_names']
    tempo = config['tempo']
    beat_resolution = config['beat_resolution']
    program_nums = config['program_nums']

    if not np.issubdtype(pianorolls.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")
    if isinstance(program_nums, int):
        program_nums = [program_nums]
    if isinstance(is_drums, int):
        is_drums = [is_drums]

    if program_nums is None:
        program_nums = [0] * len(pianorolls)
    if is_drums is None:
        is_drums = [False] * len(pianorolls)

    multitrack = Multitrack(beat_resolution=beat_resolution, tempo=tempo)
    for idx in range(pianorolls.shape[2]):
        if track_names is None:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx])
        else:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx], track_names[idx])
        multitrack.append_track(track)
    multitrack.write(filepath)


# In[19]:


def save_midi(filepath, phrases, config):
    if not np.issubdtype(phrases.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")

    reshaped = phrases.reshape(-1, phrases.shape[1] * phrases.shape[2],
                               phrases.shape[3], phrases.shape[4])

    # print("reshaped shape:", reshaped.shape)
    # result final shape: (5, 1, 96, 84, 5)

    pad_width = ((0, 0), (0, config['pause_between_samples']),
                 (config['lowest_pitch'],
                  128 - config['lowest_pitch'] - config['num_pitch']),
                 (0, 0))

    # pad width 表示前补和后补的长度
    # print('pad_width:',pad_width)
    padded = np.pad(reshaped, pad_width, 'constant')

   #  print("padded shape:", padded.shape)
    pianorolls = padded.reshape(-1, padded.shape[2], padded.shape[3])
    # print("pianorolls shape:", pianorolls.shape)
    write_midi(filepath, pianorolls, config)

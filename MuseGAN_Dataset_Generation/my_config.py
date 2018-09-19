config_8b = {
    # Music
    'track_num': 8,
    'program_nums': [0, 0, 24, 32, 48, 64, 80, 88],
    'is_drums': [True, False, False, False, False, False, False, False],
    'track_names': ['Drums', 'Piano', 'Guitar', 'Bass', 'Ensemble',
                    'Reed', 'Synth Lead','Synth Pad'],
    'tempo': 120,
    'velocity': 100,
    # Dataset
    'dataset_path': '/Users/mac/Desktop/Brain/MuseGAN/training_data/lastfm_alternative_8b_phrase.npy',
    'dataset_name': 'lastfm_alternative_8b_phrase',
    # Playback
    'pause_between_samples': 96,
    # Data
    'num_bar': 4,
    'num_beat': 4,
    'num_pitch': 84,
    'num_track': 8,
    'num_timestep': 96,
    'beat_resolution': 24,
    'lowest_pitch': 24
    # MIDI note number of the lowest pitch in data tensors
}


config_5b = {
    # Music
    'track_num': 5,
    'program_nums': [0, 0, 24, 32, 48],
    'is_drums': [True, False, False, False, False],
    'track_names': ['Drums', 'Piano', 'Guitar', 'Bass', 'Ensemble'],
    'tempo': 120,
    'velocity': 100,
    # Dataset
    'dataset_path': '/Users/mac/Desktop/Brain/MuseGAN/training_data/lastfm_alternative_5b_phrase.npy',
    'dataset_name': 'lastfm_alternative_5b_phrase',
    # Playback
    'pause_between_samples': 96,
    # Data
    'num_bar': 4,
    'num_beat': 4,
    'num_pitch': 84,
    'num_track': 8,
    'num_timestep': 96,
    'beat_resolution': 24,
    'lowest_pitch': 24
    # MIDI note number of the lowest pitch in data tensors
}
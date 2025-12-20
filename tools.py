import numpy as np
from mne import concatenate_raws
from sklearn.model_selection import StratifiedShuffleSplit
import mne

def split_epochs_into_segments(epochs, seg_length_s, step_s=None):
    sfreq = int(epochs.info['sfreq'])
    seg_samples = int(round(seg_length_s * sfreq))
    step_samples = seg_samples if step_s is None else int(round(step_s * sfreq))

    segments = []
    segment_labels = []
    classes = list(epochs.event_id.keys())

    for cls in classes:
        sub = epochs[cls]  # Epochs for this class
        data = sub.get_data()  # shape (n_epochs, n_ch, n_times)
        for ep_idx in range(data.shape[0]):
            n_times = data.shape[2]
            for start in range(0, n_times - seg_samples + 1, step_samples):
                seg = data[ep_idx, :, start:start + seg_samples]
                segments.append(seg)
                segment_labels.append(cls)

    if len(segments) == 0:
        raise ValueError("No segments produced: check seg_length_s and epoch length.")

    data_new = np.stack(segments)  # (n_new, n_ch, seg_samples)
    info = epochs.info.copy()

    # https://mne.tools/stable/documentation/glossary.html#term-events
    event_id_map = epochs.event_id
    print(event_id_map)
    events = np.c_[np.arange(len(data_new)), np.zeros(len(data_new), int),
                   np.array([event_id_map[l] for l in segment_labels], int)]
    new_epochs = mne.EpochsArray(data_new, info, events=events, event_id=event_id_map, tmin=0.0)
    
    # preserve montage if present
    montage = epochs.get_montage()
    if montage is not None:
        new_epochs.set_montage(montage)

    return new_epochs

def merge_epochs(*epochs):
    if not epochs:
        raise ValueError("no epochs provided")
    parts = []
    for e in epochs:
        if isinstance(e, (list, tuple)):
            parts.extend(e)
        else:
            parts.append(e)
    if len(parts) == 1:
        return parts[0].copy()
    return mne.concatenate_epochs(parts)


def get_data(validation = True, resample = False, segment_length = 2.0, step = 0.5):

    raw_fnames = [r"mati/mati_imagery2_run1_20251211_211514_raw.fif",
                 r"mati/mati_imagery2_run2_20251211_205847_raw.fif",
                  r"mati/mati_imagery_1_run1_20251207_183304_raw.fif",
                  r"mati/mati_imagery_2_run1_20251207_190808_raw.fif",
                  #r"mati/mati_imagery3_run1_20251217_204245_raw.fif",
                  #r"mati/mati_imagery3_run2_20251217_212625_raw.fif"
                  ]

    konrad_mapping = {
        'A1': 'Cz',
        'A2': 'FCz',
        'A3': 'CP1',
        'A4': 'FC1',
        'A5': 'C1',
        'A6': 'CP3',
        'A7': 'C3',
        'A8': 'FC3',
        'A9': 'C4',
        'A10': 'FC4',
        'A11': 'Pz',
        'A12': 'CP2',
        'A13': 'CP4',
        'A14': 'C2',
        'A15': 'CPz',
        'A16': 'FC2'
    }

    eeg_channels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15",
                    "A16"]
    raws = [mne.io.read_raw_fif(f, preload=True) for f in raw_fnames]
    raws = [r.pick(eeg_channels) for r in raws]
    for i, r in enumerate(raws):
        print(i, r.info["sfreq"], len(r.ch_names), r.ch_names)
        r.filter(l_freq=8.0, h_freq=38.0, fir_design='firwin')
        r.rename_channels(konrad_mapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        r.set_montage(montage)
        if resample:
            r.resample(256)

    raw = mne.concatenate_raws(raws)

    events_real = {"relax": 1, "left_hand": 2, "right_hand": 3, "both_hands": 4, "both_feets": 5}
    events_predicted = {"relax_predicted": 11, "left_hand_predicted": 12, "right_hand_predicted": 13,
                        "both_hands_predicted": 14, "both_feets_predicted": 15}
    classification_result = {"correct": 20, "incorrect": 21}
    all_possible_events_id = {**events_real, **events_predicted, **classification_result}
    description_code_to_consistent_id = {str(v): v for v in all_possible_events_id.values()}
    events, _ = mne.events_from_annotations(raw, event_id=description_code_to_consistent_id)

    all_events_id = {1: 'Relax', 2: 'Left', 3: 'Right', 4: 'Both', 5: 'Feet'}
    mapping = {v: k for k, v in all_events_id.items()}

    task_margin = 0.5
    task_end = 4.5
    reject_criteria = dict(
        eeg=80e-6,  # 80 µV
    )

    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=mapping,
        baseline=None,
        tmin=task_margin,
        tmax=task_end,
        preload=True,
        reject=reject_criteria
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, rest_idx = next(sss.split(np.zeros(len(epochs.events[:, -1])), epochs.events[:, -1]))

    train_epochs = epochs[train_idx]
    rest_epochs = epochs[rest_idx]

    print(f"Train epochs: {len(train_epochs)}")
    print(f"Test+Valid epochs: {len(rest_idx)}")
    if validation:
        sss_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        test_idx, valid_idx = next(sss_valid.split(np.zeros(len(rest_epochs.events[:, -1])), rest_epochs.events[:, -1]))

        test_epochs = rest_epochs[test_idx]
        valid_epochs = rest_epochs[valid_idx]

        print(f"Test epochs: {len(test_epochs)}")
        print(f"Valid epochs: {len(valid_epochs)}")


        splitted_test_epochs = split_epochs_into_segments(test_epochs, segment_length, step)
        splitted_train_epochs = split_epochs_into_segments(train_epochs, segment_length, step)
        splitted_valid_epochs = split_epochs_into_segments(valid_epochs, segment_length, step)

        y_train = splitted_train_epochs.events[:, -1]
        y_test = splitted_test_epochs.events[:, -1]
        y_valid = splitted_valid_epochs.events[:, -1]

        print("X_train")
        print(len(splitted_train_epochs))
        print("y_train")
        print(len(y_train))
        print("X_test")
        print(len(splitted_test_epochs))
        print("y_test")
        print(len(y_test))
        print("X_valid")
        print(len(splitted_valid_epochs))
        print("y_valid")
        print(len(y_valid))
        return splitted_train_epochs, y_train, splitted_test_epochs, y_test, splitted_valid_epochs, y_valid

    else:
        splitted_test_epochs = split_epochs_into_segments(rest_epochs, segment_length, step)
        splitted_train_epochs = split_epochs_into_segments(train_epochs, segment_length, step)

        y_train = splitted_train_epochs.events[:, -1]
        y_test = splitted_test_epochs.events[:, -1]

        print("X_train")
        print(len(splitted_train_epochs))
        print("y_train")
        print(len(y_train))
        print("X_test")
        print(len(splitted_test_epochs))
        print("y_test")
        print(len(y_test))
        return splitted_train_epochs, y_train, splitted_test_epochs, y_test

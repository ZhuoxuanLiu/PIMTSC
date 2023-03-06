import os
import tempfile
from urllib.request import urlretrieve
import shutil
import sys
import warnings
import zipfile
from pathlib import Path
from sktime.datasets import load_from_tsfile
import numpy as np
import itertools


def pv(s, verbose):
    if verbose: print(s)

def get_dataset_desc(dsid):
    x_tr, y_tr, _, _ = get_UCR_data(dsid, return_split=True, path='.', parent_dir='./datasets/UCR', verbose=True)
    c_in = x_tr.shape[1]
    seq_len = x_tr.shape[2]
    num_classes = len(np.unique(y_tr))
    avg_class_examples = x_tr.shape[0] // num_classes
    return c_in, seq_len, num_classes, avg_class_examples

def delete_all_in_dir(tgt_dir, exception=None):
    if exception is not None: exception = tuple(exception)
    for file in os.listdir(tgt_dir):
        if exception is not None and file.endswith(exception): continue
        file_path = os.path.join(tgt_dir, file)
        if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
        elif os.path.isdir(file_path): shutil.rmtree(file_path)
        
        
def coll_repr(c, max_n=10):
    "String repr of up to `max_n` items of (possibly lazy) collection `c`"
    return f'(#{len(c)}) [' + ','.join(itertools.islice(map(repr,c), max_n)) + (
        '...' if len(c)>max_n else '') + ']'


def download(url, savepath):
    """
    download file from internet
    :param url: path to download from
    :param savepath: path to save files
    :return: None
    """
    def reporthook(a, b, c):
        """
        显示下载进度
        :param a: 已经下载的数据块
        :param b: 数据块的大小
        :param c: 远程文件大小
        :return: None
        """
        print("\rdownloading: %5.1f%%" % (a * b * 100.0 / c), end="")
    filename = os.path.basename(url)
    # 判断文件是否存在，如果不存在则下载
    if not os.path.isfile(os.path.join(savepath, filename)):
        print('Downloading data from %s' % url)
        urlretrieve(url, os.path.join(savepath, filename), reporthook=reporthook)
        print('\nDownload finished!')
    else:
        print('File already exsits!')

    filesize = os.path.getsize(os.path.join(savepath, filename))
    print('File size = %.2f Mb' % (filesize/1024/1024))

    
def match_seq_len(*arrays):
    max_len = np.stack([x.shape[-1] for x in arrays]).max()
    return [np.pad(x, pad_width=((0,0), (0,0), (max_len - x.shape[-1], 0)), mode='constant', constant_values=0) for x in arrays]


def get_predefined_splits(*xs):
    '''xs is a list with X_train, X_valid, ...'''
    splits_ = []
    start = 0
    for x in xs:
        splits_.append(list(np.arange(start, start + len(x))))
        start += len(x)
    return tuple(splits_)


def decompress_from_url(url, target_dir=None, verbose=False):
    # Download
    try:
        fname = os.path.basename(url)
        tmpdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tmpdir, fname)
        download(url, tmpdir)
        # Decompress
        try:
            pv("decompressing data...", verbose)
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            shutil.unpack_archive(tmpfile, target_dir)
            shutil.rmtree(tmpdir)
            pv("...data decompressed", verbose)
            return target_dir
        
        except:
            shutil.rmtree(tmpdir)
            if verbose: sys.stderr.write("Could not decompress file, aborting.\n")

    except:
        shutil.rmtree(tmpdir)
        if verbose:
            sys.stderr.write("Could not download url. Please, check url.\n")


def get_UCR_univariate_list():
    return [
        'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
        'AllGestureWiimoteZ', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken',
        'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration',
        'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
        'CricketZ', 'Crop', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame',
        'DodgerLoopWeekend', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
        'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal',
        'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
        'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain',
        'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
        'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham',
        'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
        'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
        'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
        'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
        'MoteStrain', 'NonInvasiveFetalECGThorax1',
        'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
        'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ',
        'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane',
        'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
        'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
        'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
        'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
        'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
        'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
        'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
        'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
        'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
    ]


def get_UCR_multivariate_list():
    return [
        'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
        'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
        'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
        'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
        'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
        'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports',
        'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits',
        'StandWalkJump', 'UWaveGestureLibrary'
    ]
    
    
def get_UCR_data(dsid, path='.', parent_dir='data/UCR', on_disk=True, mode='c', Xdtype='float32', ydtype=None, return_split=True, split_data=True, 
                 force_download=False, verbose=False):
    UCR_list = get_UCR_univariate_list() + get_UCR_multivariate_list()
    dsid_list = [ds for ds in UCR_list if ds.lower() == dsid.lower()]
    assert len(dsid_list) > 0, f'{dsid} is not a UCR dataset'
    dsid = dsid_list[0]
    return_split = return_split and split_data # keep return_split for compatibility. It will be replaced by split_data
    if dsid in ['InsectWingbeat']:
        warnings.warn(f'Be aware that download of the {dsid} dataset is very slow!')
    pv(f'Dataset: {dsid}', verbose)
    full_parent_dir = Path(path)/parent_dir
    full_tgt_dir = full_parent_dir/dsid
#     if not os.path.exists(full_tgt_dir): os.makedirs(full_tgt_dir)
    full_tgt_dir.parent.mkdir(parents=True, exist_ok=True)
    if force_download or not all([os.path.isfile(f'{full_tgt_dir}/{fn}.npy') for fn in ['X_train', 'X_valid', 'y_train', 'y_valid', 'X', 'y']]):
        # Option A
        src_website = 'http://www.timeseriesclassification.com/Downloads'
        decompress_from_url(f'{src_website}/{dsid}.zip', target_dir=full_tgt_dir, verbose=verbose)
        if dsid == 'DuckDuckGeese':
            with zipfile.ZipFile(Path(f'{full_parent_dir}/DuckDuckGeese/DuckDuckGeese_ts.zip'), 'r') as zip_ref:
                zip_ref.extractall(Path(parent_dir))
        if not os.path.exists(full_tgt_dir/f'{dsid}_TRAIN.ts') or not os.path.exists(full_tgt_dir/f'{dsid}_TRAIN.ts') or \
        Path(full_tgt_dir/f'{dsid}_TRAIN.ts').stat().st_size == 0 or Path(full_tgt_dir/f'{dsid}_TEST.ts').stat().st_size == 0: 
            print('It has not been possible to download the required files')
            if return_split:
                return None, None, None, None
            else:
                return None, None, None
    
        X_train, y_train = load_from_tsfile(full_tgt_dir/f'{dsid}_TRAIN.ts', return_data_type='numpy3d')
        X_valid, y_valid = load_from_tsfile(full_tgt_dir/f'{dsid}_TEST.ts', return_data_type='numpy3d')
        pv('...ts files loaded', verbose)
        pv('preparing numpy arrays...', verbose)
        X_train, X_valid = match_seq_len(X_train, X_valid)  # make sure each sequence length is equal
    
        np.save(f'{full_tgt_dir}/X_train.npy', X_train)
        np.save(f'{full_tgt_dir}/y_train.npy', y_train)
        np.save(f'{full_tgt_dir}/X_valid.npy', X_valid)
        np.save(f'{full_tgt_dir}/y_valid.npy', y_valid)
        np.save(f'{full_tgt_dir}/X.npy', np.concatenate([X_train, X_valid], axis=0))
        np.save(f'{full_tgt_dir}/y.npy', np.concatenate([y_train, y_valid], axis=0))
        del X_train, X_valid, y_train, y_valid
        delete_all_in_dir(full_tgt_dir, exception='.npy')
        pv('...numpy arrays correctly saved', verbose)

    mmap_mode = mode if on_disk else None
    X_train = np.array(np.load(f'{full_tgt_dir}/X_train.npy', mmap_mode=mmap_mode))
    y_train = np.array(np.load(f'{full_tgt_dir}/y_train.npy', mmap_mode=mmap_mode), dtype=float).astype(int)
    X_valid = np.array(np.load(f'{full_tgt_dir}/X_valid.npy', mmap_mode=mmap_mode))
    y_valid = np.array(np.load(f'{full_tgt_dir}/y_valid.npy', mmap_mode=mmap_mode), dtype=float).astype(int)

    if return_split:
        if Xdtype is not None: 
            X_train = X_train.astype(Xdtype)
            X_valid = X_valid.astype(Xdtype)
        if ydtype is not None: 
            y_train = y_train.astype(ydtype)
            y_valid = y_valid.astype(ydtype)
        if verbose:
            print('X_train:', X_train.shape)
            print('y_train:', y_train.shape)
            print('X_valid:', X_valid.shape)
            print('y_valid:', y_valid.shape, '\n')
        return X_train, y_train, X_valid, y_valid
    else:
        X = np.array(np.load(f'{full_tgt_dir}/X.npy', mmap_mode=mmap_mode))
        y = np.array(np.load(f'{full_tgt_dir}/y.npy', mmap_mode=mmap_mode), dtype=int)
        splits = get_predefined_splits(X_train, X_valid)
        if Xdtype is not None: 
            X = X.astype(Xdtype)
        if verbose:
            print('X      :', X .shape)
            print('y      :', y .shape)
            print('splits :', coll_repr(splits[0]), coll_repr(splits[1]), '\n')
        return X, y, splits
    

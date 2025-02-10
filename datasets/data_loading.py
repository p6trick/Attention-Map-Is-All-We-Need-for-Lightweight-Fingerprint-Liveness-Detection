import h5py
import pickle
from datasets.augment import augment, augment_test
import os.path

def check_file(path):
    if os.path.exists(path):
        print(f'Path: {path}')
        return path
    else:
        ext = path.split('.')[-1]
        if ext == 'pkl':
            path.split('.')[-1] = '.hdf5'
            path = path[0] + path[1]
        else:
            path.split('.')[-1] = '.pkl'
            path = path[0] + path[1]
    print(f'Path: {path}')
    return path

def read_hdf(path, phase):
    d = {}
    with h5py.File(path,'r') as f:
        if phase != 'train':
            d['x_test'] = f['x_test'][:]
            d['x_label'] = f['x_label'][:]
            return d
        else:
            d['x_test'] = f['x_test'][:]
            d['x_label'] = f['x_label'][:]
            d['x_train'] = f['x_train'][:]
            d['x_target'] = f['x_target'][:]
            return d

def read_pkl(path, phase):
    d = {}
    with open(path,'rb') as f:
        f_ = pickle.load(f)
    if phase != 'train':
        d['x_test'] = f_['test']['x_test']
        d['x_label'] = f_['test']['x_label']
        return d
    else:
        d['x_test'] = f_['test']['x_test']
        d['x_label'] = f_['test']['x_label']
        d['x_train'] = f_['train']['x_train']
        d['x_target'] = f_['train']['x_target']
        return d


def read_data(path, phase):
    path = check_file(path)
    extension = path.split('.')[-1]
    print(extension)
    print(path)
    if extension == 'hdf5':
        return read_hdf(path, phase)
    elif extension == 'pkl':
        return read_pkl(path, phase)

def prt_data_result(d, phase):
    print('---finish---')
    if phase == 'train':
        print(f"# of x_train: {len(d['x_train'])}")
        print(f"# of x_target: {len(d['x_target'])}")
        print(f"x_train.shape: {d['x_train'][0].shape}")
        print('-------------------------------------------')
        print(f"# of x_test: {len(d['x_test'])}")
        print(f"# of x_label: {len(d['x_label'])}")
        print(f"x_test.shape: {d['x_test'][0].shape}")
    else:
        print(f"# of x_test: {len(d['x_test'])}")
        print(f"# of x_label: {len(d['x_label'])}")
        print(f"x_test.shape: {d['x_test'][0].shape}")
        

def get_dir(path, sensor):
    assert os.path.isdir(path), '%s is not a valid directory' % path
    
    for r,_,fnames in os.walk(path):
        for f in fnames:
            if f.split('_')[0] == sensor:
                path = os.path.join(r,f)
    print(path)
    return path

def data_loading(year, sensor, data_root='../Data/', phase='train'):
    assert year in ['2015', '2019'], 'year = {2015, 2019}'
    assert sensor in ['GreenBit','CrossMatch','DigitalPersona','HiScan','Orcathus'], 'not match sensor'
    
    
    print('x_train, x_test, x_target, x_label')
    if sensor == 'GreenBit':
        path = f'{data_root+year}/{sensor}'
        path = get_dir(path, sensor)
        Data = read_data(path, phase)
        if phase == 'train':
            if len(Data['x_train']) < 10000:
                Data['x_train'], Data['x_target'] = augment(Data['x_train'], Data['x_target'], sensor)
        
            assert len(Data['x_train']) == len(Data['x_target']),'The # of x_train & x_target is different'
        assert len(Data['x_test']) == len(Data['x_label']), 'The # of x_test & x_label is different'
        
        prt_data_result(Data, phase)
        return Data
    
    elif sensor == 'CrossMatch':
        if year == 2019:
            path = f'{data_root+year}/{sensor}/{sensor}_data.hdf5'
        else:
            path = f'{data_root+year}/{sensor}/{sensor}_data.pkl'
        
        
        Data = read_data(path, phase)
        
        if phase == 'train':
            if len(Data['x_train']) < 10000:
                Data['x_train'], Data['x_target'] = augment(Data['x_train'], Data['x_target'], sensor)
            
        
            assert len(Data['x_train'])== len(Data['x_target']),'The # of x_train & x_target is different'
        assert len(Data['x_test']) == len(Data['x_label']), 'The # of x_test & x_label is different'
        
        prt_data_result(Data, phase)
        return Data
    
    elif sensor == 'DigitalPersona':
        if year == 2019:
            path = f'{data_root+year}/{sensor}/{sensor}_data.hdf5'
        else:
            path = f'{data_root+year}/{sensor}/{sensor}_data.pkl'
        
        
        Data = read_data(path, phase)

        if phase == 'train':
            if len(Data['x_train']) < 10000:
                Data['x_train'], Data['x_target'] = augment(Data['x_train'], Data['x_target'], sensor)
            
            assert len(Data['x_train']) == len(Data['x_target']),'The # of x_train & x_target is different'
        assert len(Data['x_test']) == len(Data['x_label']), 'The # of x_test & x_label is different'
        
        prt_data_result(Data, phase)
        return Data
    
    elif sensor == 'HiScan':
        path = f'{data_root+year}/{sensor}/{sensor}_data.pkl'
        
        Data = read_data(path, phase)

        if phase == 'train':
            if len(Data['x_train']) < 10000:
                Data['x_train'], Data['x_target'] = augment(Data['x_train'], Data['x_target'], sensor)
            
        
            assert len(Data['x_train']) == len(Data['x_target']),'The # of x_train & x_target is different'
        assert len(Data['x_test']) == len(Data['x_label']), 'The # of x_test & x_label is different'
        
        prt_data_result(Data, phase)
        return Data
    
    elif sensor == 'Orcathus':
        path = f'{data_root+year}/{sensor}/{sensor}_data.pkl'

        Data = read_data(path, phase)

        if phase == 'train':
            if len(Data['x_train']) < 10000:
                Data['x_train'], Data['x_target'] = augment(Data['x_train'], Data['x_target'], sensor)
            assert len(Data['x_train']) == len(Data['x_target']),'The # of x_train & x_target is different'
        Data['x_test'] = augment_test(Data['x_test'])
        assert len(Data['x_test']) == len(Data['x_label']), 'The # of x_test & x_label is different'

        prt_data_result(Data, phase)
        return Data
    
    elif sensor == 'Biometrika': # train 384 // test 1000 
        path = f'{data_root+year}/{sensor}/{sensor}_Train.hdf5'
        print(f'Path: {path}')
        with h5py.File(path,'r') as f:
            Data['x_train'] = f['x_train'][:]
            Data['x_target'] = f['x_target'][:]
        path = f'{data_root+year}/{sensor}/{sensor}_Test.hdf5'
        print(f'Path: {path}')
        with h5py.File(path,'r') as f:
            Data['x_test'] = f['x_test'][:]
            Data['x_label'] = f['x_label'][:]
         
        
        
        if phase == 'train':
            if len(Data['x_train']) < 10000:
                Data['x_train'], Data['x_target'] = augment(Data['x_train'], Data['x_target'], sensor)
        
            assert len(Data['x_train']) == len(Data['x_target']),'The # of x_train & x_target is different'
        assert len(Data['x_test']) == len(Data['x_label']), 'The # of x_test & x_label is different'
            
 
        prt_data_result(Data, phase)
        return Data

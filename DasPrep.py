import numpy as np
from scipy import signal
# from nptdms import TdmsFile
import h5py
import segyio
import datetime
import os
import shutil
def read_das(fname, **kwargs):
    if fname.lower().endswith('.tdms'):
        # return _read_das_tdms(fname, **kwargs)
        return 0
    elif fname.lower().endswith('.h5'):
        return _read_das_h5(fname, **kwargs)
    
    elif fname.lower().endswith(('.segy', 'sgy')):
        return _read_das_segy(fname, **kwargs)
    
    else:
        print('DAS data format not supported.')

    
def _read_das_h5(fname, **kwargs):
    
    with h5py.File(fname,'r') as h5_file :
    # h5_file = h5py.File(fname,'r')
        nch = h5_file['Acquisition'].attrs['NumberOfLoci']
        metadata = kwargs.pop('metadata', False)
        
        if metadata:
            
            time_arr = h5_file['Acquisition/Raw[0]/RawDataTime/']
            dt = np.diff(time_arr).mean()/1e6
            nt = len(time_arr)
            dx = h5_file['Acquisition'].attrs['SpatialSamplingInterval']
            GL = h5_file['Acquisition'].attrs['GaugeLength']
            headers = dict(h5_file['Acquisition'].attrs)
            h5_file.close()
            return {'dt': dt, 
                    'nt': nt,
                    'dx': dx,
                    'nch': nch,
                    'GL': GL,
                    'headers': headers}   
        else:
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            array_shape = h5_file['Acquisition/Raw[0]/RawData/'].shape
            if array_shape[0] == nch:
                data = h5_file['Acquisition/Raw[0]/RawData/'][ch1:ch2,:]
            else:
                data = h5_file['Acquisition/Raw[0]/RawData/'][:, ch1:ch2].T
            h5_file.close()
            return data
    
    
    
# def _read_das_tdms(fname, **kwargs):
    
    ### https://nptdms.readthedocs.io/en/stable/quickstart.html
    
#     tdms_file = TdmsFile.read(fname) 
#     nch = len(tdms_file['Measurement'])
#     metadata = kwargs.pop('metadata', False)

#     if metadata:
#         dt = 1./tdms_file.properties['SamplingFrequency[Hz]']
#         dx = tdms_file.properties['SpatialResolution[m]']
#         nt = len(tdms_file['Measurement']['0'])
#         GL = tdms_file.properties['GaugeLength']
#         headers = tdms_file.properties
#         return {'dt': dt, 
#                 'nt': nt,
#                 'dx': dx, 
#                 'nch': nch,
#                 'GL': GL,
#                 'headers': headers}
#     else:
#         ch1 = kwargs.pop('ch1', 0)
#         ch2 = kwargs.pop('ch2', nch)
#         data = np.vstack(tdms_file['Measurement'].channels()[ch1:ch2])
# #         data = np.asarray([tdms_file['Measurement'][str(i)] for i in range(ch1, ch2)])
#         return data

    
def _read_das_segy(fname, **kwargs):
    
    # https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/02_segy_quicklook.ipynb
    
    metadata = kwargs.pop('metadata', False)
    
    with segyio.open(fname, ignore_geometry=True) as segy_file:
    
        nch = segy_file.tracecount
        
        if metadata:
            dt = segyio.tools.dt(segy_file) / 1e6
            nt = segy_file.samples.size
            return {'dt': dt, 
                    'nt': nt,
                    'nch': nch}
        else:   
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            data = segy_file.trace.raw[ch1:ch2]
            return data
    
    
def das_preprocess(data_in):
    data_out = signal.detrend(data_in)
    data_out = data_out - np.median(data_out, axis=0) 
    return data_out

def tapering(data, alpha):
    nt = data.shape[1]
    window = signal.windows.tukey(nt, alpha)
    data = data * window[None, :]
    return data


def bandpass(data, dt, fl, fh):
    sos = signal.butter(6, [fl, fh], 'bp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data


def highpass(data, dt, fl):
    sos = signal.butter(6, fl, 'hp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def lowpass(data, dt, fh):
    sos = signal.butter(6, fh, 'lp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data




def get_das_file_time(das_filename):
    das_file_time_str = ' '.join(os.path.splitext(os.path.basename(das_filename))[0].split('_')[-2:])
    return datetime.datetime.strptime(das_file_time_str, '%Y%m%d %H%M%S.%f')

def get_ev_id_in_das_window(event_time_arr, start_time, end_time):
    return np.where((event_time_arr > start_time) & (event_time_arr < end_time))

def get_time_step(start, end, dt):
    return int((start - end).total_seconds() / dt + 1)

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))



def extract_das_data(das_file, ev_time, dt_before, dt_after, save_file_name_prefix, overwrite=False, verbose=False):
    
    das_file_time = np.array([get_das_file_time(das_file[i]) for i in range(len(das_file))])
    
    ev_id_in_das_win = get_ev_id_in_das_window(ev_time, das_file_time.min(), das_file_time.max())
    ev_time_in_das_win = ev_time[ev_id_in_das_win]
    
    ev_time_before = ev_time_in_das_win - datetime.timedelta(minutes=dt_before)
    ev_time_after  = ev_time_in_das_win + datetime.timedelta(minutes=dt_after)

    for iev in range(len(ev_id_in_das_win[0])):
       

        ins_start = np.searchsorted(das_file_time, ev_time_before[iev:(iev+1)])[0] - 1
        ins_end = np.searchsorted(das_file_time, ev_time_after[iev:(iev+1)])[0]

        das_file_time_select = das_file_time[ins_start:ins_end]
        das_file_select = das_file[ins_start:ins_end]

        ev_t0 = ev_time_before[iev]
        ev_t1 = ev_time_after[iev]

        data = []
        for i in range(len(das_file_select)):
            mycopyfile(das_file_select[i] ,save_file_name_prefix + ev_time[iev].strftime('%Y%m%d_%H%M%S') +'/'  )
                
def get_das_file_time(das_filename):
    das_file_time_str = ' '.join(os.path.splitext(os.path.basename(das_filename))[0].split('_')[-2:])
    return np.datetime64(datetime.datetime.strptime(das_file_time_str, '%Y%m%d %H%M%S.%f'))

def read_das_data(filename, ch1, ch2):
    data = read_das(filename, ch1=ch1, ch2=ch2)
    metadata = read_das(filename, metadata=True)
    return data, metadata['dt'], metadata['nt']

# def read_das_data(filename, ch1, ch2):
#     st = _read_segy(filename)
#     data = np.asarray([st[i].data for i in range(ch1, ch2)])
#     return data, st[0].stats['delta'], st[0].stats['npts']

def extended_data_decimate(data_prev, data_curr, data_next, mlist, pad=0.2):
    
    M = mlist.prod()
    
    padding = int(pad*data_curr.shape[1]) // M * M
    data = np.concatenate([data_prev[:,-padding:], data_curr, data_next[:,:padding]], axis=1)
    
    if M>1:
        for m in mlist:
            data = signal.decimate(data, int(m), axis=1, zero_phase=True)

    t1 = padding // M
    t2 = t1 + data_curr.shape[1] // M 
#     print(t1, t2, padding, M, data.shape)
#     assert t2 + padding//M == data.shape[1]
    
    data = data[:,t1:t2]
    
    return data.astype('float32'), data_curr, data_next


def rolling_decimate(continous_data_list, ch1, ch2, mlist):
    data = np.empty((ch2-ch1, 0))
    data_prev, dt, *_ = read_das_data(continous_data_list[0], ch1, ch2)
    data_curr, *_ = read_das_data(continous_data_list[1], ch1, ch2)
    for file in continous_data_list[2:]:
        print(file)
        data_next, *_ = read_das_data(file, ch1, ch2)
        data_deci_new, data_prev, data_curr = extended_data_decimate(data_prev, data_curr, data_next, mlist)
        data = np.concatenate([data, data_deci_new], axis=1)
    mdt = dt * mlist.prod()
    return data, mdt


def das_processing(begin_date, interval, datafile, datafile_time, ch1, ch2, mlist):

    end_date = begin_date + interval 
    datafile_arg_choose = np.where((datafile_time>=begin_date)&(datafile_time<end_date))[0]
    datafile_arg_choose = np.r_[datafile_arg_choose[0]-1, datafile_arg_choose, datafile_arg_choose[-1]+1]
    datafile_choose = [datafile[i] for i in datafile_arg_choose]
    datafile_time_choose = datafile_time[datafile_arg_choose]
        
    data, dt = rolling_decimate(datafile_choose, ch1, ch2, mlist)
    return data, dt

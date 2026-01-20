# -*- coding: utf-8 -*-
"""I/O helpers for SEA-eco signals and metadata."""

#Imports: Typing
from typing import List, Dict, Any, Literal
from collections.abc import Iterable
from numpy.typing import NDArray

#Imports: External
import json
import sys
import os
from glob import glob
from warnings import warn
from h5py import File
import json
from pathlib import Path
import zipfile

import numpy as np

#Imports: Internal
from pySEA.sea_eco.architecture.base_structure_numpy import SEASerializable, Metadata, Dimension, Dimensions, Signal, SignalSet, SignalCollection, SEAFile


def load_memmap_from_npz(path: str, name: str) -> NDArray:
    """Load an array inside an ``npz`` as a read-only memmap.

    Notes
    -----
    NumPy cannot directly memmap arrays embedded in ``npz`` archives; this is a
    lightweight workaround adapted from
    https://github.com/numpy/numpy/issues/5976.
    """
    zf = zipfile.ZipFile(path)
    info = zf.NameToInfo[name + '.npy']
    assert info.compress_type == 0
    offset = zf.open(name + '.npy')._orig_compress_start

    fp = open(path, 'rb')
    fp.seek(offset)
    version = np.lib.format.read_magic(fp)
    assert version in [(1,0), (2,0)]
    if version == (1,0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2,0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    data_offset = fp.tell() # file position will be left at beginning of data
    return np.memmap(path, dtype=dtype, shape=shape,
                     order='F' if fortran_order else 'C', mode='r',
                     offset=data_offset)

def parse_file_path(file_path: str) -> tuple[str, str, str]:
    """Split a file path into directory, stem, and extension.

    If no extension is present, attempt to infer one matching ``.n*`` patterns.
    """
    directory, name = os.path.split(file_path)
    name, extension = os.path.splitext(name)
    if extension=='':
        extension = os.path.splitext(glob(file_path+'.n*')[0])[-1]
    return directory, name, extension

def collect_swift_file(file_path: str, lazy: bool) -> tuple[dict[str, Any], NDArray]:
    """Load Swift-generated data/metadata, handling multiple container types."""
    _, _, file_extension = parse_file_path(file_path)
    file_path = os.path.splitext(file_path)[0]
    data = None

    if file_extension == '.npy':
        with open(file_path+'.json') as f: meta = json.load(f)
        if not lazy:
            data = np.load(file_path+'.npy', mmap_mode='r')
    elif file_extension in [ '.ndata1', '.ndata' ]:
        file = np.load(file_path+file_extension, mmap_mode='r')
        meta = json.loads(file['metadata.json'].decode())
        if not lazy:
            data = load_memmap_from_npz(file_path+file_extension, 'data')
    #elif file_extension == '.ndata':
    #    file = np.load(file_path+'.ndata', mmap_mode='r')
    #    meta = json.loads(file['metadata.json'].decode())
    #    data = load_memmap_from_npz(file_path+'.ndata', 'data')
    elif file_extension == '.h5':
        opened = File(file_path+'.h5')
        meta = json.loads(opened['data'].attrs['properties'])
        if not lazy:
            data = np.asarray(opened['data'])
    else:
        raise Exception(f'The Swift files could not be collected.\nA file with extension `.npy`, `.ndata1`, `.ndata`, or `h5` was not found or provided.\n{file_path}')
    
    return meta, data
        
def swift_to_sea_metadata(swift_metadata:Metadata, 
                           signal_type:Literal['2D-EELS','1D-EELS','Diffraction','Image']|None=None,
                           file_path:str=None, 
                           print_missed:bool=False,
                           read_aberrations=False) -> Metadata:
    """Convert Metadata object constructed from swift metadata to the SEA format of metadata.

    Parameters
    ----------
    swift_metadata : Metadata
        Metadata object constructed from swift json metadata that will be converted to the SEA format of metadata.
    signal_type : Literal['2D-EELS','1D-EELS','Diffraction','Image'] | None, optional
        The type of signal, by default None
    file_path : str, optional
        File path to the original file, by default None
    print_missed : bool, optional
        Print if the metadata attribute was not found, by default False
    read_aberrations : bool, optional
        Read the instrument aberrations, by default False

    Returns
    -------
    Metadata
        SEA format of metadata
    """
    if file_path is not None:
        file_directory, file_name, file_extension = parse_file_path(file_path)
    else:
        file_directory, file_name, file_extension = None, None, None

    map_general = {'title':  'title',
                   'uuid': 'uuid',
                   'timestamp_original': 'datetime_original/local_datetime',
                   'timestamp_modified': 'datetime_modified/local_datetime'}
    map_dimensions = {'is_series':   'is_sequence',
                'nav_ndim':    'collection_dimension_count', #I do not think this includes time. A (t,x,y,E) set has nav=2. is_scan might not be necessary.
                'sig_ndim':    'datum_dimension_count',
                'dimensions':        ['spatial_calibrations','dimensional_calibrations'],
             }

    map_instrument = {'beam_energy':   'metadata/instrument/high_tension', #in keV need to /1e3
                      }
    map_gun = {'beam_energy':   'metadata/instrument/high_tension', #in keV need to /1e3
               }
    
    map_C_gen  = {'state_name': 'metadata/instrument/condenser_setting'}
    map_C_eles = {'C1 ConstW': 'metadata/instrument/ImageScanned/C1 ConstW'}
    abb_C = ['10', '12.a', '12.b',
             '21.a', '21.b', '23.a','23.b',
             '30', '32.a', '32.b', '34.a', '34.b',
             '41.a', '41.b', '43.a','43.b', '45.a','45.b',]
    map_C_abb = {f'C{c}': [f'metadata/instrument/{s}/C{c}' for s in ['ImageScanned', 'ImageRonchigram']] for c in abb_C}

    map_detector = {'name':['metadata/hardware_source/channel_name',
                            'metadata/hardware_source/hardware_source_name'], 
                    'exposure': 'metadata/hardware_source/exposure', 
                    'readout_area': [
                                    'metadata/hardware_source/camera_processing_parameters/readout_area', 
                                    #'metadata/hardware_source/sensor_readout_area_tlbr'
                                    ], 
                    'binning': [
                                'metadata/hardware_source/binning',
                                'metadata/hardware_source/camera_processing_parameters/binning', #[t,l,b,r]#'properties/binning'
                                ],
                    'flip_x': 'metadata/hardware_source/camera_processing_parameters/flip_l_r'}
    map_stage = {'x':       [f'metadata/instrument/{s}/StageOutX' for s in ['ImageScanned', 'ImageRonchigram']], 
                 'y':       [f'metadata/instrument/{s}/StageOutY' for s in ['ImageScanned', 'ImageRonchigram']], 
                 'z':       [f'metadata/instrument/{s}/StageOutZ' for s in ['ImageScanned', 'ImageRonchigram']], 
                 'alpha':   [f'metadata/instrument/{s}/StageOutA' for s in ['ImageScanned', 'ImageRonchigram']], 
                 'beta':    [f'metadata/instrument/{s}/StageOutB' for s in ['ImageScanned', 'ImageRonchigram']], }
    map_scan = {'scan_rotation': 'metadata/scan/rotation_deg',
                'scan_uuid': 'metadata/scan/scan_id',
                'dwell_time': 'metadata/scan/scan_device_parameters/pixel_time_us',
                 }

    map_spectrometer_gen = {'state_name': 'metadata/instrument/ImageScanned/S_EELS'}
    
    maps = {'General': map_general,
            'Instrument': {'_': map_instrument,
                           'Gun': map_gun,
                           'Condensers': {'_':map_C_gen,
                                          'Aberrations':map_C_abb,
                                          'Elements': map_C_eles
                                          },
                           'Detectors': {"Detector":map_detector},
                           'Stage': map_stage,
                           "Scan": map_scan},
            'Dimensions': {'_':map_dimensions},
             }
    
    is_scan = hasattr(swift_metadata.metadata, 'scan')
    if is_scan: maps['Instrument']['Scan'] = map_scan
    if read_aberrations: maps['Instrument']['Aberrations'] = map_C_abb #TODO: Need to conver these from ab to complex

    def recurse_maps(d:dict) -> dict:
        meta_map = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if k == '_':
                    sub_map = recurse_maps(v)
                    meta_map.update(sub_map)
                    continue
                else:
                    v_out = recurse_maps(v)
            else:
                if k == '_':
                    sub_map = recurse_sub_maps(k,v)
                    meta_map.update(sub_map)
                    continue
                else:
                    v_out = recurse_sub_maps(k, v)
            meta_map[k] = v_out

        return meta_map#, dimensions
    
    def recurse_sub_maps(k:str, v:List|str|Metadata) -> Any:
        """Loops through an end level map to parse strings and get values.

        Parameters
        ----------
        k : str
            Key of the sub map.
        v : Any
            Object being parsed.

        Returns
        -------
        Any
            _description_
        """
        
        if not isinstance(v, list): v=[v]
        for i in v:
            v_out = swift_metadata
            for j in i.split('/'): v_out = getattr(v_out, j,None)
            if isinstance(v_out, Metadata):
                v_out = v_out.to_dict()
                if i == 'spatial_calibrations': v_out = {f'dimension_{k2[1:]}': v2 for k2, v2 in v_out.items()}
            if v_out is not None: break
            #return v_out
        if v_out is not None: return v_out
        else:
            if print_missed: print(f'Could not find {v} key in {k}. Setting to None.')
            return None
        #if not v_out: return None
            # except:
            #     if print_missed: print(f'Could not find {i} key in {k}. Setting to None.')
            #     return None

    meta = recurse_maps(maps)
    meta['Dimensions'] = {'is_scan': is_scan, **meta['Dimensions']}
    meta['General']['signal_type'] = signal_type # Note that we do not infer the signal type because this should be a restructure of the swift metadata just restructured.
    det_name = meta['Instrument']['Detectors']['Detector']['name']
    if det_name is None: det_name = 'Unknown'
    else: det_name = det_name.replace(' ','_')
    meta['Instrument']['Detectors'][det_name] = meta['Instrument']['Detectors']['Detector']
    del(meta['Instrument']['Detectors']['Detector'])

    if file_path is not None:
        file_directory, file_name, file_extension = parse_file_path(file_path)
        meta_file = {'orignal_file_path':      file_path+file_extension,
                     'orignal_file_directory': file_directory,
                     'orignal_file_name':      file_name,
                     'orignal_file_extension': file_extension}
        meta['General'].update(meta_file)

    return Metadata(meta)

    #TODO: convert mapped values to complex as below.
    ### Snipet from old funciton. 
    #def read_abberations(self):
    #     if self.meta['metadata']['instrument'].get('ImageScanned') is None:
    #         return None
    #     aber = {k:v for k,v in self.meta['metadata']['instrument']['ImageScanned'].items() if k[0]=='C' and k[1:3].isdigit()}
    #     aber_c = {}
    #     #mags = {}
    #     #angs = {}
    #     for k,v in aber.items():
    #         ab = k[:3]
    #         if k[0]=='C' and k[-1]!='b':
    #             if k[-1] == 'a':
    #                 aber_c[ab] = aber[ab+'.a']+1j*aber[ab+'.b']
    #             else:
    #                 aber_c[ab] = v
    #    return #aber_c

def infer_dimensions(dimensions_meta:Metadata|Dict, data_shape:Iterable[int], signal_type:Literal['2D-EELS','1D-EELS','Diffraction','Image']|None=None, small_angle:bool=True) -> str|None:
    if isinstance(dimensions_meta, Metadata): dimensions_meta = dimensions_meta.to_dict()
    if small_angle: s_pre = 'q'
    else: s_pre = 'θ'
    
    dimensions = dimensions_meta['dimensions']    #[meta[f'dimension_{i}'] for i in range(ndim)]
    ndim = dimensions_meta['nav_ndim']+dimensions_meta['sig_ndim']
    if dimensions_meta['is_series']:
        ndim += 1
        dimensions[0]['units'] = 'frame'
    

    # Infer dimensions space
    pos_dimensions = []
    scat_dimensions = []
    for i, ax in enumerate(dimensions):
        ax['size'] = data_shape[i]
        if ax['units'] in ('um','nm','A','pm'):
            ax['space'] = 'position'
            pos_dimensions.append(i)
        elif ax['units'] == 'rad': 
            ax['space'] = 'scattering'
            scat_dimensions.append(i)
        elif ax['units'] == 'eV':
            ax['space'] = 'spectral'
        elif ax['units'] == 'frame':
            ax['space'] = 'temporal'
        else:
            ax['space'] = None
    
    # Name dimensions
    for ax in dimensions:
        if ax.get('name') is not None: continue
        if ax['space'] == 'position':
            if len(pos_dimensions)==1: ax['name'] = 'x'
            elif len(pos_dimensions)==2: ax['name'] = 'yx'[pos_dimensions.index(dimensions.index(ax))]
            elif len(pos_dimensions)==3: ax['name'] = 'zyx'[pos_dimensions.index(dimensions.index(ax))]
            else: ax['name'] = f'pos{pos_dimensions.index(dimensions.index(ax))}'
        elif ax['space'] == 'scattering':
            if len(scat_dimensions)==1: ax['name'] = s_pre
            elif len(scat_dimensions)==2: ax['name'] = s_pre+'yx'[scat_dimensions.index(dimensions.index(ax))]
            elif len(scat_dimensions)==3: ax['name'] = s_pre+'zyx'[scat_dimensions.index(dimensions.index(ax))]
            else: ax['name'] = f'scat{scat_dimensions.index(dimensions.index(ax))}'
        elif ax['space'] == 'spectral':
            ax['name'] = 'E'
        elif ax['space'] == 'temporal':
            ax['name'] = 't'
        else:
            ax['name'] = f'dimension_{dimensions.index(ax)}'

    
    # Infer signal type if not provided
    if signal_type is None:
        if dimensions[-1]['space'] == 'spectral':
            if dimensions_meta['sig_ndim']==2:
                signal_type = '2D-EELS'
            elif dimensions_meta['sig_ndim']==1:
                signal_type = '1D-EELS'
        elif dimensions_meta['sig_ndim']==2 and dimensions[-1]['space'] == 'scattering' and dimensions[-2]['space'] == 'scattering':
            signal_type == 'Diffraction'
        elif dimensions_meta['sig_ndim']==2 and dimensions[-1]['space'] == 'position' and dimensions[-2]['space'] == 'position':
            signal_type == 'Image'
        else:
            raise Warning('Unrecognized Signal_type: '+str(signal_type)+'. The signal will be set as provided, but no signal dimension inference is provided.')
    

    # Force dimensions based upon signal type
    if signal_type=='2D-EELS':
        dimensions[-2]['name'] = s_pre+r'∥'
        dimensions[-2]['units'] = 'px'
        dimensions[-2]['space'] = 'scattering'
    # else:
    #     print(f'No signal inference is provided for signal_type of {signal_type}')
    
    return signal_type

def load_swift_to_sea(file_path:str,
                       signal_type:str=None, lazy:bool=False,# name:str=None,
                       print_missed:bool=False, read_aberrations=False,
                       **kwargs) -> object:

    swift_meta, data = collect_swift_file(file_path, lazy) #TODO: Only collect json

    if lazy:
        data_shape = swift_meta["data_shape"]
    else:
        data_shape = data.shape

    swift_meta = Metadata(swift_meta)

    meta = swift_to_sea_metadata(swift_meta.deepcopy(), file_path=file_path,
                                 signal_type=signal_type, print_missed=print_missed, read_aberrations=read_aberrations)
    signal_type = infer_dimensions(meta.Dimensions, data_shape, signal_type=signal_type)

    nav_dimensions = list(range(len(data_shape) - meta.Dimensions.nav_ndim-meta.Dimensions.sig_ndim, len(data_shape) - meta.Dimensions.sig_ndim)) if meta.Dimensions.nav_ndim>0 else None
    sig_dimensions = list(range(len(data_shape) - meta.Dimensions.sig_ndim, len(data_shape)))
    dimensions = Dimensions(dimensions=meta.Dimensions.dimensions, nav_dimensions=nav_dimensions, sig_dimensions=sig_dimensions)
    del(meta.Dimensions)

    #if lazy: Warning('Lazy implementation needs introduced.')
    #mmap = 'c' if kwargs.get('lazy') else None
    #data = np.load(file_path+'.npy', mmap_mode=mmap)

    signal = Signal(data=data, metadata=meta, dimensions=dimensions,
                    name=meta.General.title, uuid=meta.General.uuid,
                    signal_type=signal_type,
                    original_metadata=swift_meta)
    del(signal.metadata.General.title)
    del(signal.metadata.General.uuid)
    del(signal.metadata.General.signal_type)
    
    return signal

def _check_and_convert_numpy(value: Any) -> Any:
    """Convert numpy scalars/arrays to native Python containers."""
    if isinstance(value, np.generic):
        return _check_and_convert_numpy(value.item())
    if isinstance(value, np.ndarray):
        return [_check_and_convert_numpy(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        converted = [_check_and_convert_numpy(v) for v in value]
        return type(value)(converted)
    return value


def safe_decode(value: Any) -> Any:
    """Safely decode bytes and numpy values to native Python objects."""
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value
    return _check_and_convert_numpy(value)

def load(file_path: str) -> SEASerializable:
    file_path = Path(file_path)
    
    if file_path.suffix == '.sea' or file_path.suffix == '':
        file_path = str(file_path.with_suffix(''))+'.sea'
        file = File(file_path, "r")
        if len(file)!=1:
            raise ValueError("The hdf5 file contains multiple groups so can not be loaded directly. Consider using `from_hdf5_group` instead to append to the current class.")
        else:
            main_group = file[list(file.keys())[0]]

        if 'sea_type' not in main_group.attrs:
            raise ValueError("Could not locate an SEA group matching this object.")
        else:
            sea_type = safe_decode(main_group.attrs.get('sea_type', b''))
            obj = globals().get(sea_type)()
            obj.from_sea(file_path=file_path)
            return obj

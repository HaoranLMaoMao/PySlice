"""Core numpy-based SEA-eco data structures and serialization helpers."""

#Imports: Typing
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple, Callable, Literal, Union, Sequence
from types import EllipsisType
from collections.abc import Iterable
from matplotlib.axes import Axes as mplAxes
import matplotlib.pyplot as plt
from numpy.typing import NDArray, DTypeLike

#Imports: External
from warnings import warn
from copy import deepcopy
from pathlib import Path
from inspect import signature, Parameter
from functools import wraps


from h5py import Group, File, Dataset
from uuid import UUID, uuid4

import numpy as np
import pickle

#Imports: Internal
from pySEA.sea_eco.architecture import check_dimensions_call
from pySEA.sea_eco.mixins.signal_mixins import SignalProcessingMixin
from pySEA.sea_eco._plotting.plot import PlotImage, plot_nd_array, save_fig, save_image
#from pySEA.sea_eco._plotting.interactive.ipywidget import interactive_signal_plot

def generate_uuid() -> str:
    """Create a random UUID string.

    Returns
    -------
    str
        Newly generated UUID4 identifier.
    """
    return str(uuid4())

def _check_and_convert_numpy(value: Any) -> Any:
    """Convert numpy scalars/arrays to native Python containers.

    Parameters
    ----------
    value : Any
        Arbitrary Python or numpy object to normalize.

    Returns
    -------
    Any
        Native Python type (lists/tuples of built-ins) with numpy scalars
        unwrapped.
    """
    if isinstance(value, np.generic):
        return _check_and_convert_numpy(value.item())
    if isinstance(value, np.ndarray) and len(value.shape)<3:
        return [_check_and_convert_numpy(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        converted = [_check_and_convert_numpy(v) for v in value]
        return type(value)(converted)
    if isinstance(value,str) and value == 'None':
        return None
    return value

def safe_decode(value: Any) -> Any:
    """Safely decode bytes/numpy values to Python-native objects.

    Parameters
    ----------
    value : Any
        Raw value (often from HDF5 attrs/datasets) to decode.

    Returns
    -------
    Any
        Decoded Python object; bytes are UTF-8 decoded when possible, numpy
        types are converted to built-ins.
    """
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value
    return _check_and_convert_numpy(value)

def safe_encode(value: Any) -> Any:
    """Encode Python values (including numpy types) for HDF5 attributes.

    Parameters
    ----------
    value : Any
        Object to encode for storage.

    Returns
    -------
    Any
        UTF-8 encoded bytes for strings/None or numpy-safe Python types.
    """
    if isinstance(value, str):
        return value.encode('utf-8')
    if value is None:
        return 'None'.encode('utf-8')
    return _check_and_convert_numpy(value)

def ask_to_proceed() -> None:
    """Prompt for user confirmation in interactive flows.

    The function blocks until a ``Y``/``N`` (or empty for ``Y``) response is
    entered, printing a short status before returning.

    Returns
    -------
    None

    Notes
    -----
    Intended for CLI/debug workflows; avoid in library code.
    """
    while True:
        user_input = input("Do you want to continue? (Y/N): ").upper()  # Convert to uppercase for case-insensitivity
        if user_input == 'Y' or user_input== '':
            print("Proceeding...")
            break  # Exit the loop once a valid 'Y' is entered
        elif user_input == 'N':
            print("Exiting...")
            break  # Exit the loop once a valid 'N' is entered
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

def get_index_or_none(list_in: List[Any], value: Any) -> int | None:
    """Return the index of ``value`` in ``list_in`` or ``None`` if absent.

    Parameters
    ----------
    list_in : list
        Container to search.
    value : Any
        Target value.

    Returns
    -------
    int | None
        Index of ``value`` or ``None`` when not found.
    """
    try:
        index = list_in.index(value)
        return index
    except ValueError:
        return None

def get_property_dict(obj: object) -> dict:
    """Return the property attributes defined on a class.

    Parameters
    ----------
    obj : object
        Instance whose class properties should be enumerated.

    Returns
    -------
    dict
        Mapping of property names to their current values.
    """
    cls = type(obj)
    return {
        name: getattr(obj, name)
        for name in dir(cls)
        if isinstance(getattr(cls, name), property)
    }

def get_tree_html(obj, recursive_level: int = 0, 
                  exclude_keys: Sequence[str] | None = None,
                  exclude_hidden: bool = True,
                  exclude_properties: bool = False,
                  promote_itterable_keys: Sequence[str] | None = None
                  ) -> str:
    """Build an HTML representation of a SEA-eco object's attribute tree.

    Parameters
    ----------
    obj : Any
        Serializable object implementing the SEA interface.
    recursive_level : int, optional
        Current recursion depth used to control expansion, by default 0.
    exclude_keys : Sequence[str] | None, optional
        Attribute names to omit from the rendered tree, by default None.
    exclude_hidden : bool, optional
        Skip attributes starting with ``_`` when True, by default True.
    exclude_properties : bool, optional
        Skip property attributes when True, by default False.
    promote_itterable_keys : Sequence[str] | None, optional
        Iterable attributes to expand vertically rather than inline, by
        default None.

    Returns
    -------
    str
        HTML unordered list representing the object tree.
    """
    exclude_keys = [] if exclude_keys is None else list(exclude_keys)
    promote_itterable_keys = [] if promote_itterable_keys is None else list(promote_itterable_keys)
    html = ""
    tree_dict = dict(obj.__dict__)
    if not exclude_properties: tree_dict.update(get_property_dict(obj))

    for key, value in tree_dict.items():
        #Handle exclusions and promotions of keys
        if key in exclude_keys: continue
        if exclude_hidden and key.startswith('_'): continue
        
        #build the html tree
        if isinstance(value, SEASerializable):
            html += (
                "<ul style='margin:0px; list-style-position:outside;'>"
                f"<details {'open' if recursive_level < 0 else ''}>"
                "<summary style='display:list-item;'>"
                f"<li style='display:inline;'><b>{key}</b></li></summary>"
                f"{value.get_tree_html(recursive_level+1)}"
                "</details></ul>"
            )
        elif isinstance(value, Sequence) and not isinstance(value, str)and key in promote_itterable_keys:
            html += (
                        "<ul style='margin:0px; list-style-position:outside;'>"
                        f"<details {'open' if recursive_level > 0 else ''}>"
                        "<summary style='display:list-item;'>"
                        f"<li style='display:inline;'><b>{key}</b></li></summary>"
                    )
            for val in value:
                html += (
                    "<ul style='margin:0px; list-style-position:outside;'>"
                    f"<details {'open' if recursive_level > 0 else ''}>"
                    "<summary style='display:list-item;'>"
                    f"<li style='display:inline;'><b>{val}</b></li></summary>"
                    f"{val.get_tree_html(recursive_level+2)}"
                    "</details></ul>"
                )
            html += "</details></ul>"
        else:
            html += (
                "<ul style='margin:0px; list-style-position:outside;'>"
                f"<li style='margin-left:1em; padding-left:0.5em'><b>{key}</b> = {value}</li>"
                "</ul>"
            )
    return html

class SEASerializable(ABC):
    """Base class for objects that can be serialized to/from HDF5.

    Provides shared helpers for converting objects to dictionaries, writing/reading
    SEA-formatted HDF5 files, and rendering tree views for inspection.

    Methods
    -------
    to_dict(hidden=True, properties=False, exclude_keys=None, deep=True)
        Recursively convert the object to a nested dictionary.
    to_hdf5_group(parent_group, force_datasets=None, name=None, exclude_keys=None)
        Persist the object into an HDF5 group with SEA metadata.
    to_sea(file_path, force_datasets=None)
        Serialize into a standalone ``*.sea`` file.
    from_sea(file_path)
        Populate the instance from a ``*.sea`` file.
    from_hdf5_group(group)
        Populate the instance from an existing HDF5 group.
    get_tree_html(...)
        Render an HTML representation of the object tree.
    """

    def _repr_html_(self):
        """Return a compact HTML representation of the Metadata tree."""
        return self.get_tree_html()
    
    def deepcopy(self):
        """Create a deep copy of the object."""
        return deepcopy(self)
    
    def to_dict(self,
                hidden: bool = True,
                properties: bool = False,
                exclude_keys: List[str] = [],
                deep: bool = True
                #promote_itterable_keys: List[str] = [] # Iterables atributes like Dimensions.dimensions or Signals.signals to promote. #?This was used in get_tree_html but I am not sure if it will be needed here.
                ):
        """
        Recursively convert Dimension object to a dictionary.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden attributes (starting with '_'), by default Truee.
        properties : bool, optional
            Include properties, by default False.
        exclude_keys : List[str], optional
            Keys to exclude from the dictionary, by default [].
        deep : bool, optional
            Recursively convert nested SEASerializable objects, by default True.

        Returns
        -------
        dict
            Dictionary representation of the Dimension object.
        """
        to_convert = dict(self.__dict__)
        if properties: to_convert.update(get_property_dict(self))

        result = {}
        for key, value in to_convert.items():
            if not hidden and key.startswith('_'): continue #if converting hiddens then continue to the next key
            if key in exclude_keys: continue #if an exclueded key then continue to the next key
            
            if isinstance(value, SEASerializable):
                if deep: value = value.to_dict(hidden=hidden, properties=properties, deep=deep)
                result[key] = value
            elif isinstance(value, dict): # In case a dict is stored directly
                if deep: value = {k: v.to_dict(deep=deep) if isinstance(v, SEASerializable) else v for k, v in value.items()}
                result[key] = value
            elif isinstance(value, Sequence) and len(value) > 0 and isinstance(value[0], SEASerializable): #If an iterable of SEASerializable is given
                if deep: value = type(value)([v.to_dict(hidden=hidden, properties=properties, deep=deep) for v in value]) #keep the iterable type and loop it converting each element to a dict\
                result[key] = value
            else:
                result[key] = value
        return result

    def to_hdf5_group(self, parent_group: File|Group,
                force_datasets: List = [], 
                name: str | None = None,
                exclude_keys: List[str] = []
                ) -> None:
        """Save object to SEA formated HDF5 group."""
        
        sea_type = type(self).__name__ # Get the SEA class type
        
        # Handle group name: use provided name, fall back to object's name attribute, or default
        if name is None:
            if hasattr(self, 'name') and getattr(self, 'name') is not None: name = getattr(self, 'name')
            else: name = sea_type
        
        # Create the group and assign SEA class type
        group = parent_group.create_group(name, track_order=True)
        group.attrs['sea_type'] = sea_type

        # Iteratively assign attributes by iterating to_dict.
        # Keep the to_dict local so that to_hdf5 can be called for any serializable value.
        to_write = self.to_dict(deep=False, hidden=True,
                                properties=False,
                                exclude_keys=exclude_keys)
        to_write = {**{k:v for k,v in to_write.items() if '_' not in str(k)},
                    **{k:v for k,v in to_write.items() if '_' in str(k)}}
        storage_name_counts: Dict[str, int] = {}
        for key, val in to_write.items():
            if not hasattr(self, key):
                continue
            storage_key = key[1:] if key.startswith('_') else key
            storage_name_counts[storage_key] = storage_name_counts.get(storage_key, 0) + 1
            if storage_name_counts[storage_key] > 1:
                warn(f'HDF5 serialization collision for attribute name \"{storage_key}\". Hidden and public attributes share the same name.')
            #Check the value to see if it is Serializable
            if isinstance(val, SEASerializable):
                val.to_hdf5_group(parent_group=group, name=storage_key)
                continue
            elif isinstance(val, Iterable) and not isinstance(val, str) and len(val) > 0 and all(isinstance(v, SEASerializable) for v in val):
                if key in force_datasets or storage_key in force_datasets:
                    key_group = group
                else:
                    key_group = group.create_group(name=storage_key, track_order=True)
                    key_group.attrs['sea_type'] = val.__class__.__name__ #get the iterable type as string
                for v in val:
                    v.to_hdf5_group(parent_group=key_group)
                continue
            
            # Check the keys to see if it should be a attribute or dataset
            if key in force_datasets or storage_key in force_datasets: 
                group.create_dataset(storage_key, data=val)
            else:
                group.attrs[storage_key] = safe_encode(val)
        return group
            
    def to_sea(self, file_path: str,
                force_datasets: List = []) -> None:
        """Save object to SEA formated HDF5."""
        file_path = Path(file_path)
        if file_path.suffix != '.sea' and file_path.suffix != '':
            raise ValueError("The file extension must be '.sea' or empty.")
        file_path = str(file_path.with_suffix(''))+'.sea'

        file = File(file_path, "w")
        file.attrs['file_type'] = 'SEA-eco HDF5 file'.encode('utf-8')
        file.attrs['file_version'] = '0.0'.encode('utf-8')
        file.attrs['sea_type'] = type(self).__name__.encode('utf-8')

        self.to_hdf5_group(parent_group=file, force_datasets=force_datasets)

        file.close()

    def from_sea(self, file_path: str):
        """Load object data from an HDF5 file."""
        file_path = Path(file_path)
        if file_path.suffix != '.sea' and file_path.suffix != '':
            raise ValueError("The file extension must be '.sea' or empty.")
        file_path = str(file_path.with_suffix(''))+'.sea'
        file = File(file_path, "r")
        
        if len(file)!=1:
            raise ValueError("The hdf5 file contains multiple groups so can not be loaded directly. Consider using `from_hdf5_group` instead to append to the current class.")
        else:
            main_group = file[list(file.keys())[0]]

        if 'sea_type' not in main_group.attrs:
            raise ValueError("Could not locate an SEA group matching this object.")
        elif safe_decode(main_group.attrs['sea_type']) != type(self).__name__:
            raise ValueError(f"The hdf5 group sea_type '{safe_decode(main_group.attrs['sea_type'])}' does not match the current object type '{type(self).__name__}'.")
        else:
            self.from_hdf5_group(main_group)

        file.close()

    def from_hdf5_group(self, group: Group):
        print("reading group",group)
        """Populate the current object from an HDF5 group.
        """
        def _instantiate_child(sub_group: Group):
            """Instantiate an SEASerializable child using a prototype or group metadata."""
            candidate = None
            sea_type = safe_decode(sub_group.attrs.get('sea_type', b''))
            cls = globals().get(sea_type)
            if isinstance(cls, type) and issubclass(cls, SEASerializable):
                try:
                    candidate = cls()
                except TypeError:
                    candidate = cls.__new__(cls)
                    cls.__init__(candidate)
            if candidate is None: return None
            candidate.from_hdf5_group(sub_group)
            return candidate
        def _check_attr_visibility(attr: str) -> str | None:
            """_check_attr_visibility _summary_

            Parameters
            ----------
            attr : str
                Check if the attribute exists as public or private and return the correct name.

            Returns
            -------
            str | None
                _description_
            """
            if hasattr(self, f'_{attr}'): return f'_{attr}'
            else: return attr

        for key, val in group.attrs.items():
            if key == 'sea_type': continue
            if hasattr(val,"shape"):
                print("reading key,val",key,val.shape)
            else:
                print("reading key,val",key,val)
            attr_name = _check_attr_visibility(key)
            if attr_name is not None:
                setattr(self, attr_name, safe_decode(val))

        for key, item in group.items():
            if isinstance(item, Group):
                if 'sea_type' in item.attrs:
                    sea_type = safe_decode(item.attrs.get('sea_type', b''))
                else:
                    warn(f'Group {item.name} has no sea_type attribute. Skipping.')
                    continue
                attr_name = _check_attr_visibility(key)
                if attr_name is None:
                    warn(f'Attribute {key} from group {item.name} not found on target object.')
                    continue
                if  sea_type=='list':
                    new_items = []
                    for sub_item in item.values():
                        child = _instantiate_child(sub_item)
                        if child is not None:
                            new_items.append(child)
                    setattr(self, attr_name, new_items)
                else:
                    child = _instantiate_child(item)
                    if child is not None:
                        setattr(self, attr_name, child)
                    else:
                        warn(f'Attribute {key} could not be instantiated from group {item.name}. Skipping.')
            else:
                value = item[()]
                value = safe_decode(value)
                attr_name = _check_attr_visibility(key)
                if attr_name is None:
                    warn(f'Dataset {key} could not be assigned to target object.')
                    continue
                setattr(self, attr_name, value)

    def save(self, file_path: str) -> None:
        file_path = Path(file_path)
        if file_path.suffix == '.sea' or file_path.suffix == '':
            self.to_sea(file_path)
        elif file_path.suffix == '.pkl':
            with open(file_path,'wb') as f: pickle.dump(self,f)
        else:
            raise ValueError("The file extension must be '.sea', '.pkl' or empty.")
    
    def get_tree_html(self, recursive_level:int=0, 
                      exclude_keys:List[str] = [], 
                      exclude_hidden:bool=True,
                      exclude_properties:bool=False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
        #TODO: Have this work of of self.to_dict() instead of self.__dict__ and put the global get_tree_html() kwargs in self.to_dict
        return get_tree_html(self, recursive_level=recursive_level, 
                             exclude_keys=exclude_keys, 
                             exclude_hidden=exclude_hidden,
                             exclude_properties=exclude_properties,
                             promote_itterable_keys=promote_itterable_keys)

    def get_tree_str(self, pad:str='', recursive_level=None):
        """Print class as a tree.

        Parameters
        ----------
        pad : str, optional
            What to add before each entry, by default ''
        recursive_level : int, optional
           What depth to stop recursing. Not implemented, by default 0

        Returns
        -------
        _type_
            _description_
        """
        if recursive_level is not None:
            #TODO: have the tree stop at a level
            raise NotImplementedError('kwarg recursive_level not implemented yet.')
        string = ''
        N_values = len(self)
        for i, (key, value) in enumerate(self.__dict__.items()):
            if i==N_values-1: cnct = '└── '
            else: cnct = '├── '

            if isinstance(value, Metadata):
                string += f'{pad}{cnct}{key}\n'
                if i== N_values-1: pad_next = '   '
                else: pad_next = '|  '
                string += value.get_tree_str(pad=pad+pad_next)
            else:
                string += f'{pad}{cnct}{key}: {value}\n'
        return string
    
    def show_tree(self, show:Literal['html','str']|None='html', recursive_level=0):
        if show=='html':
            from IPython.display import display, HTML
            display(HTML(self.get_tree_html(recursive_level=recursive_level)))
        elif show=='str': 
            print(self.get_tree_str(self, recursive_level=recursive_level))
        else:
            try:
                from IPython.display import display, HTML
                display(HTML(self.get_tree_html(recursive_level=recursive_level)))
            except:
                print(self.get_tree_str(self, recursive_level=recursive_level))

    def find_attribute(self, attr_name: str, return_first: bool = False, include_hidden: bool = True) -> Dict[str, Any]:
        """
        Recursively search ``self`` for attributes named ``attr_name``.

        Returns
        -------
        Dict[str, Any]
            Mapping of dotted/bracketed paths to the matching attribute values. If
            ``return_first`` is ``True``, traversal stops after the first match (depth-first order).
            If ``include_hidden`` is ``False``, attributes/keys starting with ``_`` are skipped.
        """

        results: Dict[str, Any] = {}
        visited: set[int] = set()

        def _join_path(base: str, key: Any) -> str:
            if isinstance(key, int):
                segment = f"[{key}]"
            elif isinstance(key, str) and key.isidentifier():
                segment = f".{key}"
            elif isinstance(key, str):
                segment = f"[{key!r}]"
            else:
                segment = f"[{key!r}]"

            if base == "":
                return segment.lstrip(".")
            return f"{base}{segment}"

        def _search(target: Any, path: str) -> bool:
            """Depth-first traversal of objects, mappings, and iterables."""
            track = isinstance(target, (SEASerializable, dict)) or (
                isinstance(target, Iterable) and not isinstance(target, (str, bytes, np.ndarray))
            ) or hasattr(target, "__dict__")

            target_id = id(target)
            if track and target_id in visited:
                return False
            if track:
                visited.add(target_id)

            if not include_hidden and isinstance(attr_name, str) and attr_name.startswith("_"):
                return False

            if hasattr(target, attr_name):
                try:
                    value = getattr(target, attr_name)
                except Exception:
                    value = None
                results[_join_path(path, attr_name)] = value
                if return_first:
                    return True

            if isinstance(target, SEASerializable):
                try:
                    children = target.to_dict(hidden=include_hidden, properties=False, deep=False)
                except Exception:
                    children = dict(getattr(target, "__dict__", {}))
                for key, value in children.items():
                    if not include_hidden and isinstance(key, str) and key.startswith("_"):
                        continue
                    if _search(value, _join_path(path, key)):
                        return True
            elif isinstance(target, dict):
                for key, value in target.items():
                    if not include_hidden and isinstance(key, str) and key.startswith("_"):
                        continue
                    if _search(value, _join_path(path, key)):
                        return True
            elif isinstance(target, Iterable) and not isinstance(target, (str, bytes, np.ndarray)):
                for idx, value in enumerate(target):
                    if _search(value, _join_path(path, idx)):
                        return True
            elif hasattr(target, "__dict__"):
                for key, value in vars(target).items():
                    if not include_hidden and isinstance(key, str) and key.startswith("_"):
                        continue
                    if _search(value, _join_path(path, key)):
                        return True
            return False

        _search(self, "")
        return results

    def ensure_uuid(self) -> UUID:
        """Ensure ``self`` exposes a UUID attribute and return it as ``uuid.UUID``."""

        found = self.find_attribute("uuid", return_first=True)
        if found:
            try:
                return UUID(str(next(iter(found.values()))))
            except (TypeError, ValueError):
                pass

        new_uuid = uuid4()
        if hasattr(self, "uuid"):
            try:
                setattr(self, "uuid", str(new_uuid))
            except Exception:
                # Some objects may expose a read-only UUID; ignore assignment failures.
                pass
        elif isinstance(self, dict):
            self["uuid"] = str(new_uuid)
        return new_uuid

    def get_uuid_recursive(self) -> UUID | None:
        """Return a UUID extracted recursively from ``self`` if available."""

        found = self.find_attribute("uuid", return_first=True)
        if not found:
            return None
        try:
            return UUID(str(next(iter(found.values()))))
        except (TypeError, ValueError):
            return None
    
class Metadata(SEASerializable):
    """Hierarchical metadata container used across SEA objects.

    Methods
    -------
    update_from_dict(dictionary)
        Recursively populate nested metadata entries from dictionaries.
    merge(other, kind='skip', warn_duplicate=False, inform_new=False)
        Combine two metadata trees with configurable overwrite/append behavior.
    to_hdf5_group(parent_group, force_datasets=None, name='Metadata')
        Persist metadata into an HDF5 group.
    """

    def __init__(self, meta: dict | None = None) -> None:
        """Create a Metadata tree from a mapping.

        Parameters
        ----------
        meta : dict, optional
            Dictionary to convert into nested ``Metadata`` objects. Defaults to
            an empty mapping.
        """
        meta = {} if meta is None else meta
        self.update_from_dict(dictionary=meta)

    def __len__(self):
        """Return the number of items in the Metadata."""
        return len(self.__dict__)
            
    def __repr__(self):
       """Return a compact HTML representation of the Metadata tree."""
       return self.get_tree_str()
    
    def to_hdf5_group(self, parent_group, force_datasets = [], name = 'Metadata'):
        return super().to_hdf5_group(parent_group, force_datasets, name)

    def update_from_dict(self, dictionary:Dict[str, Dict | List | Any | None]) -> None:
        '''Recursively update Metadata object from a dictionary.

        Parameters
        ----------
        dictionary : dict
            The key will define the Node name.
            If the value is an empty dictionary the value will be asigned as None.
            If the value is a dictionary then the value will be another Metadata object.
            Otherwise the value will be asigned directly.'''
        for key, value in dictionary.items():
            if isinstance(value, dict):
                if len(value)==0: setattr(self, key, None)
                else: setattr(self, key, Metadata(value))
            else:
                setattr(self, key, value)

    def merge(self, other: 'Metadata',
                               kind:Literal['overwrite','append','skip']='skip', warn_duplicate:bool=False, 
                               inform_new=False) -> None:
        """Merge another Metadata object into this one.

        Parameters
        ----------
        other : Metadata
            The other Metadata object to merge.
        kind : bool, optional
            If True, existing keys will be overwritten. Default is False.
        warn : bool, optional
            If True, warn when duplicates arise. Default is False.
        """
        for key, value in other.__dict__.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), Metadata) and isinstance(value, Metadata):
                    getattr(self, key).merge(value, kind=kind, warn_duplicate=warn_duplicate, inform_new=inform_new)
                elif kind=='overwrite':
                    setattr(self, key, value)
                    if warn_duplicate: warn(f'Duplicate - Key {key} already exists. Overwriting.')
                elif kind=='append':
                    existing = getattr(self, key)
                    if existing != value:
                        if not isinstance(value, List): value = [value]
                        if not isinstance(existing, List): existing = [existing]
                        setattr(self, key, value+existing)
                    if warn_duplicate: warn(f'Duplicate - Key {key} already exists. Appending.')
                elif warn_duplicate: warn(f'Duplicate - Key {key} already exists. Use kind=True to overwrite.')
            else:
                setattr(self, key, value)
                if inform_new: print(f'New - Key {key} added to metadata.')

class Dimension(SEASerializable):
    """Calibrated axis descriptor for SEA signals.

    Tracks axis name, units, scale, offset, optional explicit coordinate values,
    and semantic role (navigation vs signal; position/scattering/temporal/
    spectral). Dimensions are serializable to SEA HDF5.

    Methods
    -------
    get_calibrated_value(index, direction='both')
        Convert index/indices to calibrated coordinate(s).
    get_extent()
        Return plotting extents for the axis.
    to_hdf5_group(parent_group, force_datasets=None, name=None)
        Serialize the dimension into an HDF5 group.
    """
    def __init__(self, dimension: Dict|Metadata|None = None,
                 name: str = 'Unnamed Dimension',
                 space: Literal["position", "scattering", "temporal", "spectral"] | None = None,
                 scale: float|int|Iterable[float|int]|None = None,
                 offset: float|int|Iterable[float|int]|None = None,
                 size: int|Iterable[int]|None = None,
                 units: str|Iterable[str] = '',
                 values: NDArray[Any] | Iterable[Any] = None,
                 #device: torch.device = field(default_factory=get_default_device)
                 ) -> None:
        """Initialize a calibrated dimension.

        Parameters
        ----------
        dimension : Dict | Metadata | None, optional
            Existing dimension mapping; when provided, keys seed attributes.
        name : str, optional
            Human-readable axis name, by default 'Unnamed Dimension'.
        space : Literal[&quot;position&quot;, &quot;scattering&quot;, &quot;temporal&quot;, &quot;spectral&quot;] | None, optional
            Semantic category for the axis.
        scale : float | int | Iterable[float | int] | None, optional
            Calibration increment per index step.
        offset : float | int | Iterable[float | int] | None, optional
            Offset applied to index 0.
        size : int | Iterable[int] | None, optional
            Size of the dimension.
        units : str | Iterable[str], optional
            Units of the dimension, by default ''.
        values : NDArray[Any] | Iterable[Any], optional
            Explicit coordinate values; overrides ``scale``/``offset`` when provided.

        Raises
        ------
        TypeError
            If ``dimension`` is not a dict, Metadata, or ``None``.
        """
        #Initialize instance attributes
        self.dimension   = dimension
        self._name   = name
        self.space  = space
        self.scale  = scale
        self.offset = offset
        self.size   = size
        self.units  = units
        self.values = values
        
        defaults = {'name':'Unnamed Dimension', 'space':None, 'scale':None, 'offset':None, 'units':'', 'size':None, 'values':None} #self.__init__.__kwdefaults__
        if self.dimension is not None and not isinstance(self._values, Sequence): #if a dimension and not values is given
            if isinstance(dimension, Metadata): self.dimension = self.dimension.to_dict() #convert to dict if GeneralMatadata
            for k,v in self.dimension.items(): #loop dimension dict and
                if k in defaults.keys() and defaults[k]==getattr(self,k): #asign the key if it is equal to the default
                    setattr(self, k, v)
        elif self.dimension is None and isinstance(self._values, Sequence): #if values and not an dimension is given
            self.size = self._values.shape[-1] #HACK: this should not be size but some sort of tuple that acounts for ndim.
        elif self.dimension is None and not isinstance(self._values, Sequence): #if no values or dimensions are supplied we set the offset and scale. We don't do it by default because these should be None if the dimension is not parametric.
            if self.offset is None: self.offset = 0
            if self.scale is None: self.scale = 1
        else: 
            self._values = None
        del(self.dimension)

    def __str__(self):
        return f'{self.name} (Dimension)'

    def __repr__(self):
        return f'<Dimension name:{self.name} ndim:{self.ndim} size:{self.size}>'

    def __getitem__(self, key: Union[int, float, slice, tuple, EllipsisType]):
        """Support indexing and slicing of the Signal.
        
        Parameters
        ----------
        key : Union[int, float, slice, tuple, EllipsisType]
            Index specification. Can include integers, floats (converted to nearest index),
            slices, ellipsis, or tuples of these.
            - int: direct indexing
            - float: converted to nearest index using dimension calibration
            - slice: regular Python slicing, can include float values
            - Ellipsis: expands to cover remaining dimensions
        """
        if not isinstance(key, tuple): key = (key,)
        
        key_reg = tuple()
        for i, k in enumerate(key):
            if isinstance(k, float):
                k = self.find_nearest_index(k)
            elif isinstance(k, slice):
                start = k.start
                stop = k.stop
                step = k.step
                if isinstance(start, float): start = self.find_nearest_index(start)
                if isinstance(stop, float): stop = self.find_nearest_index(stop)
                if isinstance(step, float):  step = int(round(step / self.scale, 0))
                k = slice(start, stop, step)
            else:
                pass # Ellipse or int do not remove
            key_reg += (k,)
        
        return self.values[key_reg]

    def _check_ndim(self):
        checks = {'size':self.size,
                  'offset':self.offset,
                  'scale':self.scale}
        if all((var is not None for var in checks.values())):
            n_scale = len(self.scale) if isinstance(self.scale,Iterable) else 1
            n_offset = len(self.offset) if isinstance(self.offset,Iterable) else 1
            n_units = len(self.units) if isinstance(self.units,Iterable) and not isinstance(self.units,str) else 1
            if n_scale>1 and n_offset>1 and n_scale!=n_offset: 
                raise ValueError('Scale and offset are >1 dimensions but not consistent dimensionality.')
            if n_units>2 and n_units!=max(n_scale,n_offset): 
                raise ValueError('Units is >1 dimensions but not consistent dimensionality with the larger of scale or offset.')
            return max(n_scale,n_offset)
        elif self._values is not None:
            print(self._values)
            return np.ndim(self._values)
        else:
            warn('The dimensions chould not be determined from the calibrations or values.')
            return None
    
    @property
    def ndim(self):
        return np.ndim(self.values)
    @ndim.setter
    def ndim(self, values: Any) -> UserWarning:
        raise UserWarning('`ndim` is read-only.')

    @property
    def values(self):
        #if self._values is None:
        checks = {'size':self.size,
                  'offset':self.offset,
                  'scale':self.scale}
        if all((var is not None for var in checks.values())):
            self._values = np.arange(self.size)*np.expand_dims(self.scale, axis=-1) + np.expand_dims(self.offset, axis=-1)
        elif self._values is not None:
            pass
        else:
            if all((var is None for var in checks.values())) and self._values is None:
                pass
            else:
                for key, var in checks.items():
                    if key not in checks.items(): warn(f'{key} is None.')
        return self._values
    @values.setter
    def values(self, values):
        self._values = values

    @property
    def name(self): 
        if not isinstance(self._name, str) and isinstance(self._name, Iterable):
            return '('+', '.join(self._name)+')'
        else: return self._name
    @name.setter
    def name(self, value):
        self._name = value
        #self._check_ndim()
    
    def to_dict(self,
                hidden: bool = False,
                properties: bool = True,
                exclude_keys: List[str] = [],
                deep: bool = True
                ):
        """
        Recursively convert Dimension object to a dictionary.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden attributes (starting with '_'), by default False.
        properties : bool, optional
            Include properties, by default True.
        exclude_keys : List[str], optional
            Keys to exclude from the dictionary, by default [].
        deep : bool, optional
            Recursively convert nested SEASerializable objects, by default True.

        Returns
        -------
        dict
            Dictionary representation of the Dimension object.
        """
        return super().to_dict(hidden=hidden, properties=properties, exclude_keys=exclude_keys, deep=deep)

    def to_hdf5_group(self, parent_group: File|Group,
                      force_datasets: List = ['_values'],
                      name: str | None = None
                      ) -> None:
        super().to_hdf5_group(parent_group=parent_group,
                              force_datasets=force_datasets,
                              name=name)
    def to_sea(self, file_path: str,
                force_datasets: List = ['_values']) -> None:
        """Save dimension to HDF5."""
        super().to_sea(file_path,
                         force_datasets=force_datasets)

    def _get_tree_html(self, recursive_level: List[str] = 0, 
                       exclude_keys: List[str]= ['values'], 
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = []
                       ) -> str:
        return super().get_tree_html(recursive_level, 
                                     exclude_keys=exclude_keys, 
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )
    def get_tree_html(self, recursive_level: int = 0, 
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties:bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
                      return self._get_tree_html(recursive_level, 
                                   exclude_keys=exclude_keys+['values'],
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys
                                   )
    
    def get_calibrated_value(self, indices: int | Iterable[int]) -> float:
        """Get calibrated value at a specific index.

        Parameters
        ----------
        indices : int | Iterable[int]
            Indices in axis array.

        Returns
        -------
        float
            Value at the specified index.

        Raises
        ------
        IndexError
            Out of range index.
        """
        if self.ndim > 1:
            if len(indices) != self.ndim: raise ValueError(f"Expected {self.ndim} indices, got {len(indices)}")
            for ind, dim in zip(indices, self.size):
                if ind >= dim: raise IndexError(f"Index {ind} out of range for size {dim}")
        else:
            if np.any(indices >= self.size): raise IndexError(f"Index {indices} out of range for size {self.size}")
        return np.take(self.values, indices=indices)
    
    def find_nearest_index(self, value: float, 
                           direction:Literal['boht','above','below']='both', 
                           warn_bounds=True) -> int:
        """Find the index of the nearest calibrated value.

        Parameters
        ----------
        value : float
            Calibrated value to find the nearest index to.
        direction : Literal['both','above','below']
            Direction to resolve ties. 'both' returns nearest, 'above' returns next higher index, 'below' returns next lower index. Default is 'both'.
        warn_bounds : bool, optional
            Warn if the nearest index is at the bounds of the axis, by default True.

        Returns
        -------
        int
            The nearest index.
        """
        if self.ndim > 1: raise NotImplementedError('find_nearest_index only implemented for 1D dimensions.') #TODO: Implement for multi-D dimensions.
        distances = np.abs(self.values - value)
        index = int(np.argmin(distances))
        if warn_bounds and (index==0 or index==self.size-1):
            print(f'Warning: Nearest index {index} is at the bounds of the axis (0, {self.size-1}).')
        if direction == 'both':
            return index
        elif direction == 'above':
            return index + 1
        elif direction == 'below':
            return index - 1

    def get_extent(self) -> List[float]| List[Tuple[float]]:
        """Get the extent of the dimension for plotting.

        Returns
        -------
        List[float]
            Extent as [min, max].
        """
        if self.ndim == 1:
            return [np.min(self.values[0]), np.max(self.values[-1])]
        if self.ndim > 1:
            return list(zip(np.min(self.values, axis=0), np.max(self.values, axis=0)))

class Dimensions(SEASerializable):
    """Ordered collection of calibrated Dimension objects.

    Maintains navigation/signal role bookkeeping, provides name/index mapping,
    and supplies convenience properties for spectral/temporal/position/
    scattering axes.

    Methods
    -------
    get_dims_as_int(dims)
        Normalize dimension references (names/indices) to integer indices.
    get_extents(kind='Default')
        Return extents suitable for plotting.
    add_dimension(dimension, position=None)
        Insert a new dimension and update bookkeeping.
    remove_dimension(key)
        Remove a dimension by name or index.
    """
    def __init__(self,
                 dimensions: Iterable[Dict|Dimension] = [],
                 nav_dimensions: List[int] = [],
                 sig_dimensions: List[int] = [],
                 ) -> None:
        """Initialize a container of Dimension objects.

        Parameters
        ----------
        dimensions : Iterable[Dict | Dimension], optional
            Dimension definitions or objects in order, by default [].
        nav_dimensions : list[int], optional
            Indices marking navigation axes, by default [].
        sig_dimensions : list[int], optional
            Indices marking signal axes, by default [].

        Raises
        ------
        TypeError
            If an entry in ``dimensions`` is neither a dict nor ``Dimension``.
        """
        self.dimensions = dimensions #! This could be set as private then have __get_item__ iterate the private class.
        self.nav_dimensions = nav_dimensions
        self.sig_dimensions = sig_dimensions
        self.order = list(range(len(self.dimensions)))
        #? The below hidden dimensions are storgage for the get/set property. The getter always stores in the hidden, so is there any point in having the hiddens? I guess as is, the set would store in the hidden, then that hidden could be accessed directly allowing the user to hack if needed.
        self._spectral_dimension = []
        self._temporal_dimension = []
        self._position_dimensions = []
        self._scattering_dimensions = []
    
        dimensions_list = []
        for i, dimension in enumerate(self.dimensions):
            if isinstance(dimension, Dict):
                dimension_obj = Dimension(dimension)
            elif isinstance(dimension, Dimension):
                dimension_obj = dimension
            else:
                raise TypeError(f'Dimensions iterable value of {type(dimension)} was provided but is not an allowed type.')
            dimensions_list.append(dimension_obj)
        self.dimensions = dimensions_list
    
    @property
    def ndim(self) -> int:
        """Get the total number of dimensions across all dimensions."""
        ndim = np.sum([ax.ndim for ax in self.dimensions], dtype=int)
        return ndim
    @ndim.setter
    def ndim(self, value:Any) -> UserWarning:
        raise UserWarning('ndim should not be set by the user.')

    @property
    def spectral_dimension(self) -> int:
        for i, dimension in enumerate(self.dimensions):
            if dimension.space=='spectral':
                self._spectral_dimension = i
                break
        return self._spectral_dimension
    @spectral_dimension.setter
    def spectral_dimension(self, value:int) -> None:
        self._spectral_dimension = value

    @property
    def temporal_dimension(self) -> int:
        for i, dimension in enumerate(self.dimensions):
            if dimension.space=='temporal':
                self._temporal_dimension = i
                break
        return self._temporal_dimension
    @temporal_dimension.setter
    def temporal_dimension(self, value:int) -> None:
        self._temporal_dimension = value

    @property
    def position_dimensions(self) -> List:
        if len(self._position_dimensions)==0: # first run only, assemble based on dimension.space
            for i, dimension in enumerate(self.dimensions):
                if dimension.space=='position':
                    self._position_dimensions.append(i)
        return self._position_dimensions
    @position_dimensions.setter
    def position_dimensions(self, value:List) -> None:
        self._position_dimensions = value

    @property
    def scattering_dimensions(self) -> List:
        if len(self._scattering_dimensions)==0: # first run only, assemble based on dimension.space
            for i, dimension in enumerate(self.dimensions):
                if dimension.space=='scattering':
                    self._scattering_dimensions.append(i)
        return self._scattering_dimensions
    @scattering_dimensions.setter
    def scattering_dimensions(self, value:List) -> None:
        self._scattering_dimensions = value
    
    def __repr__(self):
        return f'<Dimensions ndim:{self.ndim} dimensions:[{", ".join([ax.name for ax in self.dimensions])}]>'
    
    def __getitem__(self, key:int|str|Iterable[int|str]):
        if isinstance(key, int):
            if key<0: return self.dimensions[len(self)+key]
            else: return self.dimensions[key]
        elif isinstance(key, str): return self.dimensions[self.get_index_from_name(key)]
        elif isinstance(key, Iterable):
            ret = []
            for k in key:
                if isinstance(k, int):
                    if k<0: ret.append(self.dimensions[len(self)+k])
                    else: ret.append(self.dimensions[k])
                elif isinstance(k, str): ret.append(self.dimensions[self.get_index_from_name(k)])
            return ret
        else: raise TypeError(f'Only integers and strings are allowed but a {type(key)} was provided.')
    
    def __len__(self):
        return len(self.dimensions)

    def add_dimension(self, dimension:Dict|Dimension) -> None:
        """Add an dimension to the Dimensions object.

        Parameters
        ----------
        dimension : Dict | Dimension
            Dictionry with dimension calibrations or Dimension class to add.

        Raises
        ------
        TypeError
            If the provided dimension is not a dictionary or Dimension object.
        """
        dimension_n = len(self.dimensions)
        if isinstance(dimension, Dict):
            dimension = Dimension(dimension)
        if isinstance(dimension, Dimension):
            self.dimensions.append(dimension)
        else: raise TypeError(f'Dimensions iterable value of {type(dimension)} was provided but is not an allowed type.')

        self.order.append(dimension_n)
    def remove_dimension(self, dimension:int|str):
        if isinstance(dimension, str): dimension = self.get_index_from_name(dimension)
        del self.dimensions[dimension]
        self.order = [i if i < dimension else i - 1 for i in self.order if i != dimension]

    def get_names(self) -> List[str]:
        return [ax.name for ax in self.dimensions]

    def get_index_from_name(self, name:str) -> int:
        names = self.get_names()
        if name in names: return names.index(name)
        else: raise KeyError(f'A key of {name} was provided and the dimensions names are {names}')
    
    def get_dims_as_int(self, 
                        dims: str | int | Iterable[str | int] | None
                        ) -> List | int:
        """Convert named or integer indices to integers.

        Parameters
        ----------
        dims : str | int | Iterable[str  |  int] | None
            The index to convert.

        Returns
        -------
        List | int
            List of integer indicies

        Raises
        ------
        IndexError
            _description_
        IndexError
            _description_
        """
        # Convert to int, tuple(int), or None
        if dims is None: out = None
        elif isinstance(dims, str): out = self.get_index_from_name(dims)
        elif isinstance(dims, int):
            if dims > len(self):
                raise IndexError(f'Axis index {dim} is out of bounds for signal with {self.ndim} dimensions.')
            else:
                out = dims if dims>=0 else len(self) + dims
        elif isinstance(dims, Iterable):
            out = []
            for dim in dims:
                if isinstance(dim, str):
                    out.append(self.get_index_from_name(dim))
                elif isinstance(dim, int):
                    if dim > self.ndim:
                        raise IndexError(f'Axis index {dim} is out of bounds for signal with {self.ndim} dimensions.')
                    else:
                        out.append(dim if dim>=0 else len(self.dimensions) + dim)
        return out

    def to_dict(self,
                hidden: bool = False,
                properties: bool = True,
                exclude_keys: List[str] = [],
                deep: bool = True
                ):
        """
        Recursively convert Dimension object to a dictionary.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden attributes (starting with '_'), by default False.
        properties : bool, optional
            Include properties, by default True.
        exclude_keys : List[str], optional
            Keys to exclude from the dictionary, by default [].
        deep : bool, optional
            Recursively convert nested SEASerializable objects, by default True.

        Returns
        -------
        dict
            Dictionary representation of the Dimension object.
        """
        return super().to_dict(hidden=hidden, properties=properties, exclude_keys=exclude_keys, deep=deep)

    def to_hdf5_group(self, parent_group: File|Group,
                      force_datasets: List = [],
                      name: str | None = None
                      ) -> None:
        force_datasets = force_datasets#?+['dimensions']
        super().to_hdf5_group(parent_group=parent_group,
                              force_datasets=force_datasets,
                              name=name)
    def to_sea(self, file_path: str,
                force_datasets: List = []) -> None:
            """Save dimension to HDF5."""
            force_datasets = force_datasets#?+['dimensions']
            super().to_sea(file_path,
                            force_datasets=force_datasets)

    def _get_tree_html(self, recursive_level: List[str] = 0, 
                       exclude_keys: List[str] = [], 
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = ['dimensions']
                       ) -> str:
        return super().get_tree_html(recursive_level, 
                                     exclude_keys=exclude_keys, 
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )
    def get_tree_html(self, recursive_level: int = 0, 
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties:bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
                      return self._get_tree_html(recursive_level, 
                                   exclude_keys=exclude_keys,
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys + ['dimensions']
                                   )

    def get_extents(self, kind: Literal['Axes','Image'] = 'Axes') -> List[float, List[float], List[Tuple[float]]]:
        """Get the extents of all dimensions for plotting.

        Returns
        -------
        List[float, Tuple[float]]
            List of extents as [min, max] or list of (min, max) tuples for multi-D dimensions.
        """
        extents = [dim.get_extent() for dim in self.dimensions]
        hf_sc = [dim.scale/2 for dim in self.dimensions]

        if kind=='Image' and self.ndim==2:
            # Flatten one level so extents becomes a single list like [xmin, xmax, ymax, ymin]
            extents = [extents[1][0]-hf_sc[1], extents[1][1]+hf_sc[1], 
                       extents[0][1]+hf_sc[0], extents[0][0]-hf_sc[0]]
        return extents


dims_set_maps = {'signal':'sig_dimensions', 'navigation':'nav_dimensions',
                 'temporal':'temporal_dimension', 'spectral':'spectral_dimension',
                 'position':'position_dimensions', 'scattering':'scattering_dimensions'}
dims_set_Lit = Literal['signal', 'navigation', 'temporal', 'spectral', 'position', 'scattering']

class Signal(SEASerializable, SignalProcessingMixin):
    """N-dimensional signal with calibrated dimensions and metadata.

    Wraps ndarray data with `Dimensions`, `Metadata`, and plotting/processing
    helpers while preserving provenance (UUIDs) and SEA serialization support.

    Methods
    -------
    deepcopy_with_new_data(data)
        Clone the signal, replacing the underlying data.
    deepcopy_with_reduced_data_dim(data, keep_dim)
        Clone the signal while adjusting dimensions after reduction.
    show(...), image(...)
        Plot or export arrays with calibrated axes.
    to_sea(file_path, force_datasets=None)
        Serialize to SEA HDF5.
    """

    def __init__(self, data: NDArray|None = None,
                 name: str = 'Signal',
                 uuid: str = generate_uuid(),
                 dimensions: Dimensions|List|None = None, #BUG Not sure if Dict will work
                 signal_type: Literal['2D-EELS','1D-EELS','Diffraction','Image']|None = None,
                 dimensions_domain:Literal['local','global'] = 'local',
                 #? metadata_domain: Literal['local','global'] = 'local',
                  original_metadata: Metadata|None = None,
                  is_lazy: bool = False,
                  metadata: Metadata|Dict|None = None
                  ):
        """Create a signal with calibrated dimensions and metadata.

        Parameters
        ----------
        data : NDArray | None, optional
            Underlying array. Can be ``None`` for delayed population.
        name : str, optional
            Signal name, by default 'Signal'.
        uuid : str, optional
            UUID identifier, by default a new UUID4 string.
        dimensions : Dimensions | list | None, optional
            Dimension definitions; accepts a ``Dimensions`` object or iterable
            convertible to one. Defaults to ``None``.
        signal_type : {'2D-EELS','1D-EELS','Diffraction','Image'} | None, optional
            Semantic type hint for downstream tools.
        dimensions_domain : {'local','global'}, optional
            Whether dimensions are owned locally or reference a shared set,
            by default 'local'.
        original_metadata : Metadata | None, optional
            Metadata before processing, by default None.
        is_lazy : bool, optional
            Flag indicating delayed data loading, by default False.
        metadata : Metadata | dict | None, optional
            Attached metadata; dicts are converted to ``Metadata``.
        """
        self.data = data
        self.name = name
        self.uuid = uuid
        self.dimensions_domain = dimensions_domain
        self.dimensions = dimensions
        #self.dimensions = dimensions if isinstance(dimensions, Dimensions) else Dimensions(dimensions)
        self.signal_type = signal_type
        #self.metadata_domain = metadata_domain
        self._original_metadata = original_metadata
        self.is_lazy = is_lazy
        if isinstance(metadata, Dict):
            metadata = Metadata(metadata)
        self.metadata = metadata
        
        self._fold_state = None

        self._parent: "SignalCollection | None" = None
        self._dimensions_shared: Dimensions | None = None

        #HACK This is gross. should make Metadata sliceable like dict or with ints. Also not sure this will work when not available.
        #? Might be worth thinking about where this should go. Could go in meta.instrument, even if it means all but instrument.detector is promoted, but this depends on how meta.instrument will be handled afer promotion.
        self.detector: str = list(self.metadata.Instrument.Detectors.to_dict().keys())[0] if self.metadata is not None else None
    
    @property
    def dimensions(self):
        """Return the Dimensions currently applied to the Signal.

        Returns
        -------
        Dimensions | None
            Shared dimensions when the signal is in ``'global'`` mode and bound
            to a shared object; otherwise the local dimensions. If no shared
            dimensions are bound while in global mode, the local dimensions are
            returned as a fallback.
        """
        if self.dimensions_domain == 'global':
            if self._dimensions_shared is not None:
                return self._dimensions_shared
            return self._local_dimensions
        return self._local_dimensions
    @dimensions.setter
    def dimensions(self, dimensions):
        """Set local dimensions and detach from any shared binding.

        Parameters
        ----------
        dimensions : Dimensions or array-like or None
            Dimensions to assign locally. Non-``Dimensions`` inputs are coerced.
        """
        if dimensions is not None and not isinstance(dimensions, Dimensions):
            dimensions = Dimensions(dimensions)
        self._local_dimensions = dimensions
        self._dimensions_shared = None
        self.dimensions_domain = 'local'

    def bind_shared_dimensions(self, dims: Dimensions) -> None:
        """Bind the signal to shared dimensions.

        Parameters
        ----------
        dims : Dimensions
            Shared dimensions object to reference live.

        Notes
        -----
        Binding sets ``dimensions_domain`` to ``'global'`` so future mutations
        to the shared ``Dimensions`` are reflected in this signal.
        """
        if not isinstance(dims, Dimensions):
            raise TypeError(f'Shared dimensions must be a Dimensions instance, received {type(dims)}.')
        self._dimensions_shared = dims
        self.dimensions_domain = 'global'

    def detach_shared_dimensions(self) -> None:
        """Detach from shared dimensions by snapshotting the current layout.

        Notes
        -----
        The current resolved dimensions are deep-copied into the local store and
        ``dimensions_domain`` is set to ``'local'``.
        """
        self._local_dimensions = self.dimensions.deepcopy()
        self._dimensions_shared = None
        self.dimensions_domain = 'local'

    @property
    def original_metadata(self) -> Metadata:
        return self._original_metadata
    @original_metadata.setter
    def original_metadata(self, values: Any) -> UserWarning | None:
        if self._original_metadata is None: self._original_metadata = values
        else: raise UserWarning('original_metadata is read-only. If it is necessary to change the original metadata use `_original_metadata` to set the value, but we do recomend against changing such values.')

    def __str__(self):
        if self.name is None: name = 'Unnamed'
        else: name = self.name
        return f'{name} (Signal)'

    def __repr__(self):
        return f'<Signal name="{self.name}" signal_type={self.signal_type} dimensions_domain={self.dimensions_domain}>'
    
    def __getitem__(self, key: Union[int, float, slice, tuple, EllipsisType]):
        """Support indexing and slicing of the Signal.
        
        Parameters
        ----------
        key : Union[int, float, slice, tuple, EllipsisType]
            Index specification. Can include integers, floats (converted to nearest index),
            slices, ellipsis, or tuples of these.
            - int: direct indexing
            - float: converted to nearest index using dimension calibration
            - slice: regular Python slicing, can include float values
            - Ellipsis: expands to cover remaining dimensions

        ToDo
        ----
        """
        if not isinstance(key, tuple): key = (key,)
            
        # Track which dimensions remain and their new sizes
        remaining_dims = self.dimensions.dimensions.copy()
        new_sizes = []
        
        # Handle Ellipsis expansion
        n_indices = len(key)
        n_dims = len(self.dimensions.dimensions)
        ellipsis_pos = None
        
        for i, k in enumerate(key):
            if k is Ellipsis:
                ellipsis_pos = i
                break
        
        if ellipsis_pos is not None:
            # Calculate how many dimensions the Ellipsis represents
            n_extra = n_dims - (n_indices - 1)
            # Replace Ellipsis with appropriate number of slice(None)
            expanded_key = key[:ellipsis_pos] + (slice(None),) * n_extra + key[ellipsis_pos + 1:]
        else:
            expanded_key = key
            
        # Ensure we don't have too many indices
        if len(expanded_key) > n_dims:
            raise IndexError(f'Too many indices: array is {n_dims}-dimensional, but {len(expanded_key)} were indexed')
        
        # Pad with full slices if we have too few indices
        if len(expanded_key) < n_dims:
            expanded_key = expanded_key + (slice(None),) * (n_dims - len(expanded_key))
        
        key_reg = tuple()
        # Now process each key with the proper dimension
        for i, k in enumerate(expanded_key):
            orig_dim_i = self.dimensions.order[i]
            orig_dim = self.dimensions.dimensions[orig_dim_i]
            if isinstance(k, int):
                remaining_dims.remove(orig_dim)
            elif isinstance(k, float):
                k = orig_dim.find_nearest_index(k)
                # This dimension is removed by indexing
                remaining_dims.remove(orig_dim)
            elif isinstance(k, slice):
                start = k.start
                stop = k.stop
                step = k.step
                if isinstance(start, float): start = orig_dim.find_nearest_index(start)
                if isinstance(stop, float): stop = orig_dim.find_nearest_index(stop)
                if isinstance(step, float):  step = int(round(step / orig_dim.scale, 0))
                k = slice(start, stop, step)
                # This dimension remains but might have a new size
                # Calculate new size for sliced dimension
                slice_indices = k.indices(orig_dim.size)
                new_size = len(range(*slice_indices))
                new_sizes.append(new_size)
            else:
                # Ellipse do not remove
                pass
            key_reg += (k,)
        
        sliced_data = self.data[key_reg]
        
        # Create new signal with updated dimensions
        new_signal = self.deepcopy()
        new_signal.data = sliced_data
        
        # Update dimensions for the new signal
        if remaining_dims:
            new_dims = Dimensions(
                dimensions=[dim.deepcopy() for dim in remaining_dims],
                nav_dimensions=[i for i, dim in enumerate(remaining_dims) 
                              if dim in [self.dimensions[d] for d in self.dimensions.nav_dimensions]],
                sig_dimensions=[i for i, dim in enumerate(remaining_dims) 
                              if dim in [self.dimensions[d] for d in self.dimensions.sig_dimensions]],
            )
            # Update sizes for sliced dimensions
            for dim, new_size in zip(new_dims.dimensions, new_sizes):
                dim.size = new_size
            new_signal.dimensions = new_dims
        
        return new_signal
    
    def __algebra_wrapper(op: Callable[[NDArray, NDArray | float | int], NDArray],
                          reverse: bool = False) -> Callable[["Signal", int | float | "Signal" | NDArray], "Signal"]:
        """
        Factory for binary algebra dunder methods.

        Parameters
        ----------
        op : function
            A binary function like operator.add, operator.sub, etc.,
            taking (left, right) and returning an NDArray.
        reverse : bool
            If True, operands are swapped: op(other, self) instead of op(self, other).
            Used for __rsub__, __rtruediv__, etc.
        """
        @wraps(op)
        def method(self: "Signal", value: int | float | "Signal" | NDArray) -> "Signal":
            # Normalize `value` to something we can combine with self.data
            if isinstance(value, (int, float)):
                other = value
            elif isinstance(value, Signal):
                other = value.data
            elif isinstance(value, np.ndarray):
                other = value
            else:
                raise TypeError(
                    f"A disallowed type is being combined with Signal. "
                    f"Type: {type(value)}"
                )

            # If it's an array, enforce shape compatibility
            if isinstance(other, np.ndarray) and other.shape != self.data.shape:
                raise ValueError(
                    f"The array and signal are not compatible shapes. "
                    f"Array shape of {other.shape} does not match the signal {self.data.shape}"
                )

            left, right = (other, self.data) if reverse else (self.data, other)
            result = op(left, right)

            return self.deepcopy_with_new_data(result)

            # Note: we rely on deepcopy_with_new_data to clone metadata, etc.

        return method
    
    # ------------------------------------------------------------------
    # Dunder methods built from the wrapper
    # ------------------------------------------------------------------
    import operator as _op

    __add__      = __algebra_wrapper(_op.add)
    __radd__     = __algebra_wrapper(_op.add, reverse=True)

    __sub__      = __algebra_wrapper(_op.sub)
    __rsub__     = __algebra_wrapper(_op.sub, reverse=True)

    __mul__      = __algebra_wrapper(_op.mul)
    __rmul__     = __algebra_wrapper(_op.mul, reverse=True)

    __truediv__  = __algebra_wrapper(_op.truediv)
    __rtruediv__ = __algebra_wrapper(_op.truediv, reverse=True)

    __floordiv__  = __algebra_wrapper(_op.floordiv)
    __rfloordiv__ = __algebra_wrapper(_op.floordiv, reverse=True)

    __pow__      = __algebra_wrapper(_op.pow)
    __rpow__     = __algebra_wrapper(_op.pow, reverse=True)

    #TODO: Implement a wrapper for numpy functions that modifies the dimensions accordingly
    def __array__(self, dtype:DTypeLike=None) -> Any:
        """Allow numpy to treat this object as an array."""
        if dtype is None:
            return self.data
        return self.data.astype(dtype)
    
    def __array_function__(self, func, types, args, kwargs):
        """Handle numpy array functions like sum(), mean(), etc.
        
        This is called by numpy when array-like objects are passed to numpy functions.
        It differs from __array_ufunc__ which handles element-wise operations.
        """
        # Get the actual array data from any Signal objects
        arrays = []
        signal_inputs = []
        for arg in args:
            if isinstance(arg, Signal):
                arrays.append(arg.data)
                signal_inputs.append(arg)
            else:
                arrays.append(arg)
        
        # Convert any dimension references and get the dimension key being used
        dim_key = check_dimensions_call(func, kwargs)
        if dim_key:
            kwargs[dim_key] = self.dimensions.get_dims_as_int(kwargs[dim_key])
            if isinstance(kwargs[dim_key], List): kwargs[dim_key] = tuple(kwargs[dim_key])
            if isinstance(kwargs[dim_key], int | str | Iterable): dims_to_remove = np.atleast_1d(kwargs[dim_key])
            else: raise KeyError(f'{dim_key} takes int, str, or an Iterable and {kwargs[dim_key]} was provided.')
            remaining_dims = {i: dim for i, dim in enumerate(self.dimensions.dimensions) 
                              if i not in dims_to_remove}
        
        # Call the numpy function with our data arrays
        result = func(*arrays, **kwargs)

        # If the result is an array, wrap it in a Signal
        if isinstance(result, np.ndarray):
            if np.ndim(result) == self.data.ndim:
                result = self.deepcopy_with_new_data(result)
            elif np.ndim(result) < self.data.ndim:
                result = self.deepcopy_with_reduced_data_dim(data=result, keep_dim=remaining_dims)                        
        return result

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs) -> Any:
        """Handle numpy universal functions (ufuncs).
        
        Parameters
        ----------
        ufunc : np.ufunc
            A numpy function that is applied to the input data.
        method : str
            How the ufunc operates on the inputs. This partially handles dimensionality expansion and contraction.
        *inputs : tuple
            Input arrays to the ufunc.
        **kwargs : dict
            Additional keyword arguments to pass to ufunc.
        """
        # Convert any Signal objects in inputs to their data arrays
        args = []
        signal_inputs = []
        for i in inputs:
            if isinstance(i, Signal):
                args.append(i.data)
                signal_inputs.append(i)
            else:
                args.append(i)

        dim_key = check_dimensions_call(ufunc, kwargs)
        
        # Call the ufunc on the underlying data
        result = getattr(ufunc, method)(*args, **kwargs)
        
        # Convert any dimension references and get the dimension key being used
        if dim_key:
            kwargs[dim_key] = self.dimensions.get_dims_as_int(kwargs[dim_key])
            if isinstance(kwargs[dim_key], List): kwargs[dim_key] = tuple(kwargs[dim_key])
            if isinstance(kwargs[dim_key], int | str | Iterable): dims_to_remove = np.atleast_1d(kwargs[dim_key])
            else: raise KeyError(f'{dim_key} takes int, str, or an Iterable and {kwargs[dim_key]} was provided.')
            remaining_dims = {i: dim for i, dim in enumerate(self.dimensions.dimensions) 
                              if i not in dims_to_remove}

        # If the result is an array, wrap it in a Signal
        if isinstance(result, np.ndarray):
            if np.ndim(result) == self.data.ndim:
                result = self.deepcopy_with_new_data(result)
            elif np.ndim(result) < self.data.ndim:
                result = self.deepcopy_with_reduced_data_dim(data=result, keep_dim=remaining_dims)                        
        return result

    def deepcopy(self):
        """Copy the signal"""
        out = super().deepcopy()
        setattr(out, 'uuid', generate_uuid())
        return out
    def deepcopy_with_new_data(self, data:np.ndarray):
        """Copy the signal with new data."""
        out = self.deepcopy()
        out.data = data
        return out
    def deepcopy_with_reduced_data_dim(self, data:np.ndarray, keep_dim:Tuple[int|str]) -> Signal: #!
        """Copy the signal with new data of reduced dimensionality."""
        out = self.deepcopy()
        out.data = data
        
        # Update dimensions based on the ufunc operation                
        new_dims = Dimensions(
            dimensions=[out.dimensions[i] for i in keep_dim],
            nav_dimensions=[i for i, dim in enumerate(keep_dim) 
                            if dim in [self.dimensions[d] for d in self.dimensions.nav_dimensions]],
            sig_dimensions=[i for i, dim in enumerate(keep_dim) 
                            if dim in [self.dimensions[d] for d in self.dimensions.sig_dimensions]],
        )
        out.dimensions = new_dims
    
        return out

    def generate_child_signal(self, *args, **kwargs):
        Signal.__init__.__doc__
        child = Signal(*args, **kwargs)
        child.input_uuid = self.uuid
        return child

    def to_dict(self,
                hidden: bool = False,
                properties: bool = True,
                exclude_keys: List[str] = ['data'],
                deep: bool = True
                ):
        """
        Recursively convert Dimension object to a dictionary.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden attributes (starting with '_'), by default False.
        properties : bool, optional
            Include properties, by default True.
        exclude_keys : List[str], optional
            Keys to exclude from the dictionary, by default [].
        deep : bool, optional
            Recursively convert nested SEASerializable objects, by default True.

        Returns
        -------
        dict
            Dictionary representation of the Dimension object.
        """
        return super().to_dict(hidden=hidden, properties=properties, exclude_keys=exclude_keys, deep=deep)
    
    def to_hdf5_group(self, parent_group: File|Group,
                      force_datasets: List = ['data'],
                      name: str | None = None
                      ) -> None:
        """Save the signal to an HDF5 group.

        Parameters
        ----------
        parent_group : File or Group
            HDF5 file or group to write into.
        force_datasets : list, optional
            Attributes to coerce into datasets, by default ``['data']``.
        name : str or None, optional
            Group name override, by default ``None``.

        Notes
        -----
        When the signal is using shared dimensions (``dimensions_domain == 'global'``),
        the dimensions are deep-copied into the local store for serialization so the
        exported signal remains self-contained. The runtime-only shared binding
        (``_dimensions_shared``) is excluded from serialization.
        """
        shared_backup = self._dimensions_shared
        local_backup = getattr(self, '_local_dimensions', None)
        domain_backup = self.dimensions_domain
        if self.dimensions_domain == 'global':
            self._local_dimensions = deepcopy(self.dimensions)
            self.dimensions_domain = 'local'
            self._dimensions_shared = None
        try:
            super().to_hdf5_group(parent_group=parent_group,
                                  force_datasets=force_datasets,
                                  name=name,
                                  exclude_keys=['_dimensions_shared', '_parent'])
        finally:
            self._dimensions_shared = shared_backup
            self._local_dimensions = local_backup
            self.dimensions_domain = domain_backup
    def to_sea(self, file_path: str,
                force_datasets: List = ['data']) -> None:
        """Save dimension to HDF5."""
        super().to_sea(file_path=file_path,
                       force_datasets=force_datasets)

    def _get_tree_html(self, recursive_level: List[str] = 0, 
                       exclude_keys: List[str]= ['data', 'original_metadata'], 
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = []
                       ) -> str:
        return super().get_tree_html(recursive_level, 
                                     exclude_keys=exclude_keys, 
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )
    def get_tree_html(self, recursive_level: int = 0, 
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties:bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
                      return self._get_tree_html(recursive_level, 
                                   exclude_keys=exclude_keys+['data', 'original_metadata'],
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys
                                   )

    def unfold_axes(self, keep_axes: Sequence[int | str] = []) -> dict:
        """
        Flatten all axes except those listed in keep_axes.
        Example: shape (l,m,n,o), keep_axes=(1,2) -> (l*o, m, n);
        keep_axes=(2,) -> (l*m*o, n).
        Returns a state dict for fold_axes.
        """
        # Check that the signal is not already folded
        if self._fold_state is not None:
            raise UserWarning("_fold_state is already populated indicating the data is folded.")
        
        # Check if axes are provided to keep then sort the kept and other axes as int lists
        if len(keep_axes) == 0 and len(self.dimensions.sig_dimensions)>0:
            keep_axes = self.dimensions.sig_dimensions
        keep_axes = [v if isinstance(v, int) 
                     else self.dimensions.get_index_from_name(v) 
                     for v in keep_axes ]
        ndim = self.dimensions.ndim
        keep_axes = [v if v>0 else v+ndim for v in keep_axes]
        other_axes = [i for i in range(ndim) if i not in keep_axes]
        permute = other_axes + keep_axes

        # Form the new shape
        transposed = np.transpose(self.data.copy(), axes=permute)
        other_shape = transposed.shape[: len(other_axes)]
        kept_shape = transposed.shape[len(other_axes):]
        flat = transposed.reshape((int(np.prod(other_shape)), *kept_shape))

        state = {
            "other_axes": other_axes,
            "keep_axes": keep_axes,
            "permute": permute,
            "original_shape": self.data.shape,
            "other_shape": other_shape,
            "kept_shape": kept_shape,
        }
        # Set the state
        self._fold_state = state
        self.data = flat
        return state
    def fold_axes(self):
        """
        Restore data shape from a state produced by unfold_axes.
        """
        state = getattr(self, "_fold_state", None)
        if state is None:
            raise ValueError("No fold state available; call unfold_axes first or supply state.")

        data = np.asarray(self.data).reshape((*state["other_shape"], *state["kept_shape"]))
        inv_perm = np.argsort(state["permute"])
        restored = np.transpose(data, axes=inv_perm)
        self.data = restored
        self._fold_state = None

    def infer_plot_dims(self,dims,fnc):
        """Infer and validate plot dimensions from the given input.

        This method determines which dimensions of the data should be plotted based on
        the input specification. It handles cases where dimensions are not explicitly
        provided by inferring them from the data structure's signal or navigation
        dimensions.

        Parameters
        ----------
        dims : {None, 'sig', 'nav', int, tuple of int}, optional
            Specification of dimensions to plot:
            - None: Automatically infer dimensions from data structure
            - 'sig': Use signal dimensions if <= 2D
            - 'nav': Use navigation dimensions if <= 2D
            - int: Single dimension index
            - tuple of int: Explicit dimension indices
            Default is None.
        fnc : callable
            Function to apply for reducing remaining dimensions. Typically used
            for averaging or slicing over non-plotted dimensions.
        Returns
        -------
        data : BaseStructureNumpy
            The data object, possibly reduced by fnc if dimensions remain after
            selecting plot dimensions.
        dims : tuple of int
            The validated dimension indices to be plotted, converted to integer
            indices relative to the data structure.
        Raises
        ------
        ValueError
            If the inferred or specified dimensions exceed 2D, which is not yet
            implemented for plotting.
        ValueError
            If dims cannot be inferred because total, signal, and navigation
            dimensions are all larger than 2D.
        Notes
        -----
        The method automatically reduces the dataset along non-plotted dimensions
        using the provided function if remaining dimensions exist.
        """
        if dims is None:
            if self.dimensions.ndim   == 2: dims = [0,1] # The signal is already plottable
            elif self.dimensions.ndim == 1: dims = [0]   # The signal is already plottable
            elif 0 < len(self.dimensions.sig_dimensions) <= 2: dims = self.dimensions.sig_dimensions # Plot the signal
            elif 0 < len(self.dimensions.nav_dimensions) <= 2: dims = self.dimensions.nav_dimensions # Plot the navigation
            else: raise ValueError('The total, signal, and navigation dimensions are larger than 2 dimensions so a dimension could not be infered. Ploting >2D is not yet implemented')
        elif dims == 'sig':
            if len(self.dimensions.sig_dimensions) <=2: dims = self.dimensions.sig_dimensions
            else: raise ValueError('The signal dimensions being plotted is larger than 2 dimensions, which is not yet implemented')
        elif dims == 'nav':
            if len(self.dimensions.nav_dimensions) <=2: dims = self.dimensions.nav_dimensions
            else: raise ValueError('The navigation dimensions being plotted is larger than 2 dimensions, which is not yet implemented')
        elif isinstance(dims, int):
            dims = tuple([dims])
        #TODO use self.dimensions.get_dims_as_int()
        if len(dims)>2: raise ValueError('The signal dimensions being plotted is larger than 2 dimensions, which is not yet implemented')
        else: dims = self.dimensions.get_dims_as_int(dims)
        
        dims_remain = tuple(i for i in range(len(self.dimensions)) if i not in dims)
        
        if len(dims_remain) == 0:
            return self,dims
        else:
            return fnc(self, axis=dims_remain),dims

    def show(self, 
             ax: mplAxes|None = None,
             dims: None|Literal['sig','nav']|Iterable[int, str]= None,
             fnc: Callable|None = np.sum,
             filename: str|None = None,
             **kwargs
             ) -> PlotImage | None:
        """A shortcut plotting function that automatically handles things like dimensionality and matplotlib kwargs.

        Parameters
        ----------
        ax : mplAxes | None, optional
            Axis to plot to. If None then an axis is both created and returned, by default None
        dims : None | Literal['sig','nav'] | Iterable[int, str], optional
            Specification of dimensions to plot
            - None: Automatically infer dimensions from data structure
            - 'sig': Use signal dimensions if <= 2D
            - 'nav': Use navigation dimensions if <= 2D
            - int: Single dimension index
            - tuple of int: Explicit dimension indices
            Default is None.
        fnc : Callable | None, optional
            Function used to collapse the not `dims` dimensions, by default np.sum
        filename : str | None, optional
            File to save the plot to, by default None
        **kwarg:
            kwargs supplied to imshow or plot.

        Returns
        -------
        plot: PlotImage | None
            Generated object that was plotted in the axis.

        Raises
        ------
        NotImplementedError
            Plotting is implemented for multivariate dimensions but not yet for multivariate signals.
        """
        
        if self.data.ndim != self.dimensions.ndim: raise NotImplementedError('Plotting is implemented for multivariate dimensions but not yet for multivariate signals.')

        sig,dims = self.infer_plot_dims(dims,fnc)

        if 'xlabel' not in kwargs:
            kwargs['xlabel'] = f'{sig.dimensions[-1].name} ({sig.dimensions[-1].units})'
        if len(dims)==2:
            if 'extent' not in kwargs: kwargs['extent'] = sig.dimensions.get_extents(kind='Image')
            if 'ylabel' not in kwargs: kwargs['ylabel'] = f'{sig.dimensions[-2].name} ({sig.dimensions[-2].units})'
            if 'scale_bar_kwargs' not in kwargs: kwargs['scale_bar_kwargs'] = {'units':sig.dimensions[-1].units}
            else:
                if 'units' not in kwargs['scale_bar_kwargs']:
                    kwargs['scale_bar_kwargs']['units'] =sig.dimensions[-1].units
            #p = PlotImage(sig.data, ax=ax, **kwargs)
        if len(dims)==1:
            if 'ylabel' not in kwargs: kwargs['xlabel'] = f'{sig.dimensions[0].name} ({sig.dimensions[0].units})'
            if 'x' not in kwargs: kwargs['x'] = sig.dimensions[0].values
            #p = ax.plot(sig.dimensions[0], sig.data, **kwargs)

        if ax is None: fig, ax = plt.subplots()
        p = plot_nd_array(sig.data, ax=ax, **kwargs)
        if filename is not None: save_fig(filename)
        return p
    
    #BUG circular import. Need to fix before it can be implemented
    # def show_interactive(self,
    #                      nav_dimensions: Tuple[int, ...] | dims_set_Lit = 'navigation',
    #                      sig_dimensions: Tuple[int, ...] | dims_set_Lit = 'signal',
    #                      nav_fnc: Callable = np.sum,
    #                      show_dimension_selector: bool = True):
    #     """Create a complete interactive signal plot with optional dimension selector.
        
    #     Parameters:
    #     -----------
    #     nav_dimensions : tuple | Literal['signal', 'navigation', 'temporal', 'spectral', 'position', 'scattering']
    #         Initial navigation dimensions
    #     sig_dimensions : tuple | Literal['signal', 'navigation', 'temporal', 'spectral', 'position', 'scattering']
    #         Initial signal dimensions
    #     nav_fnc : Callable
    #         Function to apply along function_dimensions (default: sum)
    #     show_dimension_selector : bool
    #         Whether to show the dimension selection widget (default: True)
    #     """
    #     if nav_dimensions == 'navigation' and len(self.dimensions.nav_dimensions)==0:
    #         nav_dimensions = None
    #     if sig_dimensions == 'signal' and len(self.dimensions.sig_dimensions)==0:
    #         nav_dimensions = None
    #     interactive_signal_plot(self, 
    #                             nav_dimensions=nav_dimensions, sig_dimensions=sig_dimensions, 
    #                             nav_fnc=nav_fnc,
    #                             show_dimension_selector=show_dimension_selector
    #                             )
            

    def image(self,
             dims: None|Literal['sig','nav']|Iterable[int, str]= None,
             fnc: Callable|None = np.sum,
             filename: str|None = None ) -> None:

        sig,dims = self.infer_plot_dims(dims,fnc)
        size = np.asarray( [ d.get_extent() for d in sig.dimensions ] )
        size = size[:,1]-size[:,0]
        units = [ d.units for d in sig.dimensions ]
        save_image(sig.data,size,units[0],filename=filename)

class SignalCollection(SEASerializable):
    """Collection of Signals and/or SignalSets with shared metadata.

    Methods
    -------
    add_dataset(dataset, merge_metadata=False, merge_dimensions=False, meta_kwargs=None)
        Append a dataset, optionally merging metadata/dimensions.
    remove_dataset(key)
        Remove a dataset by index or name.
    to_sea(file_path)
        Persist the collection to a SEA file.
    load(file_path)
        Classmethod to build a collection from disk.
    """

    allowed_types: tuple[type, ...] = (Signal,)

    def __init__(self,
                 datasets: Sequence[Signal | SignalSet] | None = None,
                 metadata: Metadata | Dict | None = None,
                 name: str | None = None,
                 uuid: str | None = None) -> None:
        """Initialize a collection of signals or signal sets.

        Parameters
        ----------
        datasets : Sequence[Signal | SignalSet] | None, optional
            Items to include initially, by default None.
        metadata : Metadata | dict | None, optional
            Shared metadata; dicts are converted to ``Metadata``.
        name : str | None, optional
            Optional collection name.
        uuid : str | None, optional
            UUID for the collection; generated when omitted.
        """
        self.uuid = uuid if uuid is not None else generate_uuid()
        self.name = name
        if isinstance(metadata, Dict): metadata = Metadata(metadata)
        self.metadata = metadata
        self.datasets: list[Signal | SignalSet] = []

        if datasets is not None:
            for ds in datasets:
                self.add_dataset(ds)

    def __len__(self) -> int:
        """
        Return the number of datasets in the collection.

        Returns
        -------
        int
            Number of contained datasets.
        """
        return len(self.datasets)

    def __str__(self):
        """
        Return a human-readable representation.

        Returns
        -------
        str
            Name and class label.
        """
        if self.name is None: name = 'Unnamed'
        else: name = self.name
        return f'{name} (SignalCollection)'

    def __getitem__(self, key: int | str):
        """
        Retrieve a dataset by index or name.

        Parameters
        ----------
        key : int | str
            Index or dataset name.

        Returns
        -------
        Signal | SignalSet
            Retrieved dataset.
        """
        key = self._resolve_index(key)
        return self.datasets[key]

    def deepcopy(self):
        """
        Deep copy the collection and refresh its UUID.

        Returns
        -------
        SignalCollection
            Deep-copied collection.
        """
        out = super().deepcopy()
        setattr(out, 'uuid', generate_uuid())
        return out

    def _get_tree_html(self, recursive_level: int = 0,
                       exclude_keys: List[str] = [],
                       exclude_hidden: bool = True,
                       exclude_properties: bool = False,
                       promote_itterable_keys: List[str] = ['datasets']
                       ) -> str:
        """
        Internal HTML tree helper.

        Parameters
        ----------
        recursive_level : int, optional
            Depth to expand, by default 0.
        exclude_keys : list[str], optional
            Attributes to omit, by default [].
        exclude_hidden : bool, optional
            Exclude hidden attributes, by default True.
        exclude_properties : bool, optional
            Exclude properties, by default False.
        promote_itterable_keys : list[str], optional
            Iterable attributes to promote, by default ['datasets'].

        Returns
        -------
        str
            HTML representation.
        """
        return super().get_tree_html(recursive_level,
                                     exclude_keys=exclude_keys,
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys)

    def get_tree_html(self, recursive_level: int = 0,
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties: bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
        """
        HTML tree representation that promotes datasets.

        Parameters
        ----------
        recursive_level : int, optional
            Depth to expand, by default 0.
        exclude_keys : list[str], optional
            Attributes to omit, by default [].
        exclude_hidden : bool, optional
            Exclude hidden attributes, by default True.
        exclude_properties : bool, optional
            Exclude properties, by default False.
        promote_itterable_keys : list[str], optional
            Iterable attributes to promote, by default [].

        Returns
        -------
        str
            HTML representation.
        """
        return self._get_tree_html(recursive_level,
                                   exclude_keys=exclude_keys,
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys + ['datasets'])

    def get_dataset_names(self) -> List[str]:
        """
        Return dataset names.

        Returns
        -------
        list[str]
            Names of contained datasets.
        """
        return [getattr(ds, 'name', None) for ds in self.datasets]

    def get_index_from_name(self, name: str) -> int:
        """
        Resolve a dataset name to its index.

        Parameters
        ----------
        name : str
            Dataset name.

        Returns
        -------
        int
            Index of the dataset.

        Raises
        ------
        KeyError
            If the name is not found.
        """
        names = self.get_dataset_names()
        if name in names: return names.index(name)
        else: raise KeyError(f'A key of {name} was provided and the signal names are {names}')

    def _resolve_index(self, key: int | str) -> int:
        """
        Resolve a key to an index.

        Parameters
        ----------
        key : int | str
            Index or dataset name.

        Returns
        -------
        int
            Dataset index.
        """
        if isinstance(key, int): return key
        if isinstance(key, str): return self.get_index_from_name(key)
        raise TypeError(f'Only integers and strings are allowed but a {type(key)} was provided.')

    def add_dataset(self, dataset: Signal | SignalSet,
                    merge_metadata: bool = False,
                    name: str | None = None,
                    meta_kwargs: dict[str, Any] | None = None) -> None:
        """
        Add a dataset to the collection.

        Parameters
        ----------
        dataset : Signal | SignalSet
            Dataset to add.
        merge_metadata : bool, optional
            Merge dataset metadata into the collection metadata, by default False.
        meta_kwargs : dict[str, Any] | None, optional
            Keyword arguments forwarded to ``Metadata.merge``, by default
            ``dict(kind='append', warn_duplicate=False, inform_new=False)``.
        """
        if meta_kwargs is None:
            meta_kwargs = dict(kind='append', warn_duplicate=False, inform_new=False)
        if not isinstance(dataset, self.allowed_types):
            raise TypeError(f'Datasets must be instances of {self.allowed_types}, received {type(dataset)}.')

        dataset_copy = dataset.deepcopy()
        if name is not None:
            dataset_copy.name = name
        else:
            if dataset_copy.name in self.get_dataset_names():
                new_name = f'{dataset_copy.name} - {dataset_copy.uuid}'
                warn(f'A dataset with the name {dataset_copy.name} already exists in this collection. The name will be changed to {new_name}.')
                dataset_copy.name = new_name
        if hasattr(dataset_copy, '_parent'):
            dataset_copy._parent = self
        incoming_meta = getattr(dataset_copy, 'metadata', None)
        if merge_metadata and incoming_meta is not None:
            if self.metadata is None:
                self.metadata = incoming_meta
            else:
                self.metadata.merge(incoming_meta, **meta_kwargs)
        self.datasets.append(dataset_copy)

    def remove_dataset(self, key: int | str) -> Signal | SignalSet:
        """
        Remove and return a dataset by index or name.

        Parameters
        ----------
        key : int | str
            Dataset index or name.

        Returns
        -------
        Signal | SignalSet
            Removed dataset.
        """
        idx = self._resolve_index(key)
        return self.datasets.pop(idx)

    @classmethod
    def load(cls, file_path: str) -> "SignalCollection":
        """
        Create a SignalCollection from a SEA file.

        Parameters
        ----------
        file_path : str
            Path to the SEA file.

        Returns
        -------
        SignalCollection
            Loaded collection.
        """
        instance = cls()
        instance.from_sea(file_path)
        return instance

class SignalSet(SignalCollection):
    """Specialized collection that stores Signals with shared Dimensions.

    Methods
    -------
    add_dataset(dataset, merge_metadata=False, meta_kwargs=None)
        Append a signal while binding shared dimensions and optionally merging metadata.
    decompose_metadata(standalone=True)
        Split common vs detector-specific metadata and optionally attach per-signal copies.
    from_acquisition(signals, name=None, metadata=None, dimensions=None, enforce_aquisition_uuid=False, merge_metadata=True)
        Classmethod factory for acquisition-style sets with optional UUID enforcement.
    """

    allowed_types: tuple[type, ...] = (Signal,)

    def __init__(self,
                 signals: Sequence[Signal] | None = None,
                 main_signal: int | None = None,
                 name: str | None = None,
                 metadata: Metadata | Dict | None = None,
                 dimensions: Dimensions | None = None,
                 uuid: str | None = None,
                 merge_metadata: bool = False) -> None:
        """Create a set of signals that share dimension definitions.

        Parameters
        ----------
        signals : Sequence[Signal] | None, optional
            Signals to include, by default None.
        main_signal : int | None, optional
            Index of the primary signal; defaults to first element.
        name : str | None, optional
            Collection name.
        metadata : Metadata | dict | None, optional
            Shared metadata; dicts are converted to ``Metadata``.
        dimensions : Dimensions | None, optional
            Shared dimensions object; created empty when ``None``.
        uuid : str | None, optional
            UUID for the set; generated when omitted.
        merge_metadata : bool, optional
            Whether to merge child metadata into the set, by default False.
        """
        if main_signal is not None:
            main_signal = main_signal if main_signal>=0 else len(signals)+main_signal
        self.main_signal = main_signal
        super().__init__(datasets=None, metadata=metadata, name=name, uuid=uuid)

        # Ensure shared dimensions are a Dimensions instance; coerce if needed
        if dimensions is None:
            self.dimensions = Dimensions()
        elif isinstance(dimensions, Dimensions):
            self.dimensions = dimensions
        else:
            self.dimensions = Dimensions(dimensions)

        if signals is not None:
            signals_list = list(signals)
            primary_idx = 0 if main_signal is None else int(main_signal)
            if 0 <= primary_idx < len(signals_list):
                signals_list.insert(0, signals_list.pop(primary_idx))
            self.main_signal = 0 if len(signals_list) > 0 else None
            for signal in signals_list:
                self.add_dataset(signal, merge_metadata=merge_metadata)
        
    def __str__(self):
        """
        Return a human-readable representation.

        Returns
        -------
        str
            Name and class label.
        """
        if self.name is None: name = 'Unnamed'
        else: name = self.name
        return f'{name} (SignalSet)'

    def add_dataset(self, dataset: Signal,
                    merge_metadata: bool = False,
                    meta_kwargs: dict[str, Any] | None = None) -> None:
        """
        Add a Signal to the set, binding shared dimensions.

        Parameters
        ----------
        dataset : Signal
            Signal to add.
        merge_metadata : bool, optional
            Merge signal metadata into set metadata, by default False.
        meta_kwargs : dict[str, Any] | None, optional
            Keyword arguments forwarded to ``Metadata.merge``.
        """
        if meta_kwargs is None:
            meta_kwargs = dict(kind='append', warn_duplicate=False, inform_new=False)
        if not isinstance(dataset, self.allowed_types):
            raise TypeError(f'Signals must be instances of {self.allowed_types}, received {type(dataset)}.')

        signal_copy = dataset.deepcopy()
        signal_copy._parent = self
        
        # Ensure the SignalSet aggregates dimension definitions from each signal.
        # For robustness, always add any missing axes and map navigation/signal
        # role indices from the incoming signal (by name) if they are present.
        for ax in signal_copy.dimensions.dimensions:
            if ax.name not in self.dimensions.get_names():
                self.dimensions.add_dimension(ax.deepcopy())

        # Map nav/sig indices from the signal into the set's coordinate space
        # using axis names. Only set the set-level nav/sig lists if they are
        # currently unset (empty) so the first signal with role info defines
        # the roles for the set.
        if len(self.datasets) == 0:
            sig_names = [d.name for d in signal_copy.dimensions.dimensions]
            set_names = self.dimensions.get_names()
            if len(self.dimensions.nav_dimensions) == 0 and len(signal_copy.dimensions.nav_dimensions) != 0:
                self.dimensions.nav_dimensions = [set_names.index(sig_names[i]) for i in signal_copy.dimensions.nav_dimensions]
            if len(self.dimensions.sig_dimensions) == 0 and len(signal_copy.dimensions.sig_dimensions) != 0:
                self.dimensions.sig_dimensions = [set_names.index(sig_names[i]) for i in signal_copy.dimensions.sig_dimensions]

        signal_copy.bind_shared_dimensions(self.dimensions)

        incoming_meta = getattr(signal_copy, 'metadata', None)
        if merge_metadata and incoming_meta is not None:
            if self.metadata is None:
                self.metadata = incoming_meta
            else:
                self.metadata.merge(incoming_meta, **meta_kwargs)
        self.datasets.append(signal_copy)

    def decompose_metadata(self, standalone: bool = True) -> tuple[Metadata | None, dict[str, Metadata]]:
        """
        Decompose set metadata into common and detector-specific parts.

        Parameters
        ----------
        standalone : bool, optional
            When True, assign each signal a standalone Metadata object combining
            common metadata with its detector-specific metadata. When False,
            signals have their metadata cleared but detector-specific metadata is
            still returned. Defaults to True.

        Returns
        -------
        tuple[Metadata | None, dict[str, Metadata]]
            Common metadata and a mapping of detector name to Metadata.
        """
        if self.metadata is None:
            return None, {}

        common = self.metadata.deepcopy()
        detector_map: dict[str, Metadata] = {}

        detectors_source = getattr(getattr(self.metadata, "Instrument", None), "Detectors", None)
        detectors_dict: dict[str, Any] = {}
        if detectors_source is not None and hasattr(detectors_source, "to_dict"):
            detectors_dict = detectors_source.to_dict(hidden=True, properties=False, deep=True)

        instrument_common = getattr(common, "Instrument", None)
        if instrument_common is not None and hasattr(instrument_common, "Detectors"):
            instrument_common.Detectors = Metadata({})

        for signal in self.datasets:
            detector_key = getattr(signal, "detector", None)
            if detector_key is None:
                continue
            detector_payload = detectors_dict.get(detector_key)
            if detector_payload is None:
                continue

            detector_meta = detector_payload if isinstance(detector_payload, Metadata) else Metadata(detector_payload if isinstance(detector_payload, dict) else {'Detector': detector_payload})
            detector_map[detector_key] = detector_meta

            if standalone:
                signal_meta = common.deepcopy() if common is not None else Metadata({})
                if getattr(signal_meta, "Instrument", None) is None:
                    signal_meta.Instrument = Metadata({})
                if getattr(signal_meta.Instrument, "Detectors", None) is None:
                    signal_meta.Instrument.Detectors = Metadata({})
                setattr(signal_meta.Instrument.Detectors, detector_key, detector_meta.deepcopy() if isinstance(detector_meta, SEASerializable) else detector_meta)
                signal.metadata = signal_meta
            else:
                signal.metadata = None

        return common, detector_map

    @classmethod
    def from_acquisition(cls,
                         signals: Sequence[Signal] | None = None,
                         name: str | None = None,
                         metadata: Metadata | Dict | None = None,
                         dimensions: Dimensions | None = None,
                         enforce_aquisition_uuid: bool = False,
                         merge_metadata: bool = True) -> "SignalSet":
        """
        Factory for acquisition-style SignalSets.

        Parameters
        ----------
        signals : Sequence[Signal] | None, optional
            Signals to include.
        name : str | None, optional
            Optional name.
        metadata : Metadata | Dict | None, optional
            Set-level metadata.
        dimensions : Dimensions | None, optional
            Shared dimensions.
        enforce_aquisition_uuid : bool, optional
            If True, ensure all signals share the same acquisition UUID and use
            it for the set UUID. Defaults to False.
        merge_metadata : bool, optional
            Merge signal metadata into the set metadata on add. Defaults to True.

        Returns
        -------
        SignalSet
            Constructed acquisition-style set.
        """
        instance = cls(signals=None, name=name, metadata=metadata,
                       dimensions=dimensions, merge_metadata=merge_metadata)

        if signals is not None:
            uuids = []
            for sig in signals:
                scan_uuid = None
                try:
                    scan_uuid = sig.metadata.Instrument.Scan.scan_uuid  # type: ignore[attr-defined]
                except Exception:
                    scan_uuid = None
                uuids.append(scan_uuid)
                instance.add_dataset(sig, merge_metadata=merge_metadata)

            if enforce_aquisition_uuid and len(set([u for u in uuids if u is not None])) > 1:
                raise ValueError('Signals do not share a common acquisition UUID.')
            if enforce_aquisition_uuid and len(uuids) > 0 and all(u is not None for u in uuids):
                instance.uuid = uuids[0]
        return instance

# After specialization is defined, update the general collection allowed types.
SignalCollection.allowed_types = (Signal, SignalSet)

class SEAFile(SEASerializable):
    """Top-level SEA container that mirrors the contents of a ``.sea`` file.

    Bundles simulations, experiments, and analysis collections alongside global
    metadata; provides helpers to merge detector metadata into signals.

    Methods
    -------
    add_simulation(dataset, merge_metadata=False, meta_kwargs=None)
        Add a dataset to the ``Simulations`` collection.
    add_experiment(dataset, merge_metadata=False, meta_kwargs=None)
        Add a dataset to the ``Experiments`` collection.
    add_analysis(dataset, merge_metadata=False, meta_kwargs=None)
        Add a dataset to the ``Analysis`` collection.
    add(obj, location=None, merge_metadata=False, meta_kwargs=None)
        Generic adder for signals, sets, collections, or metadata.
    remove_simulation(key), remove_experiment(key), remove_analysis(key)
        Remove datasets by name or index from the respective collections.
    deepcopy()
        Deep copy the SEAFile with a fresh UUID.
    get_tree_html(...)
        Render an HTML tree view promoting SEA collections.
    """

    _SEA_COLLECTION_NAMES = ('Simulations', 'Experiments', 'Analysis')

    def __init__(self,
                 metadata: Metadata | Dict | None = None,
                 simulations: SignalCollection | Sequence[Signal | SignalSet] | None = None,
                 experiments: SignalCollection | Sequence[Signal | SignalSet] | None = None,
                 analysis: SignalCollection | Sequence[Signal | SignalSet] | None = None,
                 name: str | None = None,
                 uuid: str | None = None) -> None:
        """
        Initialize a SEAFile with optional metadata and SEA collections.

        Parameters
        ----------
        metadata : Metadata | Dict | None, optional
            Metadata to attach to the SEA file. A provided ``dict`` is coerced
            into :class:`Metadata`. Defaults to ``None``.
        simulations : SignalCollection | Sequence[Signal | SignalSet] | None, optional
            Data for the ``Simulations`` collection. A :class:`SignalCollection`
            is deep-copied; an iterable of ``Signal``/``SignalSet`` items is
            forwarded to the ``datasets`` argument of a new collection. If
            ``None``, the collection remains uninitialized until first use.
        experiments : SignalCollection | Sequence[Signal | SignalSet] | None, optional
            Data for the ``Experiments`` collection. Accepts the same forms as
            ``simulations``. ``None`` leaves the collection uninitialized.
        analysis : SignalCollection | Sequence[Signal | SignalSet] | None, optional
            Data for the ``Analysis`` collection. Accepts the same forms as
            ``simulations``. ``None`` leaves the collection uninitialized.
        name : str | None, optional
            Optional name for the SEA file.
        uuid : str | None, optional
            UUID override. If omitted, a new UUID is generated.

        Returns
        -------
        None
        """
        self.uuid = uuid if uuid is not None else generate_uuid()
        self.name = name
        if isinstance(metadata, Dict): metadata = Metadata(metadata)
        self.metadata = metadata
        self.Simulations = self._initialize_collection(simulations, 'Simulations')
        self.Experiments = self._initialize_collection(experiments, 'Experiments')
        self.Analysis = self._initialize_collection(analysis, 'Analysis')

    def __len__(self) -> int:
        """
        Count the datasets across all initialized SEA collections.

        Returns
        -------
        int
            Total number of datasets contained within initialized collections.
        """
        return sum(len(coll) for coll in self._collections().values())

    def deepcopy(self):
        """
        Create a deep copy of the SEAFile with a fresh UUID.

        Returns
        -------
        SEAFile
            A deep-copied SEAFile instance.
        """
        out = super().deepcopy()
        setattr(out, 'uuid', generate_uuid())
        return out

    def _get_tree_html(self, recursive_level: int = 0,
                       exclude_keys: List[str] = [],
                       exclude_hidden: bool = True,
                       exclude_properties: bool = False,
                       promote_itterable_keys: List[str] = ['Simulations', 'Experiments', 'Analysis']
                       ) -> str:
        """
        Internal HTML tree helper with default SEA collection promotion.

        Parameters
        ----------
        recursive_level : int, optional
            Depth to expand. ``0`` expands root only. Defaults to ``0``.
        exclude_keys : list of str, optional
            Attributes to omit. Defaults to ``[]``.
        exclude_hidden : bool, optional
            Whether to skip hidden attributes. Defaults to ``True``.
        exclude_properties : bool, optional
            Whether to skip properties. Defaults to ``False``.
        promote_itterable_keys : list of str, optional
            Iterable attribute names to render as tree branches. Defaults to
            the SEA collection names.

        Returns
        -------
        str
            HTML string representation of the tree.
        """
        return super().get_tree_html(recursive_level,
                                     exclude_keys=exclude_keys,
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys)

    def get_tree_html(self, recursive_level: int = 0,
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties: bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
        """
        HTML tree representation that promotes SEA collections.

        Parameters
        ----------
        recursive_level : int, optional
            Depth to expand. ``0`` expands root only. Defaults to ``0``.
        exclude_keys : list of str, optional
            Attributes to omit. Defaults to ``[]``.
        exclude_hidden : bool, optional
            Whether to skip hidden attributes. Defaults to ``True``.
        exclude_properties : bool, optional
            Whether to skip properties. Defaults to ``False``.
        promote_itterable_keys : list of str, optional
            Iterable attribute names to render as tree branches. Defaults to
            ``[]``; SEA collection names are always added.

        Returns
        -------
        str
            HTML string representation of the tree.
        """
        return self._get_tree_html(recursive_level,
                                   exclude_keys=exclude_keys,
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys + ['Simulations', 'Experiments', 'Analysis'])

    def _merge_metadata(self, incoming: Metadata | None,
                        merge_metadata: bool,
                        meta_kwargs: dict[str, Any]) -> None:
        """
        Merge incoming metadata into the SEAFile metadata when requested.

        Parameters
        ----------
        incoming : Metadata | None
            Metadata to merge.
        merge_metadata : bool
            Whether merging should occur.
        meta_kwargs : dict[str, Any]
            Keyword arguments forwarded to :meth:`Metadata.merge`.

        Returns
        -------
        None
        """
        if not merge_metadata or incoming is None:
            return
        if self.metadata is None:
            self.metadata = incoming
        else:
            self.metadata.merge(incoming, **meta_kwargs)

    def _initialize_collection(self,
                               collection: SignalCollection | Sequence[Signal | SignalSet] | None,
                               name: str) -> SignalCollection | None:
        """
        Cast an incoming collection argument.

        Parameters
        ----------
        collection : SignalCollection | Sequence[Signal | SignalSet] | None
            Provided collection instance or iterable of datasets.
        name : str
            Target SEA collection name.

        Returns
        -------
        SignalCollection | None
            A deep-copied or newly created collection, or ``None`` if no
            collection was provided.
        """
        if collection is None:
            return None
        if isinstance(collection, SignalCollection):
            initialized = collection.deepcopy()
            initialized.name = name
            return initialized
        if isinstance(collection, Sequence):
            datasets = list(collection)
            if not all(isinstance(ds, (Signal, SignalSet)) for ds in datasets):
                raise TypeError(f'Only Signal or SignalSet instances can seed {name}, received {[type(ds) for ds in datasets]}.')
            return SignalCollection(datasets=datasets, name=name)
        raise TypeError(f'Only SignalCollection instances or iterables of Signal/SignalSet can be assigned to {name}, received {type(collection)}.')

    def _collections(self) -> dict[str, SignalCollection]:
        """
        Return initialized SEA collections.

        Returns
        -------
        dict[str, SignalCollection]
            Mapping of collection name to initialized :class:`SignalCollection`.
        """
        return {label: getattr(self, label) for label in self._SEA_COLLECTION_NAMES
                if isinstance(getattr(self, label, None), SignalCollection)}

    def _normalize_location(self, location: str | None) -> str | None:
        """Normalize a location string to a canonical SEA collection name.
        This function helps with case-inensitivity, attribute existance, and checking that the atribute is a SignalCollection.
        Parameters
        ----------
        location : str | None
            Requested location (case-insensitive) or ``None``.

        Raises
        ------
        TypeError
            If ``location`` is not a string when provided.
        ValueError
            If the location does not match a SEA collection or ``metadata``.

        Returns
        -------
        str | None
            Canonical collection name or ``None`` if no location is provided.
        """
        if location is None:
            return None
        if not isinstance(location, str):
            raise TypeError('location must be a string when provided.')
        mapping = {name.lower(): name for name in self._SEA_COLLECTION_NAMES}
        mapping['metadata'] = 'metadata'
        key = location.lower()
        if key in mapping:
            return mapping[key]
        raise ValueError(f"location must be one of {list(mapping.values())}.")

    def _get_collection(self, location: str, create_if_missing: bool = False) -> SignalCollection:
        """
        Retrieve a SEA collection by location, optionally creating it if missing.

        Parameters
        ----------
        location : str
            Target collection name or alias.
        create_if_missing : bool, optional
            Create and assign an empty :class:`SignalCollection` when the target
            is currently ``None``. Defaults to ``False``.

        Returns
        -------
        SignalCollection
            The requested (or newly created) collection.

        Raises
        ------
        ValueError
            If ``location`` refers to metadata or is omitted.
        AttributeError
            If the collection is not initialized and ``create_if_missing`` is
            ``False``.
        TypeError
            If the attribute exists but is not a :class:`SignalCollection`.
        """
        normalized = self._normalize_location(location)
        if normalized in (None, 'metadata'):
            raise ValueError('A SEA collection name (Simulations, Experiments, Analysis) is required.')
        collection = getattr(self, normalized, None)
        if collection is None:
            if not create_if_missing:
                raise AttributeError(f'The {normalized} collection is not initialized on this SEAFile.')
            collection = SignalCollection(name=normalized)
            setattr(self, normalized, collection)
        if not isinstance(collection, SignalCollection):
            raise TypeError(f'The {normalized} attribute is not a SignalCollection (found {type(collection)}).')
        return collection

    def _assign_collection(self, location: str, collection: SignalCollection) -> None:
        """
        Assign a SignalCollection to a SEA collection slot.

        Parameters
        ----------
        location : str
            Target collection name or alias.
        collection : SignalCollection
            Collection to assign; deep-copied and renamed.

        Returns
        -------
        None
        """
        normalized = self._normalize_location(location)
        if normalized in (None, 'metadata'):
            raise ValueError('SignalCollections must target one of Simulations, Experiments, or Analysis.')
        collection_copy = collection.deepcopy()
        collection_copy.name = normalized
        setattr(self, normalized, collection_copy)

    def _add_to_collection(self,
                           location: str,
                           dataset: Signal | SignalSet,
                           merge_metadata: bool = False,
                           meta_kwargs: dict[str, Any] | None = None) -> None:
        """
        Add a dataset to the specified SEA collection.

        Parameters
        ----------
        location : str
            Target SEA collection name or alias.
        dataset : Signal | SignalSet
            Dataset to add.
        merge_metadata : bool, optional
            Whether to merge dataset metadata into SEAFile metadata. Defaults to
            ``False``.
        meta_kwargs : dict[str, Any] | None, optional
            Arguments forwarded to :meth:`Metadata.merge`. Defaults to a safe
            append behaviour when ``None``.

        Returns
        -------
        None
        """
        if meta_kwargs is None:
            meta_kwargs = dict(kind='append', warn_duplicate=False, inform_new=False)
        collection = self._get_collection(location, create_if_missing=True)
        if merge_metadata:
            incoming_meta = getattr(dataset, 'metadata', None)
            self._merge_metadata(incoming_meta, merge_metadata, meta_kwargs)
        collection.add_dataset(dataset,
                               merge_metadata=merge_metadata,
                               meta_kwargs=meta_kwargs)

    def add_simulation(self, dataset: Signal | SignalSet,
                       merge_metadata: bool = False,
                       meta_kwargs: dict[str, Any] | None = None) -> None:
        """
        Add a dataset to the ``Simulations`` collection.

        Parameters
        ----------
        dataset : Signal | SignalSet
            Dataset to add.
        merge_metadata : bool, optional
            Whether to merge dataset metadata into SEAFile metadata. Defaults to
            ``False``.
        meta_kwargs : dict[str, Any] | None, optional
            Arguments forwarded to :meth:`Metadata.merge`. Defaults to a safe
            append behaviour when ``None``.

        Returns
        -------
        None
        """
        self._add_to_collection('Simulations', dataset,
                                merge_metadata=merge_metadata,
                                meta_kwargs=meta_kwargs)

    def add_experiment(self, dataset: Signal | SignalSet,
                       merge_metadata: bool = False,
                       meta_kwargs: dict[str, Any] | None = None) -> None:
        """
        Add a dataset to the ``Experiments`` collection.

        Parameters
        ----------
        dataset : Signal | SignalSet
            Dataset to add.
        merge_metadata : bool, optional
            Whether to merge dataset metadata into SEAFile metadata. Defaults to
            ``False``.
        meta_kwargs : dict[str, Any] | None, optional
            Arguments forwarded to :meth:`Metadata.merge`. Defaults to a safe
            append behaviour when ``None``.

        Returns
        -------
        None
        """
        self._add_to_collection('Experiments', dataset,
                                merge_metadata=merge_metadata,
                                meta_kwargs=meta_kwargs)

    def add_analysis(self, dataset: Signal | SignalSet,
                     merge_metadata: bool = False,
                     meta_kwargs: dict[str, Any] | None = None) -> None:
        """
        Add a dataset to the ``Analysis`` collection.

        Parameters
        ----------
        dataset : Signal | SignalSet
            Dataset to add.
        merge_metadata : bool, optional
            Whether to merge dataset metadata into SEAFile metadata. Defaults to
            ``False``.
        meta_kwargs : dict[str, Any] | None, optional
            Arguments forwarded to :meth:`Metadata.merge`. Defaults to a safe
            append behaviour when ``None``.

        Returns
        -------
        None
        """
        self._add_to_collection('Analysis', dataset,
                                merge_metadata=merge_metadata,
                                meta_kwargs=meta_kwargs)

    def add(self, obj: SEASerializable,
            location: Literal['Simulations', 'Experiments', 'Analysis', 'metadata'] | None = None,
            merge_metadata: bool = False,
            meta_kwargs: dict[str, Any] | None = None) -> None:
        """
        Generic adder that dispatches to a SEA collection or metadata.

        Parameters
        ----------
        obj : SEASerializable
            Object to add (Signal, SignalSet, SignalCollection, or Metadata).
        location : {'Simulations', 'Experiments', 'Analysis', 'metadata'} | None, optional
            Target destination. If ``None``, destination is inferred when
            possible; otherwise required.
        merge_metadata : bool, optional
            Whether to merge dataset metadata into SEAFile metadata (for
            datasets). Defaults to ``False``.
        meta_kwargs : dict[str, Any] | None, optional
            Arguments forwarded to :meth:`Metadata.merge`. Defaults to a safe
            append behaviour when ``None``.

        Raises
        ------
        TypeError
            If the object type is unsupported or metadata destination is
            inconsistent.
        ValueError
            If the destination cannot be inferred.

        Returns
        -------
        None
        """
        normalized_location = self._normalize_location(location)
        if meta_kwargs is None:
            meta_kwargs = dict(kind='append', warn_duplicate=False, inform_new=False)

        if normalized_location == 'metadata' or isinstance(obj, Metadata):
            if not isinstance(obj, Metadata):
                raise TypeError('Only Metadata can be added when location is "metadata".')
            self._merge_metadata(obj, True, meta_kwargs)
            return

        if isinstance(obj, SignalCollection):
            target_location = normalized_location
            if target_location is None:
                target_location = self._infer_location_from_collection(obj)
            self._assign_collection(target_location, obj)
            return

        if isinstance(obj, (Signal, SignalSet)):
            target_location = normalized_location
            if target_location is None:
                target_location = self._infer_single_target_location()
            self._add_to_collection(target_location, obj,
                                    merge_metadata=merge_metadata,
                                    meta_kwargs=meta_kwargs)
            return

        raise TypeError(f'{type(obj)} cannot be added to a SEAFile.')

    def _infer_location_from_collection(self, collection: SignalCollection) -> str:
        """
        Infer a target location from a SignalCollection's name.

        Parameters
        ----------
        collection : SignalCollection
            Collection whose name should match a SEA collection.

        Returns
        -------
        str
            Canonical collection name.

        Raises
        ------
        ValueError
            If the name does not match a SEA collection.
        """
        if collection.name is not None:
            mapping = {name.lower(): name for name in self._SEA_COLLECTION_NAMES}
            key = collection.name.lower()
            if key in mapping:
                return mapping[key]
        raise ValueError("Specify a 'location' for the SignalCollection when adding to a SEAFile.")

    def _infer_single_target_location(self) -> str:
        """
        Infer a destination when only one collection is initialized.

        Returns
        -------
        str
            The sole initialized collection name.

        Raises
        ------
        ValueError
            If zero or multiple collections are initialized.
        """
        available = list(self._collections().keys())
        if len(available) == 1:
            return available[0]
        raise ValueError("Specify 'location' as one of Simulations, Experiments, or Analysis when adding to a SEAFile.")

    def remove_simulation(self, key: int | str) -> Signal | SignalSet:
        """
        Remove and return a dataset from ``Simulations`` by index or name.

        Parameters
        ----------
        key : int | str
            Dataset index or name.

        Returns
        -------
        Signal | SignalSet
            The removed dataset.
        """
        return self._get_collection('Simulations').remove_dataset(key)

    def remove_experiment(self, key: int | str) -> Signal | SignalSet:
        """
        Remove and return a dataset from ``Experiments`` by index or name.

        Parameters
        ----------
        key : int | str
            Dataset index or name.

        Returns
        -------
        Signal | SignalSet
            The removed dataset.
        """
        return self._get_collection('Experiments').remove_dataset(key)

    def remove_analysis(self, key: int | str) -> Signal | SignalSet:
        """
        Remove and return a dataset from ``Analysis`` by index or name.

        Parameters
        ----------
        key : int | str
            Dataset index or name.

        Returns
        -------
        Signal | SignalSet
            The removed dataset.
        """
        return self._get_collection('Analysis').remove_dataset(key)

    def _attempt_remove_from_collection(self, collection: SignalCollection,
                                        obj: SEASerializable | str | int) -> Signal | SignalSet | None:
        """
        Attempt removal of an object from a collection.

        Parameters
        ----------
        collection : SignalCollection
            Collection to mutate.
        obj : SEASerializable | str | int
            Dataset instance, name, or index.

        Returns
        -------
        Signal | SignalSet | None
            Removed dataset or ``None`` if not found.
        """
        if isinstance(obj, (str, int)):
            try:
                return collection.remove_dataset(obj)
            except (KeyError, TypeError):
                return None
        if isinstance(obj, (Signal, SignalSet)) and obj in collection.datasets:
            collection.datasets.remove(obj)
            return obj
        return None

    def _remove_metadata(self, obj: SEASerializable | str) -> Metadata | None:
        """
        Remove metadata when it matches by identity or name.

        Parameters
        ----------
        obj : SEASerializable | str
            Metadata instance or metadata name.

        Returns
        -------
        Metadata | None
            Removed metadata.

        Raises
        ------
        ValueError
            If no metadata exists or the object does not match.
        KeyError
            If a name lookup fails.

        Returns
        -------
        Metadata | None
            Removed metadata.
        """
        if self.metadata is None:
            raise ValueError('This SEAFile has no metadata to remove.')
        if isinstance(obj, str):
            if getattr(self.metadata, 'name', None) == obj:
                removed = self.metadata
                self.metadata = None
                return removed
            raise KeyError(f'No metadata named {obj} was found in the SEAFile.')
        if obj is self.metadata:
            self.metadata = None
            return obj
        raise ValueError('The provided metadata object was not found in the SEAFile.')

    def remove(self, obj: SEASerializable | str | int,
               location: Literal['Simulations', 'Experiments', 'Analysis', 'metadata'] | None = None
               ) -> SEASerializable | None:
        """
        Remove an object (or first matching name/index) from the SEA file.

        Parameters
        ----------
        obj : SEASerializable | str | int
            Dataset instance, name, index, or Metadata.
        location : {'Simulations', 'Experiments', 'Analysis', 'metadata'} | None, optional
            Target scope. When ``None``, searches collections first then
            metadata.

        Returns
        -------
        SEASerializable | None
            The removed object.

        Raises
        ------
        ValueError
            If the object cannot be found or removal is invalid for the target.

        Returns
        -------
        SEASerializable | None
            The removed object.
        """
        normalized_location = self._normalize_location(location)

        if normalized_location == 'metadata':
            return self._remove_metadata(obj)

        if normalized_location is not None:
            collection = self._get_collection(normalized_location)
            removed = self._attempt_remove_from_collection(collection, obj)
            if removed is not None:
                return removed
            raise ValueError(f'The provided object was not found in the {normalized_location} collection.')

        for _, collection in self._collections().items():
            removed = self._attempt_remove_from_collection(collection, obj)
            if removed is not None:
                return removed

        if isinstance(obj, (Metadata, str)):
            try:
                return self._remove_metadata(obj)
            except (ValueError, KeyError):
                pass

        raise ValueError('The provided object was not found in the SEAFile.')

    @classmethod
    def load(cls, file_path: str) -> "SEAFile":
        """
        Create a SEAFile from a SEA-formatted file.

        Parameters
        ----------
        file_path : str
            Path to a ``.sea`` file.

        Returns
        -------
        SEAFile
            Loaded SEAFile instance.
        """
        instance = cls()
        instance.from_sea(file_path)
        return instance

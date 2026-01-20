# -*- coding: utf-8 -*-
"""
This submodule provides basic plotting functionality for the SEA ecosystem.


ToDo
----
PROJECT: Create a class system that allows for matplotlib saving, loading, and editing
TODO: Include a general plot function similar to PlotImage
? Should PlotImage be renamed to imshow?


"""
#Imports: Typing
from __future__ import annotations
from typing import Any, List, Dict, Literal
from matplotlib.axes import Axes as mplAxes
from numpy.typing import NDArray

#Imports: External
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

#Imports: Internal
from pySEA.sea_eco._signal_processing.normalization import nv_correction

def closest_nice_number(number, round_cuttoff=8):
        oom = 10 ** np.floor(np.log10(number)) #order of magnitude
        if number/oom > round_cuttoff:
            return oom*10
        else:
            return oom * (number // oom)

# Returns length/width and label with units. if shape is None, length/width are real-space units. if shape is given, we round to pixel dimensions

def calculateScaleBar(xax_size,yax_size,units,length_percent=30,hw_ratio=5,shape=None):
    length = closest_nice_number(xax_size * length_percent/100)
    height = length*hw_ratio/100
    units,[l2] = changeUnits(units,[length])
    label = str(int(l2))+" "+units
    if shape is not None:
        length = int(np.round(shape[1]*length/xax_size))
        height = int(np.ceil(shape[0]*height/yax_size))
    return length,height,label

unitLookup={"rad":{"mrad":1e3},"m":{"mm":1e3,"um":1e6,"nm":1e9,"Å":1e10}}
def changeUnits(unit,vals): # Why take a list of vals? in case the caller wants more things scaled
    if vals[0]>=1:
        return unit,vals
    vals = np.asarray(vals)
    for k in unitLookup.keys(): # loop base units, rad or m etc
        if k not in unit:			# e.g. base unit "m" in "nm" 
            continue
        s1=unitLookup[k].get(unit,1) # "nm" has 1e9 scaling initially
        candidate_units = [] ; candidate_vals = []
        for kk in unitLookup[k].keys(): # loop through mm, um, nm, etc
            s2=unitLookup[k][kk]
            if vals[0]*s2/s1 > 1:
                candidate_units.append(kk) ; candidate_vals.append(vals*s2/s1)
        if len(candidate_units)==0: # didn't find any smaller units
            return unit,vals
        candidate_vals = np.asarray(candidate_vals)
        i=np.argmin(candidate_vals[:,0])
        print("scaling units",unit,"to",candidate_units[i],", val",vals[0],"becomes",candidate_vals[i][0])
        return candidate_units[i],candidate_vals[i]
    return unit,vals


class ScaleBar(AnchoredSizeBar):
    def __init__(self, ax: mplAxes|None = None, size: int|float|None = None, label: str|None = None,
                 color: str = 'white', size_vertical: str|int|float = None, kw_scale: Dict = {}, kw_font: Dict = {},
                 pixel_size: float|None = None, units: str|None = None, max_size_ratio: float = 0.25):
        '''
        Add an axes bar
        
        Parameters
        ----------
        ax: .axis
            Axis to add scale bar to.
        size: flaot
            Size of the scale bar in data cordinates.
        label: str
            Label to display above scale bar.
        color: str
            Color of the bar and the text.
        kw_scale: dict
            Dictionary of kwargs to supply to AnchoredSizeBar.
        kw_font: dict
            Dictionary of kwargs to supply to FontProperties.
        '''
        self.ax = ax
        self.pixel_size = pixel_size
        self.units = units if units is not None else ''
        #self.size = size if size is not None else self.calculate_size(max_size_ratio=max_size_ratio)
        #self.size_vertical = self.calculate_vertical_size(size_vertical) if size_vertical is not None else 0
        #self.label = label if label is not None else self.generate_label()
        self.color = color
    
        l,h,lbl = calculateScaleBar(xax_size=np.ptp(self.ax.set_xlim()),
			yax_size=np.ptp(self.ax.set_xlim()),
			units=units)

        self.size = size if size is not None else l
        self.size_vertical = size_vertical if size_vertical is not None else h
        self.label = label if label is not None else lbl


        fontprops = {'size': 8}
        for k,i in kw_font.items():
            fontprops[k] = i
        
        scaleprops = {'pad':0.1,
                      'color':self.color,
                      'frameon':False,
                      'label_top':True,
                      'size_vertical':self.size_vertical,
                       'loc':'lower left'}
        for k,i in kw_scale.items():
            scaleprops[k] = i
            
        
        fontprops = fm.FontProperties(**fontprops)

        if ax is not None:
            scalebar = AnchoredSizeBar(
                ax.transData,
                self.size, self.label,
                fontproperties=fontprops,
                **scaleprops)
            #scalebar0.size_bar.get_children()[0].fill = True
            ax.add_artist(scalebar)


    def calculate_size(self, max_size_ratio: float = 0.25):
        '''Calculate the size of the bar.'''
        xlims = self.ax.set_xlim()
        size = closest_nice_number(np.ptp(xlims) * max_size_ratio)
        self.size = size
        return size
    
    def calculate_vertical_size(self, size_vertical: str|int|float):
        '''Calculate the vertical size of the bar.

        Parameters
        ----------
        size_vertical: str, int, flaot
            If str then the value will be calculated based on the y-axis extents.
            If the last character is % then the string will be converted to a percent, otherwise it is assumed the value is a desired fraction.
            If int or float then absolute units are assumed and the input value is returned.
        '''
        if isinstance(size_vertical, str):
            yax_size = np.ptp(self.ax.set_ylim())
            if size_vertical[-1] == '%':
                size_vertical = yax_size * float(size_vertical[:-1]) / 100
            else:
                size_vertical = yax_size * float(size_vertical)
        return size_vertical


        size = closest_nice_number(np.ptp(xlims) * max_size_ratio)
        self.size = size
        return size
    
    def generate_label(self):
        if self.size%1 == 0:
            return f'{int(self.size):d} ' + self.units
        else:
            return f'{round(self.size, 1):.1f} ' + self.units

class PlotImage(object):
    def __init__(self, data, ax=None,
                 #? norm=None, #I am not sure this kwarg currently serves any purpose in __init__ as it is a kwarg for imshow. However, it could be nice for dyanmic update of the norm.
                 ticks_and_labels: Literal['off','empty','on'] = 'off', axes_info: List[Dict] = None,
                 scale_bar: bool = True, scale_bar_kwargs: Dict = {},
                 fix_nv: bool = False, fix_nv_kwargs: Dict = {}, 
                 **kwargs):
        """
         A shortcut plotting function for imshow that automatically handles things like imshow kwargs and intenisty bounds.
            
        Parameters
        ----------
        data: array-like
            Two dimintional data set to plot as an image.
        ax: matplotlib axis
            Axis to plot to. If None then an axis is both created and returned.
        norm: matplotlib normalization
            Normalization to be used. Default is None.
            This will overide any kwargs['norm'] provided and is implemented as an arg due to it frequent use.
        ticks_and_labels: Str
            What to do with the axes ticks and borders.
            off:    turn off the axes (default).
            empty:  turn the ticks and labels off but leave the borders.
            on:     keep the ticks and borders in on.
        axes_info: list of dict
            A list of dict where each dict contains information for the axis labeling and/or scale bar.
            Implemented dict keys are: 'name', and 'units'
        scale_bar: boolean
            Add a scalebar object to the image if True. Default if True.
            For brevity, if scale_bar_kwargs is populated then scal_bar is set to True.
        scale_bar_kwargs: dict
            kwargs for scale_bar.
        fix_nv: boolean
            Correct for negative vlaues.
        fig_nv_kwargs: dict
            kwargs for nn_correction.
        **kwarg:
            kwargs supplied to imshow.

        Returns
        -------
        None


        TODO
        ----
        Create save/load definition that saves/loads to hdf5.
            - It may be worth creating a keyword that allows for axis updates.
            - May need a parent axis class if ax.plot is implemented.
                - Allows for multiple images and/or plots.
                - The previous update would help with this
                - Then allow for saveing the axis class or the image class in the respective classes.
        """
        
        self.data = data
        self.ax = ax if ax is not None else plt.gca()
        self.ticks_and_labels = ticks_and_labels
        self.axes_info = axes_info

        self.imshow_kwargs = kwargs #asign the imshow kwargs so that they can be recalled later


        #TODO: Make fix_nv dynamic. Move out of __init__?
        if fix_nv:
            self.data = nv_correction(self.data, **fix_nv_kwargs)

        self.show_img()

        #TODO: Have everything after `self.show_img()` in __init__ wrapped into a `self.decorate_image()` function to clean up init and updates.
        
        # Add scale_bar
        if scale_bar and len(scale_bar_kwargs)>0:
            self.scale_bar = ScaleBar(self.ax, **scale_bar_kwargs)

        if self.ticks_and_labels is not None:
            self.set_ticks_and_labels(ticks_and_labels)

    def show_img(self):
        self.img = self.ax.imshow(self.data, **self.imshow_kwargs)

    def set_ticks_and_labels(self, state):
        if state == 'off':
            self.ax.axis('off')
        elif state == 'empty':
            self.ax.set_yticks([])
            self.ax.set_xticks([])
        elif state == 'on':
            if self.axes_info is not None:
                for cax, dax in zip([self.ax.xaxis, self.ax.yaxis],
                                     self.axes_info):
                    cax.set_label_text(f"{dax['name']} ({dax['units']})")
                pass
            else:
                pass

    def add_scale_bar(self, **kwargs) -> ScaleBar:
        '''Add a scale bar to the image.

        Parameters
        ----------
        kwargs: dict
            kwargs for ScaleBar class.

        Returns
        -------
        scalebar: ScaleBar
            The created ScaleBar object.
        '''
        if kwargs.get('units') is None and self.axes_info is not None:
            kwargs['units'] = self.axes_info[0]['units']
        scale_bar = ScaleBar(self.ax, **kwargs)
        return scale_bar

def plot_nd_array(array:NDArray, ax: mplAxes|None = None, **kwargs) -> PlotImage | None:
    """A shortcut plotting function that automatically handles things like dimensionality and matplotlib kwargs.
            
    Parameters
    ----------
    data: array-like
        Data to plot.
    ax: matplotlib axis
        Axis to plot to. If None then an axis is both created and returned.
    **kwarg:
        kwargs supplied to imshow.

    Returns
    -------
    plot: PlotImage or None
        Generated object that was plotted in the axis.
    """
    if ax is None: ax = plt.gca()

    kwargs_keys = ['title', 'xlabel', 'ylabel', 'xlim', 'ylim']
    ax.set(**{k:kwargs.pop(k) for k in kwargs_keys if k in kwargs.keys()})
    
    if array.ndim == 2:
        plot = PlotImage(array, ax=ax, **kwargs)
    elif array.ndim == 1:
        if 'x' in kwargs:
            x = kwargs.pop('x')
            plot = ax.plot(x, array, **kwargs)
        else:
            ax.plot(array)
    else:
        raise NotImplementedError('Only plotting of 1D and 2D arrays are supported at this time.')

    return plot

def save_fig(file_name, fig=None, file_types=['svg','png'], **kwargs):
    if fig is None: fig = plt.gcf()
    for ft in file_types:
        fig.savefig(file_name+'.'+ft, **kwargs)

def pannel_title(axs, pos=[-.2, 1], end='', title=False, **kwargs):
    '''
    Adds an alphabetical label to the figure pannels.
    '''
    alpha = 'abcdefghijklmnopqrstuvwxyz'

    for i, ax in enumerate(axs.flatten()):
        label = alpha[i]+end
        if title:
            if 'loc' not in kwargs.keys():
                kwargs['loc'] = 'left'
            ax.set_title(label, **kwargs)
        else:
            ax.text(pos[0], pos[1], label, transform=ax.transAxes,
                    fontweight='bold', va='top', ha='right', **kwargs)

def save_image(array:NDArray,size:tuple,units:str|None=None,filename=None,file_types=['png']):
    from PIL import Image, ImageDraw
    # Load and scale the data
    data = np.zeros(array.shape)+array
    data -= np.nanmin(data)	# shift min value to zero (and get rid of negatives)
    data /= np.amax(data)	# scale max value to 1
    data *= 255				# 255 max val for 8 bit int
    data = data.astype(np.uint8)
    # calculate scalebar position/size/label
    l,h,lbl = calculateScaleBar(size[1],size[0],units,shape=array.shape)
    xi=int(array.shape[1]/20) ; xf=xi+l
    yi=int(array.shape[0]*19/20)-h; yf=yi+h
    # paint on the scale bar
    data[yi:yf,xi:xf]=255
    # construct PIL Image
    img = Image.fromarray(data.astype(np.uint8),"L")
    # paint on the label
    draw = ImageDraw.Draw(img)						# drawing object so we can write text centered on bar
    draw.text(((xi+xf)//2, yi-2*h),lbl,255,anchor='mm',font_size=2*h,color='w')	# e.g. "10 nm" label for bar

    if filename is not None:
        for ft in file_types:
            img.save(filename+"."+ft)
    else:
        img.show()


#X def get_image_extent(signal:'Signal', origin:str='upper') -> List:
#     """
#     Get an array of the image extents for plotting.

#     Parameters
#     ----------
#     signal : object
#         _description_
#     origin : str, optional
#         _description_, by default 'upper'

#     Returns
#     -------
#         Array of extents

#     List
#     Raises
#     ------
#     KeyError
#         Origin incorrectly set.
#     """

#     if isinstance(signal):
#         axX = signal.dimensions[0]
#         axY = signal.dimensions[1]
#         extent = [min(axX.values), max(min(axX.values)), min(axY.values), max(axY.values)]
        
#         if origin == 'upper':
#             extent[2:] = extent[2:][::-1]
#         elif origin == 'lower':
#             extent = extent
#         else:
#         return extent
#             raise KeyError(f'origin must be upper or lower. A value of {origin} was passed.')

def legend_2axis(axes, labels='auto', display_axis=0, **kwargs):
    '''
    Plot a legend for a multi-axis subplot.
    
    Parameters
    ----------
    axes : tuple or list
        List of axes containing labels for the legend.
    labels : bool
        If True, the ionization edges with an onset below the lower
        energy limit of the SI will be included.
    kwargs: dict
        kwargs for the matplotlib legend function.
    '''
    lines = []
    labels = []
    for ax in axes:
        li, la = ax.get_legend_handles_labels()
        lines += li
        labels += la
    
    if labels == 'auto':
        axes[0].legend(lines,  labels, **kwargs)
    elif labels is not None:
        axes[0].legend(lines,  labels, **kwargs)
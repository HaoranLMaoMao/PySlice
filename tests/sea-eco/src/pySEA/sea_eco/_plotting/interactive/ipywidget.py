"""ipywidgets-based interactive plotting utilities for SEA-eco."""

#Imports: Typing
from __future__ import annotations
from typing import Optional, Tuple, List, Callable
from pySEA.sea_eco.architecture.base_structure_numpy import Signal

#Imports: External
from IPython.display import display
from ipywidgets import Dropdown, HBox, VBox, IntSlider, Label
from matplotlib.pyplot import subplots
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy import sum as sum

#Imports: Internal
from pySEA.sea_eco._plotting.interactive import get_nav_plot_data

def create_dimension_selector(signal: Signal,
                             nav_dimensions: Optional[Tuple[int, ...]] = None,
                             sig_dimensions: Optional[Tuple[int, ...]] = None) -> Tuple[HBox, Callable]:
    """Create a dimension selector widget with four dropdowns for navigation and signal dimensions.
    
    Returns:
    --------
    widget : HBox
        The widget containing the four dropdowns
    active_dimensions : Callable
        Function that returns current (nav_dimensions, sig_dimensions, other_dimensions)
    """
    all_dimensions = list(range(signal.data.ndim))
    if nav_dimensions is None: nav_dimensions = tuple(signal.dimensions.nav_dimensions)
    if sig_dimensions is None: sig_dimensions = tuple(signal.dimensions.sig_dimensions)
    
    # Create dimension options
    dimensions_options = [(n, i) for i,n in enumerate(signal.dimensions.get_names())]
    dimensions_options.append(('None', None))
    
    # Create dropdowns
    nav_dropdown1 = Dropdown(options=dimensions_options, value=nav_dimensions[0], description='Nav Y:')
    nav_dropdown2 = Dropdown(options=dimensions_options, 
                           value=nav_dimensions[1] if len(nav_dimensions) > 1 else None, 
                           description='Nav X:')
    sig_dropdown1 = Dropdown(options=dimensions_options, value=sig_dimensions[0], description='Sig Y:')
    sig_dropdown2 = Dropdown(options=dimensions_options, 
                           value=sig_dimensions[1] if len(sig_dimensions) > 1 else None, 
                           description='Sig X:')
    
    def check_dropbox_duplicates(*_):
        """Ensure no dimension is selected multiple times."""
        if nav_dropdown2.value == nav_dropdown1.value or\
           nav_dropdown2.value == sig_dropdown1.value or\
           nav_dropdown2.value == sig_dropdown2.value:
                nav_dropdown2.value = None
        if nav_dropdown1.value == sig_dropdown1.value or\
           nav_dropdown1.value == sig_dropdown2.value:
                nav_dropdown1.value = nav_dropdown2.value
                nav_dropdown2.value = None
        if nav_dropdown2.value == sig_dropdown1.value or\
           nav_dropdown2.value == sig_dropdown2.value:
                nav_dropdown2.value = None
        if sig_dropdown1.value == sig_dropdown2.value: 
            sig_dropdown2.value = None
    
    def get_active_dimensions():
        """Get current dimension selections."""
        check_dropbox_duplicates()
        nav_dims = tuple(ax for ax in (nav_dropdown1.value, nav_dropdown2.value) if ax is not None)
        sig_dims = tuple(ax for ax in (sig_dropdown1.value, sig_dropdown2.value) if ax is not None)
        other_dims = tuple(i for i in all_dimensions if i not in nav_dims + sig_dims)
        return nav_dims, sig_dims, other_dims

    # Connect callbacks for duplicate checking
    for dropdown in [nav_dropdown1, nav_dropdown2, sig_dropdown1, sig_dropdown2]:
        dropdown.observe(check_dropbox_duplicates, names='value')
    
    # Create widget
    dimension_selector = HBox([nav_dropdown1, nav_dropdown2, sig_dropdown1, sig_dropdown2])

    return dimension_selector, get_active_dimensions

def create_norm_selector() -> Tuple[HBox, Callable[[bool, bool], None], Callable[[], Tuple[str, str]]]:
    """Create normalization dropdowns for nav and signal plots."""
    options_2d = [('None', 'none'), ('Log', 'log')]
    options_1d = [('None', 'none'), ('Log X', 'logx'), ('Log Y', 'logy'), ('Log X & Y', 'logxy')]

    nav_norm = Dropdown(options=options_2d, value='none', description='Nav norm:')
    sig_norm = Dropdown(options=options_2d, value='none', description='Sig norm:')

    def _set_options(nav_is_2d: bool, sig_is_2d: bool) -> None:
        """Adjust available options based on dimensionality."""
        def _apply(dropdown: Dropdown, opts: List[Tuple[str, str]]) -> None:
            current = dropdown.value
            dropdown.options = opts
            allowed = [v for _, v in opts]
            if current not in allowed:
                dropdown.value = allowed[0]

        _apply(nav_norm, options_2d if nav_is_2d else options_1d)
        _apply(sig_norm, options_2d if sig_is_2d else options_1d)

    def _get_norms() -> Tuple[str, str]:
        return nav_norm.value, sig_norm.value

    return HBox([nav_norm, sig_norm]), _set_options, _get_norms

def create_navigation_sliders(signal: Signal, 
                            active_dimensions: Callable,
                            update_callback: Optional[Callable] = None) -> Tuple[VBox, Callable, Callable, List[IntSlider]]:
    """Create navigation sliders for the signal.
    
    Returns:
    --------
    widget : VBox
        Container with the active sliders (each slider paired with a right-hand label)
    get_values : Callable
        Function that returns current slider values for given dimensions
    update_visible : Callable
        Function to refresh which sliders are visible
    sliders_list : list
        Flat list of IntSlider objects (useful for attaching observers)
    """
    all_dimensions = list(range(signal.data.ndim))
    sliders: List[IntSlider] = []
    label_widgets: List[Label] = []
    pair_containers: List[HBox] = []
    
    # helper to format calibrated value + units
    def _format_calibrated(dim, idx):
        try:
            cal = dim.get_calibrated_value(idx)
        except Exception:
            vals = getattr(dim, "values", None)
            cal = float(vals[idx]) if (vals is not None and len(vals) > idx) else float(idx)
        units = getattr(dim, "units", "")
        if units:
            return f"{cal:.4g} {units}"
        return f"{cal:.4g}"
    
    # Create slider + right-label pair for all dimensions
    for ax_idx in all_dimensions:
        dim = signal.dimensions.dimensions[ax_idx]
        slider = IntSlider(
            value=0,
            min=0,
            max=signal.data.shape[ax_idx]-1,
            description=f'{dim.name}',  # left label stays short
            continuous_update=False,
        )
        label = Label()  # right-side text (calibrated)
        
        # updater to refresh the right-hand label when slider value changes
        def _make_label_updater(a_idx, s=slider, l=label):
            def _update_label(change=None):
                idx = int(s.value)
                d = signal.dimensions.dimensions[a_idx]
                l.value = f"({_format_calibrated(d, idx)})"
            return _update_label
        
        updater = _make_label_updater(ax_idx)
        slider.observe(updater, names='value')
        # initialize label
        updater()
        
        # attach external update callback if provided
        if update_callback:
            slider.observe(update_callback, names='value')
        
        sliders.append(slider)
        label_widgets.append(label)
        pair_containers.append(HBox([slider, label], layout={'display': 'none'}))
    
    def update_visible_sliders(*_):
        """Update which sliders are visible based on current dimension selection."""
        nav_dims, sig_dims, other_dims = active_dimensions()
        slider_dims = other_dims + nav_dims
        
        # Show/hide pair containers and refresh label text
        for i, container in enumerate(pair_containers):
            if i in slider_dims:
                container.layout.display = 'flex'
                # refresh right-hand label text to current calibrated value
                sliders[i].observe(lambda *_: None, names='value')  # ensure .value exists
                d = signal.dimensions.dimensions[i]
                idx = int(sliders[i].value)
                label_widgets[i].value = f"({_format_calibrated(d, idx)})"
            else:
                container.layout.display = 'none'
    
    def get_slider_values(dimensions: List[int]) -> List[int]:
        """Get current integer index values for the specified dimensions.""" 
        return [sliders[ax].value for ax in dimensions]
    
    # Create widget container: keep same overall layout (pairs arranged horizontally)
    slider_box = HBox(pair_containers)
    
    return slider_box, get_slider_values, update_visible_sliders, sliders

def create_signal_plots(signal: Signal, 
                       active_dimensions: Callable,
                       get_slider_values: Callable,
                       get_norms: Callable[[], Tuple[str, str]],
                       set_norm_options: Callable[[bool, bool], None],
                       nav_fnc: Callable = sum
                       ) -> Tuple[Figure, Axes, Callable]:
    """Create the signal plot display.

    Parameters:
    -----------
    signal : NDArray
        Input array with arbitrary dimensions
    active_dimensions : Callable
        Funciton to call to retrieve active dimensions
    get_norms : Callable
        Function returning (nav_norm, sig_norm)
    set_norm_options : Callable
        Function to adjust norm dropdown options based on dimensionality
    nav_fnc : Callable, optional
        Function to apply along function_dimensions (default: sum)
    
    Returns:
    --------
    figure : plt.Figure
        The matplotlib figure containing the plots
    nav_ax : matplotlib.axes.Axes
        Axis showing the navigation view (used for click selection)
    update : Callable
        Function to update the plots

    ToDo
    ----
    ?: I do not think this function should not be updated internally.
        Here we just create a figure then have an internal funciton that calls the update `self.update_display(*_)`, then returns the fig and update function.
    TODO: The nav and signal plot should not both be updated all the time.
            The signal and nav marker always requires updates.
            They nav nav only requires updates when nav dims change. This is a larger computation.
            
    """
    # Create figure
    fig, (nav_ax, sig_ax) = subplots(1, 2, figsize=(12, 5))
    
    def update_display(*_):
        """Update both plots based on current dimensions and slider values."""
        nav_dimensions, sig_dimensions, other_dimensions = active_dimensions()
        
        # Skip if dimensions aren't properly set
        if not nav_dimensions or not sig_dimensions:
            return
            
        # Create index array for signal plot
        active_values = [
            slice(None) if ax in sig_dimensions 
            else get_slider_values([ax])[0]
            for ax in range(signal.data.ndim)
        ]
        
        # Get plot data
        other_values = get_slider_values(other_dimensions)
        nav_sig = get_nav_plot_data(signal, 
                                    function_dimensions=sig_dimensions, 
                                    index_dimensions=other_dimensions,
                                    index_values=other_values,
                                    nav_fnc=nav_fnc
                                    )
        sig_sig = signal[tuple(active_values)]
        
        # Clear and redraw plots
        # TODO: update instead of clear
        nav_ax.clear()
        sig_ax.clear()
        
        if len(nav_sig.dimensions)==2:
            nav_kwargs = dict(ticks_and_labels='on', scale_bar=False, aspect='auto')
        else: nav_kwargs = dict()

        if len(sig_sig.dimensions)==2:
            sig_kwargs = dict(ticks_and_labels='on', scale_bar=False, aspect='auto')
        else: sig_kwargs = dict()

        # Plot data with proper dimensions
        nav_sig.show(ax=nav_ax, **nav_kwargs)
        sig_sig.show(ax=sig_ax, **sig_kwargs)

        # Update norm selectors to valid options based on dimensionality
        nav_is_2d = len(nav_sig.dimensions) == 2
        sig_is_2d = len(sig_sig.dimensions) == 2
        set_norm_options(nav_is_2d, sig_is_2d)
        nav_norm, sig_norm = get_norms()

        def _apply_norm(ax: Axes, norm_choice: str, is_2d: bool):
            # reset scales
            ax.set_xscale('linear')
            ax.set_yscale('linear')
            if is_2d:
                for im in ax.images:
                    try:
                        data = np.asarray(im.get_array())
                        data = data[np.isfinite(data)]
                        if data.size == 0:
                            continue
                        vmin = float(np.nanmin(data))
                        vmax = float(np.nanmax(data))
                        if norm_choice == 'log':
                            if vmin <= 0:
                                # fall back to linear scaling if data not strictly positive
                                im.set_norm(colors.Normalize(vmin=vmin, vmax=vmax))
                            else:
                                im.set_norm(colors.LogNorm(vmin=vmin, vmax=vmax))
                        else:
                            im.set_norm(colors.Normalize(vmin=vmin, vmax=vmax))
                    except Exception:
                        pass
                return
            xscale = 'log' if norm_choice in ('logx', 'logxy') else 'linear'
            yscale = 'log' if norm_choice in ('logy', 'logxy') else 'linear'
            try:
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
            except Exception:
                pass

        _apply_norm(nav_ax, nav_norm, nav_is_2d)
        _apply_norm(sig_ax, sig_norm, sig_is_2d)
        
        # Add navigation indicators
        if len(nav_dimensions) == 2:
            # slider indices for the two nav dims
            nav_i = get_slider_values(nav_dimensions)
            # calibrated values for the nav dims using the original signal dimensions
            nav_val = [signal.dimensions[ax].get_calibrated_value(i) for ax, i in zip(nav_dimensions, nav_i)]

            # draw crosshairs at the calibrated navigation point
            nav_ax.axhline(nav_val[0], color='r')
            nav_ax.axvline(nav_val[1], color='r')
        elif len(nav_dimensions) == 1:
            # single nav dim: convert index to calibrated coordinate and draw marker
            idx = get_slider_values(nav_dimensions)[0]
            cal_val = signal.dimensions[nav_dimensions[0]].get_calibrated_value(idx)
            nav_ax.axvline(cal_val, color='r')
        
        sig_ax.set_title('Signal')
        fig.canvas.draw_idle()
    
    return fig, nav_ax, update_display

def interactive_signal_plot(signal: Signal,
                          nav_dimensions: Optional[Tuple[int, ...]] = None,
                          sig_dimensions: Optional[Tuple[int, ...]] = None,
                          nav_fnc: Callable = sum,
                          show_dimension_selector: bool = True,
                          show_norm_selector: bool = True) -> None:
    """Create a complete interactive signal plot with optional dimension selector.
    
    Parameters:
    -----------
    signal : Signal
        The signal to visualize
    nav_dimensions : tuple, optional
        Initial navigation dimensions
    sig_dimensions : tuple, optional
        Initial signal dimensions
    nav_fnc : Callable, optional
        Function to apply along function_dimensions (default: sum)
    show_dimension_selector : bool
        Whether to show the dimension selection widget (default: True)
    show_norm_selector : bool
        Whether to show the normalization selector (default: True)
    """
    # Ensure all dimensions are int.
    # HACK: This is currelty needed for when dimensions are supplied manually (without selector)
    if nav_dimensions: nav_dimensions = signal.dimensions.get_dims_as_int(nav_dimensions)
    if sig_dimensions: sig_dimensions = signal.dimensions.get_dims_as_int(sig_dimensions)

    # Create dimension selector first
    dim_selector, active_dimensions = create_dimension_selector(
        signal, nav_dimensions, sig_dimensions
    )
    
    # Create navigation sliders second (before plots need them)
    slider_box, get_slider_values, update_sliders, sliders_list = create_navigation_sliders(
        signal, active_dimensions, None  # We'll update the callback after creating plots
    )

    # Create norm selector
    norm_selector, set_norm_options, get_norms = create_norm_selector()
    
    # Create plots with proper slider value access
    fig, nav_ax, update_display = create_signal_plots(
        signal, active_dimensions, get_slider_values, get_norms, set_norm_options, nav_fnc=nav_fnc
    )
    
    # Now we can set up all the callbacks
    def update_all(*_):
        update_sliders()
        update_display()
    
    # Update slider callbacks now that we have update_display
    for slider in sliders_list:
        slider.observe(update_display, names='value')
    
    # Connect dimension selector callbacks
    for dropdown in dim_selector.children:
        dropdown.observe(update_all, names='value')

    # Connect norm selector callbacks
    for dropdown in norm_selector.children:
        dropdown.observe(update_all, names='value')
    
    # Create layout and display
    controls = [slider_box]
    if show_dimension_selector:
        controls = [dim_selector] + controls
    if show_norm_selector:
        controls = [norm_selector] + controls
    if len(controls)>1:
        controls = VBox(controls)
    else:
        controls = controls[0]
    
    # Allow ctrl+click on the navigation plot to jump to a location
    def _on_nav_click(event):
        # Only respond to ctrl+clicks within the nav axis with valid coordinates
        key = (event.key or "").lower()
        if event.inaxes is not nav_ax or ('control' not in key and 'ctrl' not in key):
            return
        if event.xdata is None and event.ydata is None:
            return

        nav_dims, _, _ = active_dimensions()
        if len(nav_dims) == 2 and event.xdata is not None and event.ydata is not None:
            try:
                y_idx = int(signal.dimensions[nav_dims[0]].find_nearest_index(event.ydata, warn_bounds=False))
                x_idx = int(signal.dimensions[nav_dims[1]].find_nearest_index(event.xdata, warn_bounds=False))
            except Exception:
                return
            sliders_list[nav_dims[0]].value = y_idx
            sliders_list[nav_dims[1]].value = x_idx
        elif len(nav_dims) == 1:
            # For 1D nav, use x coordinate if available
            target_val = event.xdata if event.xdata is not None else event.ydata
            if target_val is None:
                return
            try:
                idx = int(signal.dimensions[nav_dims[0]].find_nearest_index(target_val, warn_bounds=False))
            except Exception:
                return
            sliders_list[nav_dims[0]].value = idx

    fig.canvas.mpl_connect('button_press_event', _on_nav_click)

    display(controls)
    # Initial update
    update_all()

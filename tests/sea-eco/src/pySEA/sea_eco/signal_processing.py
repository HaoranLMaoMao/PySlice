# -*- coding: utf-8 -*-
"""Public signal-processing API for SEA-eco."""
from pySEA.sea_eco._signal_processing import maxND, shift, \
    correlate_1D_in_ND, autocorrelate, get_shifts_autocorrelate
from pySEA.sea_eco._signal_processing.normalization import nv_correction, normalize, normalize_by_ZLP
from pySEA.sea_eco._signal_processing.peak_parameters import estimate_FWPM, estimate_FWPM_center, estimate_skew
from pySEA.sea_eco._signal_processing.spikeremoval import  remove_spikes, plot_spike_histogram

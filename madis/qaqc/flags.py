"""Bit-encoded QC flags for MADIS QA/QC output."""

import numpy as np
import xarray as xr

from madis.qaqc.filters import filter_completeness, filter_run_on_other_da

QC_FLAG_DTYPE = np.uint16
"""Integer type used to store QC flags."""
QC_FLAG_NBITS = np.iinfo(QC_FLAG_DTYPE).bits
"""Number of bits in QC_FLAG_DTYPE."""
QC_FLAG_SUFFIX = '_qc_flag'
"""Suffix for QC flag variable names."""

QC_FLAG_BITS = {
    'missing_raw': 0,
    'filter_completeness_initial': 1,
    'filter_completeness_final': 2,
    'filter_spikes': 3,
    'filter_spike_groups': 4,
    'filter_value_range': 5,
    'filter_linear_gaps': 6,
    'filter_constant': 7,
    'filter_percentile': 8,
    'filter_changepoint': 9,
    'filter_error': 10,
    'filter_mean': 11,
    'filter_seasonal_anomalies': 12,
    'mask_dependencies': 13,
}
"""Mapping of QC flag names to bit positions in the QC flag variable."""

assert max(QC_FLAG_BITS.values()) < QC_FLAG_NBITS
assert len(set(QC_FLAG_BITS.values())) == len(QC_FLAG_BITS)


def flag_var_name(var_name):
    """Return the name of the QC flag variable for a given variable.

    Parameters
    ----------
    var_name : str
        Name of the cleaned variable, e.g. "windspeed_10m".

    Returns
    -------
    str
        QC flag variable name, e.g. "windspeed_10m_qc_flag".
    """
    return f'{var_name}{QC_FLAG_SUFFIX}'


def flag_key(fun, kwargs, stage):
    """Return the QC flag name associated with a filter and filter stage.

    Parameters
    ----------
    fun : callable
        Filter function used in a run_filter_bank stage.
    kwargs : dict
        Keyword arguments passed to the filter function.
    stage : int
        Zero-based filter stage index.

    Returns
    -------
    str
        QC flag name for the filter result.
    """
    if fun is filter_run_on_other_da:
        return kwargs['base_filter'].__name__
    if fun is filter_completeness:
        return (
            'filter_completeness_initial'
            if stage == 0
            else 'filter_completeness_final'
        )
    return fun.__name__


def _bit_position(key):
    """Return the QC flag bit position for a flag name.

    Parameters
    ----------
    key : str
        QC flag name.

    Returns
    -------
    int
        Zero-based position of the flag bit.

    Raises
    ------
    KeyError
        If no QC flag bit has been assigned to the key.
    """
    try:
        return QC_FLAG_BITS[key]
    except KeyError:
        raise KeyError(
            f"No QC flag bit assigned to '{key}'. Add it to QC_FLAG_BITS."
        ) from None


def _mask_values(mask, template):
    """Convert a filter mask to a boolean NumPy array matching a template.

    Parameters
    ----------
    mask : array-like or xr.DataArray
        Filter mask to convert.
    template : xr.DataArray
        DataArray whose dimensions and shape define the output mask.

    Returns
    -------
    np.ndarray
        Boolean array with the template's dimensions and shape.
    """
    if isinstance(mask, xr.DataArray):
        m = mask.transpose(*template.dims).values
    else:
        m = np.broadcast_to(np.asarray(mask), template.shape)
    return np.asarray(m, dtype=bool)


def encode_flags(filter_masks, template):
    """Encode filter masks as a uint16 QC flag bitmask.

    Parameters
    ----------
    filter_masks : dict of {str: list}
        Mapping of QC flag names to lists of masks.
    template : xr.DataArray
        DataArray defining the output dimensions, coordinates, and name.

    Returns
    -------
    xr.DataArray
        uint16 QC flag bitmask with CF flag metadata.

    Raises
    ------
    KeyError
        If a filter mask key does not have an assigned flag bit.
    """
    flag = np.zeros(template.shape, dtype=QC_FLAG_DTYPE)
    for key, masks in filter_masks.items():
        bit = QC_FLAG_DTYPE(1 << _bit_position(key))
        for mask in masks:
            flag[_mask_values(mask, template)] |= bit
    name = flag_var_name(template.name) if template.name else None
    out = xr.DataArray(
        flag, dims=template.dims, coords=template.coords, name=name
    )
    out.attrs.update(cf_flag_attrs())
    return out


def decode_flags(flag_da, keys=None):
    """Decode selected bits from a QC flag bitmask.

    Parameters
    ----------
    flag_da : xr.DataArray
        uint16 QC flag bitmask.
    keys : iterable of str, optional
        QC flag names to decode. If omitted, decode every configured flag.

    Returns
    -------
    dict of {str: xr.DataArray}
        Mapping of QC flag names to boolean DataArrays.
    """
    keys = QC_FLAG_BITS if keys is None else keys
    return {
        key: (flag_da & QC_FLAG_DTYPE(1 << _bit_position(key))) != 0
        for key in keys
    }


def cf_flag_attrs():
    """Build CF metadata attributes for the QC flag bitmask.

    Returns
    -------
    dict
        Mapping containing the long name, flag masks, and flag meanings.
    """
    names = sorted(QC_FLAG_BITS, key=QC_FLAG_BITS.get)
    return {
        'long_name': 'QC flag bitmask',
        'flag_masks': np.array(
            [1 << QC_FLAG_BITS[n] for n in names],
            dtype=QC_FLAG_DTYPE,
        ),
        'flag_meanings': ' '.join(names),
    }


def build_flag(raw, filter_masks, parent_mask=None):
    """Build the QC flag variable for one cleaned variable.

    Parameters
    ----------
    raw : xr.DataArray
        The variable before any filtering (used for missing_raw and as the
        template for dims/coords/name).
    filter_masks : dict of {str: list of xr.DataArray}
        Masks from run_filter_bank, keyed by QC flag name.
    parent_mask : xr.DataArray, optional
        Mask inherited through MASK_DEPENDENCIES, if any.

    Returns
    -------
    xr.DataArray
        uint16 flag array named ``<var>_qc_flag``.
    """
    masks = dict(filter_masks)
    masks['missing_raw'] = [raw.isnull()]
    if parent_mask is not None:
        masks['mask_dependencies'] = [parent_mask]
    return encode_flags(masks, raw)

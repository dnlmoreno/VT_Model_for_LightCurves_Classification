import pandas as pd

def min_max_normalize_lc(group, dict_columns):
    flux_min = group[dict_columns['flux']].min()
    flux_max = group[dict_columns['flux']].max()
    flux_range = flux_max - flux_min

    group[dict_columns['flux']] = (group[dict_columns['flux']] - flux_min) / flux_range
    group[dict_columns['flux_err']] = group[dict_columns['flux_err']] / flux_range

    mjd_min = group[dict_columns['mjd']].min()
    mjd_max = group[dict_columns['mjd']].max()
    mjd_range = mjd_max - mjd_min

    group[dict_columns['mjd']] = (group[dict_columns['mjd']] - mjd_min) / mjd_range

    return group


def get_normalization(group, norm_name, dict_columns):
    if norm_name == 'minmax_by_obj':
        return min_max_normalize_lc(group, dict_columns).reset_index(drop=True)
    else:
        raise 'The selected normalization has not been implemented...'
        
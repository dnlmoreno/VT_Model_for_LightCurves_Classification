import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

from PIL import Image

def create_overlay_images(obj_df, config, dataset_config, name_dataset):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for band_key, j in dataset_config['all_bands'].items():
        band_data = obj_df[obj_df[dict_columns['band']] == j]

        if band_data.empty:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=ax.transAxes))
        else:
            ax.errorbar(band_data[dict_columns['mjd']], 
                        band_data[dict_columns['flux']], 
                        yerr=band_data[dict_columns['flux_err']] if config['imgs_params']['use_err'] else None,
                        color=fig_params['colors'][j] if name_dataset == 'elasticc_1' else fig_params['colors'][j+2],
                        fmt=fig_params['fmt'], 
                        alpha=fig_params['alpha'], 
                        markersize=fig_params['markersize'], 
                        linewidth=fig_params['linewidth'])

            #ax.set_xlim(fig_params['xlim'])

        ax.set_ylim(fig_params['ylim'])
        ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image


def create_2grid_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

    fig, axs = plt.subplots(2, 1, figsize=(fig_params['figsize']))  # Dos filas y tres columnas
    for band_key, j in dataset_config['all_bands'].items():
        #row, col = divmod(j, 2)
        row = j
        band_data = obj_df[obj_df[dict_columns['band']] == j]

        if band_data.empty:
            axs[row].add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=axs[row].transAxes))
        else:
            axs[row].errorbar(band_data[dict_columns['mjd']], 
                              band_data[dict_columns['flux']], 
                              yerr=band_data[dict_columns['flux_err']] if config['imgs_params']['use_err'] else None,
                              color=fig_params['colors'][j+2],
                              fmt=fig_params['fmt'], 
                              alpha=fig_params['alpha'], 
                              markersize=fig_params['markersize'], 
                              linewidth=fig_params['linewidth'])

            #axs[row].set_xlim(fig_params['xlim'])

        axs[row].set_ylim(fig_params['ylim'])
        axs[row].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Cuadrado grande (borde exterior)
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)

    # Línea entre las filas
    rect = patches.Rectangle((0, 0.5), 1, 0, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image


def create_6grid_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

    fig, axs = plt.subplots(2, 3, figsize=(fig_params['figsize']))  # Dos filas y tres columnas
    for band_key, j in dataset_config['all_bands'].items():
        row, col = divmod(j, 3)
        band_data = obj_df[obj_df[dict_columns['band']] == j]

        if band_data.empty:
            axs[row, col].add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=axs[row, col].transAxes))
        else:
            axs[row, col].errorbar(band_data[dict_columns['mjd']], 
                                   band_data[dict_columns['flux']], 
                                   yerr=band_data[dict_columns['flux_err']] if config['imgs_params']['use_err'] else None,
                                   color=fig_params['colors'][j],
                                   fmt=fig_params['fmt'], 
                                   alpha=fig_params['alpha'], 
                                   markersize=fig_params['markersize'], 
                                   linewidth=fig_params['linewidth'])

            axs[row, col].set_xlim(fig_params['xlim'])

        axs[row, col].set_ylim(fig_params['ylim'])
        axs[row, col].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Agregar rectángulos para las columnas
    for col in range(3):
        rect = patches.Rectangle((col/3, 0), 1/3, 1, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
        fig.add_artist(rect)

    # Agregar rectángulos para las filas
    for row in range(2):
        rect = patches.Rectangle((0, row/2), 1, 0.5, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
        fig.add_artist(rect)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image
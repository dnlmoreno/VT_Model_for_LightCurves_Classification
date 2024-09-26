import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import io
import matplotlib.patches as patches


def create_padding_image(fig_params):
    plt.figure(figsize=fig_params['figsize'], dpi=100)
    plt.axis('off')
    plt.ylim(fig_params['ylim'])
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    buf.close()
    return image


def create_single_png_image(data, normalization, fig_params):
    plt.figure(figsize=fig_params['figsize'], dpi=100)
    plt.errorbar(data[f"mjd_{normalization['norm_name']}"], 
                 data[f"flux_{normalization['norm_name']}"],
                 yerr=data[f"flux_err_{normalization['norm_name']}"] if fig_params['use_err'] else None,
                 color=fig_params['color'], 
                 fmt=fig_params['fmt'], 
                 alpha=fig_params['alpha'], 
                 markersize=fig_params['markersize'])
    plt.axis('off')
    plt.ylim(fig_params['ylim'])
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    plt.close()
    buf.close()
    return image    


def create_multiBand_grid(df_lc, dict_color_bands, dict_columns):
    fig, axs = plt.subplots(2, 3, figsize=(2.24, 2.24))  # Dos filas y tres columnas
    for j, band_key in dict_bands.items():
        row, col = divmod(j, 3)
        band_data = obj_df[obj_df['passband'] == j]

        if band_data.empty:
            axs[row, col].add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=axs[row, col].transAxes))
        else:
            axs[row, col].errorbar(band_data[dict_columns['mjd']], 
                                    band_data[dict_columns['flux']], 
                                    yerr=band_data[dict_columns['flux_err']] if dict_columns.get('flux_err') else None, 
                                    color=colors[j],
                                    fmt='o-', alpha=0.5, markersize=1, linewidth=0.8)
    
        if ylim is not None:
            axs[row, col].set_ylim(ylim)

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

    # Convert the figure to a PNG image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')

    return image

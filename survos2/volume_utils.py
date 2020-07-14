

import time
import numpy as np

import logging

import matplotlib
from matplotlib import colors, pyplot as plt
from matplotlib.widgets import Slider

from skimage.segmentation import find_boundaries


def _bmap(region):
    mask = find_boundaries(region)
    return np.ma.masked_where(~mask, mask)


def view(data, boundaries=None, overlay=None,
         bcolor='#000099', balpha=0.7, oalpha=0.5):
    is_3d = data.ndim >= 3 and data.shape[2] > 3
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    idx = slice(None) if not is_3d else data.shape[0]//2

    im = ax.imshow(data[idx], 'gray')
    bb = None
    ov = None

    if boundaries:
        bound = _bmap(boundaries[idx])
        cmap = colors.ListedColormap(['#000000', bcolor])
        bb = ax.imshow(bound, cmap=cmap, vmin=0, vmax=1, alpha=balpha)

    if overlay:
        ov = ax.imshow(overlay[idx], cmap='viridis', alpha=oalpha)

    if is_3d:
        size, margin = 0.03, 0.05
        axslider = plt.axes([0.01, 1-size-0.01, 0.98, size])
        plt.subplots_adjust(left=margin, bottom=margin, right=1-margin, top=1-margin)
        slider = Slider(axslider, 'Slice:', 0, data.shape[0] - 1,
                        valinit=idx, valfmt='%d')
        slider.valtext.set_x(0.5)
        slider.valtext.set_horizontalalignment('center')

        def update(idx):
            idx = min(int(idx), data.shape[0] - 1)
            if idx != int(slider.val):
                return slider.set_val(idx)
            im.set_data(data[idx])
            if bb:
                bb.set_data(_bmap(boundaries[idx]))
            if ov:
                ov.set_data(overlay[idx])
            fig.canvas.draw_idle()

        slider.on_changed(update)

    plt.show()



def view3d(data, margin=0.05, size=0.03):
    idx = data.shape[0] // 2
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    im = ax.imshow(data[idx], 'gray')

    def update(idx):
        idx = min(int(round(idx)), data.shape[0] - 1)
        im.set_data(data[idx])
        fig.canvas.draw_idle()

    axslider = plt.axes([0.01, 1-size-0.01, 0.98, size])
    plt.subplots_adjust(left=margin, bottom=margin, right=1-margin, top=1-margin)
    slider = Slider(axslider, 'Freq', 0, data.shape[0] - 1, valinit=idx)
    slider.on_changed(update)

    plt.show()


def compare3d(data1, data2, margin=0.03, size=0.03):
    assert data1.shape == data2.shape

    idx = data1.shape[0] // 2
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    im1 = axes[0].imshow(data1[idx], 'gray')
    im2 = axes[1].imshow(data2[idx], 'gray')

    def update(idx):
        idx = min(int(round(idx)), data1.shape[0] - 1)
        im1.set_data(data1[idx])
        im2.set_data(data2[idx])
        fig.canvas.draw_idle()

    axslider = plt.axes([0.01, 1-size-0.01, 0.98, size])
    plt.subplots_adjust(left=margin, bottom=margin, right=1-margin, top=1-margin)
    slider = Slider(axslider, 'Freq', 0, data1.shape[0] - 1, valinit=idx)
    slider.on_changed(update)

    plt.show()


def compare_diff3d(data1, data2, margin=0.03, size=0.03):
    assert data1.shape == data2.shape

    idx = data1.shape[0] // 2
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
    im1 = axes[0].imshow(data1[idx], 'gray')
    im2 = axes[1].imshow(data2[idx], 'gray')
    diff = data2[idx] - data1[idx]
    im3 = axes[2].imshow(diff, 'gray')
    axes[2].set_title('Min: %.7f, Max: %.7f' % (diff.min(), diff.max()))

    def update(idx):
        idx = min(int(round(idx)), data1.shape[0] - 1)
        im1.set_data(data1[idx])
        im2.set_data(data2[idx])
        diff = data2[idx] - data1[idx]
        im3.set_data(diff)
        axes[2].set_title('Min: %.7f, Max: %.7f' % (diff.min(), diff.max()))
        fig.canvas.draw_idle()

    axslider = plt.axes([0.01, 1-size-0.01, 0.98, size])
    plt.subplots_adjust(left=margin, bottom=margin, right=1-margin, top=1-margin)
    slider = Slider(axslider, 'Freq', 0, data1.shape[0] - 1, valinit=idx)
    slider.on_changed(update)

    plt.show()


##############################################################################
# Regions
##############################################################################

def compare3d_regions(data, reg1, reg2, margin=0.03, size=0.03):
    assert reg1.shape == reg2.shape

    idx = data.shape[0] // 2
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    im1 = axes[0].imshow(data[idx], 'gray')
    bb1 = axes[0].imshow(_bmap(reg1[idx]), 'viridis', alpha=0.5)
    im2 = axes[1].imshow(data[idx], 'gray')
    bb2 = axes[1].imshow(_bmap(reg2[idx]), 'viridis', alpha=0.5)

    def update(idx):
        idx = min(int(round(idx)), data.shape[0] - 1)
        im1.set_data(data[idx])
        im2.set_data(data[idx])
        bb1.set_data(_bmap(reg1[idx]))
        bb2.set_data(_bmap(reg2[idx]))
        fig.canvas.draw_idle()

    axslider = plt.axes([0.01, 1-size-0.01, 0.98, size])
    plt.subplots_adjust(left=margin, bottom=margin, right=1-margin, top=1-margin)
    slider = Slider(axslider, 'Freq', 0, data.shape[0] - 1, valinit=idx)
    slider.on_changed(update)

    plt.show()
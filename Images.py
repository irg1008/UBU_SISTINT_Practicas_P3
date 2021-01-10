import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from glob import glob
from typing import List, Tuple


# Typing.
GrayImg = List[List[float]]
ImgList = List[GrayImg]
Title = str
Titles = List[Title]


def print_img(img: GrayImg, width=10, title=""):
    """
    Prints an image using the matplot library.

    Args:
        img (GrayImg): Image to print.
        width (int, optional): Width of image. Defaults to 10.
        title (str, optional): Title of plot. Defaults to "".
    """
    img_ratio = float(img.shape[1]/img.shape[0])
    gray = len(img.shape) == 2

    plt.figure(figsize=(width, width/img_ratio))
    plt.imshow(img, cmap=plt.get_cmap("gray")) if gray else plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()


def print_n_imgs(imgs: ImgList, titles: Titles, title: Title, width=10, columnMode=False):
    """
    Prints all images in array.

    Args:
        imgs (ImgList): Images to print.
        titles (Titles): Titles for image plots.
        title (Title): Title of general plot.
        width (int, optional): Width of individual image. Defaults to 10.
        columnMode (bool, optional): Set to true images are displayed vertically. Defaults to False.
    """
    n_imgs = len(imgs)
    gray = [len(img.shape) == 2 for img in imgs]

    ncols = n_imgs if not columnMode else 1
    nrows = n_imgs if columnMode else 1

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                           figsize=(n_imgs*width, width))
    fig.suptitle(title, fontsize=width*2)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    for i in range(n_imgs):
        ax[i].imshow(
            imgs[i], cmap=plt.cm.gray) if gray[i] else ax[i].imshow(imgs[i])
        ax[i].set_title(titles[i])
        ax[i].axis("off")


def print_zoomed_imgs(imgs: ImgList, titles: Titles, title: Title, width=10):
    """
    Prints all images zoomed in the middle.

    Args:
        imgs (ImgList): Images to print.
        titles (Titles): Titles of images.
        title (Title): Title of general plot.
        width (int, optional): Width of individual image. Defaults to 10.
    """
    zoommed_imgs = []
    for img in imgs:
        w, h = img.shape[1], img.shape[0]
        zoommed_imgs.append(img[w//2-w//4:w//2+w//4, h//2-h//4:h//2+h//4])
    print_n_imgs(zoommed_imgs, titles, title, width)


def print_histogram(imgs, labels, title, width=6):
    """
    Prints image histogram with pixel value and count.

    Args:
        imgs ([type]): Images to show histogram of.
        labels ([type]): Labels of images, set to id the image.
        title ([type]): Title of general plot.
        width (int, optional): Width of individual images. Defaults to 10.
    """
    _colors = ["tomato", "aqua", "green",
               "silver", "lightgreen", "orange", "violet"]
    flatten = [img.flatten() for img in imgs]
    colors = [_colors[i % len(_colors)] for i in range(len(imgs))]

    bins = 256 * len(imgs)
    plt.figure(figsize=(width, width*9/16))
    plt.hist(x=flatten, label=labels, bins=bins,
            color=colors, edgecolor="black", linewidth=0.)

    plt.legend(loc="upper right")
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Pixel Count")
    plt.tight_layout()
    plt.show()

def print_hist_combs(imgs: ImgList, titles: Titles, title: Title, combStart=2):
    """
    Prints images combinations histograms.

    Args:
        imgs (ImgList): Images to print.
        titles (Titles): Titles of images.
        title (Title): Title of all hists.
    """
    combs = []
    labels = []
    for i in range(combStart, len(imgs)):
        combs.append((imgs[0], imgs[i]))
        labels.append((titles[0], titles[i]))
    
    for comb, label in zip(combs, labels):
        print_histogram(imgs=comb, labels=label, title=title)
        sim = img_similarity(comb[0], comb[1])
        perc_sim = "{:.2%}".format(sim)
        print(f"Image similarity: {perc_sim}")

def get_images(path: str) -> ImgList:
    """
    Gets all images in given path.

    Args:
        path (str): Path where images are stored.

    Returns:
        ImgList: Images found on given path. 
    """
    imgs = [io.imread(img) for img in glob(path)]
    return imgs


def img_similarity(imgA: GrayImg, imgB: GrayImg, th=2) -> float:
    """
    Returns percentage (0-1) of similarity between two images.
    Based on number of very, almost similar pixels.

    Args:
        imgA (GrayImg): First image.
        imgB (GrayImg): Second image.
        th (float): Threshold of pixel variance.

    Returns:
        float: Similarity value.
    """
    sim = np.subtract(imgA, imgB)
    sim = np.abs(sim)
    sim = np.trunc(sim)
    sim = np.sum(sim <= th)
    sim /= np.prod(imgA.shape)
    return sim

"""
This script performs post-segmentation processing on a set of images.
It includes functions for thresholding, skeletonization, and masking.

The script reads a folder of input images, applies thresholding to the images,
skeletonizes the thresholded images, and masks the original images using the
thresholded images. The processed images are then saved to an output folder.

The script can be run in sequential mode or parallel mode, depending on the
value of the `processing_key` variable. In sequential mode, the script processes
the images one by one. In parallel mode, the script uses multiprocessing to
process multiple images simultaneously.

The script also provides functions for reading a TIFF stack, getting the size
of a stack, and performing connected component analysis.

Author: Kentaro Ishii
Date: 5/17/2024
"""

# Rest of the code...
# This is meant to be used on TRAILMAP container
# This is a wrapper around TRAILMAP https://github.com/albert597/TRAILMAP
# TRAILMAP uses tensorflow as to interact with GPU
# Writing in python3.6 tifffile, so use compress = 5 instead of compression

import multiprocessing
import logging
from functools import partial

import os
import cc3d
#import tensorflow as tf
import numpy as np
import tifffile
import time
from PIL import Image
import scipy
from skimage.morphology import skeletonize, ball

def thresholding_tf(input_folder, outputfolder, threshold=0.5, multi_thresholds=None):
    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith('.tif'):
            continue
        # Read image
        img = tifffile.imread(os.path.join(input_folder, filename))

        # Thresholding
        if multi_thresholds is None:
            dimg = tf.cast(img > threshold, tf.uint8)
        else:
            dimg = tf.zeros_like(img, dtype=tf.uint8)
            for t in multi_thresholds:
                dimg += tf.cast(img > t, tf.uint8) * int(t * 10)

        # Write result
        tifffile.imwrite(os.path.join(outputfolder, filename), dimg.numpy(),compress = 5)

def thresholding(input_folder, outputfolder, threshold=0.5, multi_thresholds=None):
    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith('.tif'):
            continue
        # Read image
        img = tifffile.imread(os.path.join(input_folder, filename))

        # Thresholding
        if multi_thresholds is None:
            dimg = (img > threshold).astype(np.uint8) * 255
        else:
            dimg = np.zeros_like(img, dtype=np.uint8)
            for t in multi_thresholds:
                dimg += (img > t).astype(np.uint8) * int(t * 10)

        # Write result
        tifffile.imwrite(os.path.join(outputfolder, filename), dimg, compress = 5)

def thresholding_parallel(input_file,thresholding_parameters):
    outputfolder = thresholding_parameters['outputfolder']
    threshold = thresholding_parameters['threshold']
    multi_thresholds = thresholding_parameters['multi_thresholds']
    
    
    filename = os.path.basename(input_file)
    if not filename.endswith('.tif'):
        raise ValueError('Not a Tif file...')

    # Read image
    img = tifffile.imread(input_file)

    # Thresholding
    if multi_thresholds is None:
        dimg = (img > threshold).astype(np.uint8) * 255
    else:
        dimg = np.zeros_like(img, dtype=np.uint8)
        for t in multi_thresholds:
            dimg += (img > t).astype(np.uint8) * int(t * 10)

    # Write result
    tifffile.imwrite(os.path.join(outputfolder, filename), dimg, compress = 5)


def read_tiff_stack(path, zslice=None):
    if os.path.isdir(path):
        if zslice is None:
            images = [np.array(Image.open(os.path.join(path, p))) for p in sorted(os.listdir(path)) if p.endswith('.tif')]
        else:
            images = [np.array(Image.open(os.path.join(path, p))) for p in sorted(os.listdir(path))[zslice] if p.endswith('.tif')]
        return np.array(images)
    else:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
        if zslice is None:
            return np.array(images)
        else:
            return np.array(images[zslice])

def get_stack_size(path):
    if os.path.isdir(path):
        imglist = [f for f in os.listdir(path) if f.endswith('.tif')]
        z = len(imglist)
        timg = tifffile.imread(os.path.join(path, imglist[0]))
        y, x = timg.shape[0], timg.shape[1]
        return z, y, x
    else:
        timg = tifffile.imread(path)
        return timg.shape    

def thinned_component(base, vol, zwindow=250, zslice=None):    
    stack = read_tiff_stack(outputfolder, zslice=zslice)
    #plt.imshow(np.max(stack, axis = 0))
    #print("Data shape", stack.shape)
    #print("Data max",np.max(stack))
    # Create labels for connected components
    labels = cc3d.connected_components(stack)
    # Identify unique labels and count
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Remove labels that are under x connectivitiy
    labels_to_remove = unique_labels[label_counts < vol]
    #mask = np.isin(labels, labels_to_remove)
    labels[np.isin(labels, labels_to_remove)] = 0
    labels[labels>0] = 255
    labels = labels.astype('uint8')
    
    return labels


def skeleton_batch(base, target, connectives=10, zslice=None,ballsize = 1):
    # create a file for the out put file
    if not os.path.exists(target):
        os.mkdir(target)
    
    # get the file names
    if zslice is None:
        fnames = [p for p in sorted(os.listdir(base)) if p.endswith('.tif')]
    else:
        fnames = [p for p in sorted(os.listdir(base))[zslice] if p.endswith('.tif')]


    # Clear junks，removing 3D connectives with voxel number 10 by default.
    labels = thinned_component(base, connectives, zslice=zslice)
    print("Data will be saved in", target)
    # Do skeleton
    if not labels.dtype == 'uint8':
        #print('yes')
        labels = labels.astype('uint8')
    skeleton = skeletonize(labels)

    # Do dilation
    # specify the size of the ball
    skeleton = scipy.ndimage.binary_dilation(skeleton, ball(ballsize))
    for i,fname in zip(skeleton,fnames):
        tifffile.imsave(os.path.join(target,fname), i.astype(np.uint8),compress = 5)

    print('Done.')

# Parallel processing
def skeleton_batch_parallel(zslice, batch_parameters):
    """
    Perform batch processing of skeletonization on a given zslice.

    Args:
        zslice (int): The zslice to process.
        batch_parameters (dict): A dictionary containing the batch parameters.
            - target (str): The target directory to save the output files.
            - connectives (int): The number of connectives to remove.
            - base (str): The base directory containing the input files.
            - ballsize (int): The size of the ball for dilation.

    Returns:
        None
    """
    # Setup parameters
    target      = batch_parameters['target'] + '_skeleton'
    connectives = batch_parameters['connectives']
    base        = batch_parameters['base']
    ballsize    = batch_parameters['ballsize']

    # create a file for the out put file
    if not os.path.exists(target):
        os.mkdir(target)

    # Clear junks，removing 3D connectives with voxel number 10 by default.
    labels = thinned_component(base, connectives, zslice=zslice)
    print("Data will be saved in", target)

    # Do skeleton
    if not labels.dtype == 'uint8':
        labels = labels.astype('uint8')
    skeleton = skeletonize(labels)

    # Do dilation
    skeleton = scipy.ndimage.binary_dilation(skeleton, ball(ballsize))

    # get the file names
    if zslice is None:
        fnames = [p for p in sorted(os.listdir(base)) if p.endswith('.tif')]
    else:
        fnames = [p for p in sorted(os.listdir(base))[zslice] if p.endswith('.tif')]

    # save the results
    for i,fname in zip(skeleton,fnames):
        tifffile.imsave(os.path.join(target,fname), i.astype(np.uint8),compress = 5)

    print('Done.')

def write_mask_image(idx,mask_parameters):
    rawfolder = mask_parameters['rawfolder']
    outputfolder = mask_parameters['outputfolder']
    maskfolder = mask_parameters['maskfolder']
    
    rawfpath = os.path.join(rawfolder,os.listdir(rawfolder)[idx])
    outputfpath = os.path.join(outputfolder,os.listdir(rawfolder)[idx])
    maskfpath = os.path.join(maskfolder,os.listdir(maskfolder)[idx])

    rawimg = tifffile.imread(rawfpath)
    maskimg = tifffile.imread(maskfpath)

    # mask the results
    rawimg[maskimg==0] = 0
    tifffile.imsave(outputfpath, rawimg.astype(np.uint8),compress = 5)

def collect_intensity(rawfolder,maskfolder,parallel = True):
    outputfolder = rawfolder + '_masked'
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    mask_parameters = {}
    mask_parameters['rawfolder'] = rawfolder
    mask_parameters['outputfolder'] = outputfolder
    mask_parameters['maskfolder'] = maskfolder

    indexes = range(len([f for f in os.listdir(rawfolder) if '.tif' in f]))

    # maskfolder: The folder which contains thresholded images of axons
    if parallel:
        logging.debug("Starting masking")
        num_cores = multiprocessing.cpu_count()
        print("num_cores: %d" % num_cores)

        with multiprocessing.Pool(processes=num_cores) as pool:  # Added `with` statement
            # Use functools.partial to pass the constant variable to my_function
            partial_function = partial(write_mask_image, mask_parameters=mask_parameters)
            resample_imgs = pool.map(partial_function, indexes)
    else:
        for idx in indexes:
            write_mask_image(idx,mask_parameters)

# Run 
if __name__ == "__main__":
    start = time.time()

    #print("Hello, World!")
    # input folder
    #imgfolder = input("Where is the probability images...")
    rootimgfolder = r'\\10.158.246.229\DataCommon\SmartSPIM2\Ken\NAc_PRJ'
    for imgfname in [f for f in os.listdir(rootimgfolder) if '_DONE' in f]:
        findex = int(imgfname.split('_')[6])
        print(findex)
        if findex == 2:
            continue
        if findex % 2 == 1:
            fname = 'seg-Ex_488_Ch0_stitched_Right'
        elif findex % 2 == 0:
            fname = 'seg-Ex_488_Ch0_stitched_Left'
        imgfolder = os.path.join(rootimgfolder,imgfname,fname)
        #fname = os.path.basename(imgfolder)
        outputfolder = imgfolder.replace(fname,fname + '_thresholded')
        # sequential
        processing_key = 'Sequential' # or 'Parallel' or 'Sequential'
        zwindow = 50
        #threshold = 0.5
        threshold = None
        multi_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        print("Thresholding the axon probability");
        print("Processing images in",outputfolder);
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

            # save the thresholded images
        if processing_key =='Parallel':
            thresholding_parameters = {}
            thresholding_parameters['outputfolder'] = outputfolder
            thresholding_parameters['threshold'] = threshold
            thresholding_parameters['multi_thresholds'] = multi_thresholds

            logging.debug("Starting thresholding")
            num_cores = multiprocessing.cpu_count()
            print("num_cores: %d" % num_cores)
            imagefiles = [os.path.join(imgfolder,p) for p in sorted(os.listdir(imgfolder)) if p.endswith('.tif')]
            with multiprocessing.Pool(processes=num_cores) as pool:  # Added `with` statement
                # Use functools.partial to pass the constant variable to my_function
                partial_function = partial(thresholding_parallel, thresholding_parameters=thresholding_parameters)
                resample_imgs = pool.map(partial_function, imagefiles)

        else:
            thresholding(imgfolder,outputfolder,multi_thresholds = multi_thresholds)

        print("skeletonizing the axon")

        imglist = [os.path.join(outputfolder,f) for f in sorted(os.listdir(outputfolder)) if '.tif' in f]
        z = len(imglist)
        print(imgfolder.replace(fname,fname + f'_skeleton.tif'))

        if processing_key == 'Sequential':
            for idx,zstart in enumerate(np.arange(0,z,step = zwindow)):
                zslice = slice(zstart,zstart+zwindow)
                print("Start processing from",zstart)
                #img = read_tiff_stack(outputfolder,zslice = zslice)
                skeleton_batch(outputfolder, imgfolder.replace(fname,fname + '_skeleton'),
                connectives=10,zslice = zslice)
        elif processing_key == 'Parallel':
            batch_parameters = {}
            batch_parameters['target'] = imgfolder
            batch_parameters['connectives'] = 10
            batch_parameters['base'] = outputfolder
            batch_parameters['ballsize'] = 1

            zslices = []
            for idx,zstart in enumerate(np.arange(0,z,step = zwindow)):
                zslice = slice(zstart,zstart+zwindow)
                zslices.append(zslice)
            
            # Skeletonizing core function
            logging.debug("Starting Skeletonizing")
            num_cores = multiprocessing.cpu_count()
            print("num_cores: %d" % num_cores)
            with multiprocessing.Pool(processes=num_cores) as pool:  # Added `with` statement
                # Use functools.partial to pass the constant variable to my_function
                partial_function = partial(skeleton_batch_parallel, batch_parameters=batch_parameters)

                resample_imgs = pool.map(partial_function, zslices)
        else:
            skeleton_batch(outputfolder, imgfolder.replace(fname,fname + f'_skeleton') ,
                connectives=5,)
        end = time.time()
        print("The entire process ended in ",end - start)#
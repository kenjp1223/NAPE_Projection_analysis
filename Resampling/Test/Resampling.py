import multiprocessing
import logging
import time
from random import random
import ants
import numpy as np
import os
import tifffile
import cv2
from functools import partial
import shutil

#basepath = r"\\10.159.50.7\analysis2\Ken\LSMS\LS_NTS\LS_NTS_F3B\Ex_639_Em_690_stitched"
# orientation
#orientation = 'ALS'
#original_orientation = 'RAS'
#target_orientation = 'RSP'

#target_orientation = (1,-3,-2) # swap y and z, invert y and z axis


# outputpath for oriented image
#resampled_outputpath = r"\\10.159.50.7\Analysis2\Ken\LSMS\LS_NTS\LS_NTS_F3B\\" +  "resampled_auto_ants.tif"
#oriented_outputpath = r"\\10.159.50.7\Analysis2\Ken\LSMS\LS_NTS\LS_NTS_F3B\\" +  "auto_ants.tif"
# resolutions
# original orientation
#raw_resolution = (1.8,1.8,4.0)
#target_resolution = (20,20,50)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(process)s %(levelname)s %(message)s',
    filename='resampling.log',
    filemode='a'
)


#tmp_folder = 'tmp_ants'

def orientation_parameters(target_orientation):
    global flip_axis,swap_axis
    # extract how to swap the data
    target_orientation = np.array(target_orientation)[::-1]
    flip_axis = np.where(target_orientation<0)[0]
    swap_axis = np.array([2,2,2]) - (abs(target_orientation)-1)
    
    #return flip_axis,swap_axis

def downsample_parameters(raw_resolution,target_resolution,target_orientation):
    global dsrates
    target_orientation = np.array(target_orientation)[::-1]
    # calculate the downsample rate
    dsrates = [raw_resolution[idx]/target_resolution[(abs(target_orientation)-1)[idx]] for idx in range(3)]
    
    #return dsrates



#def process_XY(**kwargs, fpath):
def process_XY(fpath,sample_parameters):
    # extract necessary variables
    target_orientation = sample_parameters['target_orientation']
    raw_resolution = sample_parameters['raw_resolution']
    target_resolution = sample_parameters['target_resolution']

    # get basename
    fname = os.path.basename(fpath)
    
    with multiprocessing.Lock():
        logging.debug("Resampling XY %s" % fname)

    if not '.tif' in fpath:
        return None



    # read the image file
    timg = tifffile.imread(fpath)

    # calculate the new orientation
    orientation_parameters(target_orientation)
    # calculate the downsample rate
    downsample_parameters(raw_resolution,target_resolution,target_orientation)
    #global dsrates,flip_axis,swap_axis
    # resize the image 
    new_size = (np.round(timg.shape[1] * dsrates[0]).astype('int'), np.round(timg.shape[0] * dsrates[1]).astype('int'))
    resizeimage = cv2.resize(timg, new_size, interpolation=cv2.INTER_CUBIC)


    # logging
    with multiprocessing.Lock():
        logging.debug("Finished resample XY image with %s." % (fname))

    # Create temporary folder if it doesn't exist
    #if not os.path.exists(tmp_folder):
    #    os.makedirs(tmp_folder)

    # Save the processed image to temporary folder
    #result_path = os.path.join(tmp_folder, fname)
    #tifffile.imwrite(result_path,resizeimage)

    return resizeimage

def process_Z(resample_img,sample_parameters):
    # extract necessary variables
    target_orientation = sample_parameters['target_orientation']
    raw_resolution = sample_parameters['raw_resolution']
    target_resolution = sample_parameters['target_resolution']
    resampled_outputpath = sample_parameters['outputpath']

    # calculate the new orientation
    orientation_parameters(target_orientation)
    # calculate the downsample rate
    downsample_parameters(raw_resolution,target_resolution,target_orientation)

    # read the images
    #resample_img = tifffile.imread([os.path.join(tmp_folder,f) for f in sorted(os.listdir(tmp_folder)) if '.tif' in f])
    # resize the image
    new_size = (np.round(resample_img.shape[0] * dsrates[2]).astype('int'), 
    resample_img.shape[1],
    resample_img.shape[2])

    # orient the image
    resample_img = np.flip(resample_img,axis = flip_axis)
    resample_img = resample_img.transpose(swap_axis)
     
    resizeimage2 = []
    for f in resample_img:
        f = cv2.resize(f, np.array(new_size)[swap_axis][1:3][::-1], interpolation=cv2.INTER_LINEAR)
        resizeimage2.append(f)
    resizeimage2 = np.array(resizeimage2)

    tifffile.imwrite(resampled_outputpath,resizeimage2,)
    #return resizeimage2




# For debugging Logging functions
'''def simulation(a):
    # logging
    with multiprocessing.Lock():
        logging.debug(" with %s" % a)

    # simulation
    time.sleep(random())
    result = a * 2

    # logging
    with multiprocessing.Lock():
        logging.debug("Finished simulation with %s. Result is %s" % (a, result))

    return result'''

if __name__ == '__main__':

    # outputpath for oriented image
    resampled_outputpath = r"\\10.159.50.7\Analysis2\Ken\LSMS\LS_NTS\LS_NTS_F3B\\" +  "resampled_intensity_HRimage.tif"

    # The path to the images that requires resampling
    basepath = r"\\10.159.50.7\analysis2\Ken\LSMS\LS_NTS\LS_NTS_F3B\Ex_488_Em_525_stitched_intensity"
    sample_parameters = {}
    sample_parameters['target_orientation'] = (1,-3,-2)
    sample_parameters['raw_resolution']  = (1.8,1.8,4.0)
    sample_parameters['target_resolution'] = (5,5,50)
    sample_parameters['outputpath'] = resampled_outputpath
    
    # Get list of files to process
    file_list = [os.path.join(basepath,f) for f in sorted(os.listdir(basepath)) if '.tif' in f]
    print(len(file_list)) #for debug
    # extract the orientation parameters
    #orientation_parameters(target_orientation)
    #downsample_parameters(raw_resolution,target_resolution,target_orientation)

    # resampling core function
    logging.debug("Starting XY resampling")
    num_cores = multiprocessing.cpu_count()
    print("num_cores: %d" % num_cores)
    with multiprocessing.Pool(processes=num_cores) as pool:  # Added `with` statement
        # Use functools.partial to pass the constant variable to my_function
        partial_function = partial(process_XY, sample_parameters=sample_parameters)

        resample_imgs = pool.map(partial_function, file_list)
    #print(resample_imgs)
    resample_imgs = np.array(resample_imgs)
    #tifffile.imsave(resampled_outputpath.replace('.tif','_test.tif'),np.array(resample_imgs))
    logging.debug("The XY resampling has ended")


    logging.debug("Starting Z resampling")
    process_Z(resample_imgs,sample_parameters)
    logging.debug("Finished resampling")  
    #os.rmdir(tmp_folder)
    #shutil.rmtree(tmp_folder, ignore_errors=True)
    #resample_img = np.array([f[:,:] for f in resample_imgs])
    
# This is meant to be used on TRAILMAP container
# This is a wrapper around TRAILMAP https://github.com/albert597/TRAILMAP
# TRAILMAP uses tensorflow as to interact with GPU

import os
import cc3d
import tensorflow as tf
import numpy as np
import tifffile
from PIL import Image
from skimage.morphology import skeletonize

def thresholding(input_folder, output_folder, threshold=0.5, multi_thresholds=None):
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
        tifffile.imwrite(os.path.join(output_folder, filename), dimg.numpy())

def read_tiff_stack(path, zslice=None):
    if os.path.isdir(path):
        images = [np.array(Image.open(os.path.join(path, p))) for p in sorted(os.listdir(path))[zslice] if p.endswith('.tif')]
        return np.array(images)
    else:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
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
    stack = read_tiff_stack(base, zslice=zslice)
    print("Data shape", stack.shape)
    labels = cc3d.connected_components(stack)

    shapes = labels.shape
    labels = labels.flatten()
    mark = [0] * (labels.max() + 1)
    for i in labels:
        mark[i] += 1
    for k in range(len(labels)):
        if mark[labels[k]] < vol:
            labels[k] = 0
    labels[labels > 0] = 255

    return labels.reshape(shapes)

def skeleton_batch(base, target, device, connectives=10, zslice=None):
    # Clear junks，removing 3D connectives with voxel number 10 by default.
    labels = thinned_component(base, connectives, zslice=zslice)
    print("Data will be saved in", target)
    # Do skeleton
    skeleton = skeletonize(labels)
    
    # Do dilation
    x = tf.constant(skeleton.reshape(1, *skeleton.shape, 1), dtype=tf.float32)
    p1 = tf.nn.max_pool3d(x, (1, 3, 3, 1), (1, 1, 1, 1), 'VALID')
    p2 = tf.nn.max_pool3d(x, (3, 1, 3, 1), (1, 1, 1, 1), 'VALID')
    p3 = tf.nn.max_pool3d(x, (3, 3, 1, 1), (1, 1, 1, 1), 'VALID')
    min_pool_x = tf.math.reduce_min(tf.concat([p1, p2, p3], axis=-1), axis=-1)
    x = tf.math.reduce_max(tf.concat([p1, p2, p3], axis=-1), axis=-1)
    x = tf.where(x > 0, 255, 0)

    tifffile.imsave(target, x.numpy().astype(np.uint8))
    print('Done.')



# Run 
if __name__ == "__main__":
    start = time.time()

    #print("Hello, World!")
    # input folder
    #imgfolder = input("Where is the probability images...")
    imgfolder = r'/mmfs1/gscratch/scrubbed/ken1223/Example_projection_crop/seg-Ex_488_Em_525_stitched'
    fname = os.path.basename(imgfolder)
    outputfolder = imgfolder.replace(fname,fname + '_thresholded')
    zwindow = 50
    #threshold = 0.5
    multi_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("GPU available?",torch.cuda.is_available());
    print("Thresholding the axon probability by ", threshold);
    print("Processing images in",outputfolder);
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    # save the thresholded images
    thresholding(imgfolder,outputfolder,multi_thresholds = multi_thresholds)

    print("Skeltonizing the axon")

    imglist = [os.path.join(outputfolder,f) for f in sorted(os.listdir(outputfolder)) if '.tif' in f]
    z = len(imglist)
    for idx,zstart in enumerate(np.arange(0,z,step = zwindow)):
        zslice = slice(zstart,zstart+zwindow)
        print("Start processing from",zstart)
        #img = read_tiff_stack(outputfolder,zslice = zslice)
        labels = thined_component(outputfolder, 10,zslice = zslice)
        skeleton_batch(outputfolder, imgfolder.replace(fname,fname + f'_skelton_{idx+1}.tif'), device,
        connectives=10,zslice = zslice)

    end = time.time()
    print("The entire process ended in ",end - start)#
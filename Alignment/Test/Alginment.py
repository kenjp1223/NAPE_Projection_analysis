# This alignemnt code will run in antspy container

import ants
import numpy as np
import os
import tifffile
import shutil
import time
import pandas as pd

# Set the input path for the moving and fixed images.
# The moving image will typically be the autofluorescence images
# The fixed image will be the atlas_mri image

def ants_initial_alignment(fixp,movp,outputpath,key = 'auto_to_atlas',type_of_transform = 'antsRegistrationSyNs'):
    # Read the images into ants format
    fix = ants.image_read(fixp)
    mov = ants.image_read(movp)

    # Run ANTS alignment
    #Currently this is the most accurate. It is slow
    #type_of_transform = 'ElasticSyN' # ElasticSyN is accurate and slightly faster.
    mytx = ants.registration(fixed=fix, moving=mov, type_of_transform = type_of_transform )

    # Transform moving image for inspection
    #warpedimg = ants.apply_transforms( fixed=fix, moving=mov,transformlist=mytx['fwdtransforms'] )
    #warpedimg.to_file(os.path.join(outputpath,f'{key}_fwdtransforms.tif'))

    # Reverse transform fixedi image for inspection
    #invwarpedimg = ants.apply_transforms( fixed=mov, moving=fix,transformlist=mytx['invtransforms'] )
    #invwarpedimg.to_file(os.path.join(outputpath,f'{key}_invtransforms.tif'))
    #print(mytx['fwdtransforms'])
    # Save the Transformation matrix
    transformationpath = os.path.join(outputpath,f'{key}_transformation')
    os.makedirs(transformationpath,exist_ok = True)
    # Transfer files to the new path
    for idx in range(2):
        for T in ['fwdtransforms','invtransforms']:
            names = os.path.basename(mytx[T][idx]).split('.')
            names[0] = T
            name = '.'.join(names)
            #print(mytx[T][idx],name,T)
            shutil.copy(mytx[T][idx], os.path.join(transformationpath,name))

def ants_transformation(fixp,movp,transformationpath,outputpath,fnamekey,transform_key = 'fwdtransforms'):
    # Read the images into ants format
    fix = ants.image_read(fixp)
    mov = ants.image_read(movp)

    # set up the transformations
    # change the order if it is a reverse transformation
    if 'fwd'  in transform_key:
        transforms = [  os.path.join(transformationpath,f'{transform_key}.nii.gz'),
                        os.path.join(transformationpath,f'{transform_key}.mat')]
    elif 'inv'  in transform_key:
        transforms = [  os.path.join(transformationpath,f'{transform_key}.mat'),
                        os.path.join(transformationpath,f'{transform_key}.nii.gz')]


    # apply the transformation
    transformed_img = ants.apply_transforms( fixed=fix, moving=mov,transformlist=transforms )
    transformed_img.to_file(os.path.join(outputpath,f'{fnamekey}_transformed.tif'))


if __name__ == '__main__':
    # inputs
    outputpath = r"\\10.158.246.229\DataCommon\SmartSPIM2\Ken\MS_TRAPCeA\20250707_15_18_11_MS_CeA_m1732_Destripe_DONE"
    subset_key = ''
    fname = 'autofluo_resampled' + subset_key
    sname = 'Ex_639_Ch2_stitched_resampled' + subset_key
    movp = os.path.join(outputpath,f'{fname}.tif')
    sigp = os.path.join(outputpath,f'{sname}.tif')
    fixp = r"\\10.159.50.7\Analysis2\Ken\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations\Allen_templates\average_template_10_coronal.tif"

    # outputs
    transformationpath = os.path.join(outputpath,f'auto_to_atlas{subset_key}_rotated_transformation')
    key = 'auto_to_atlas' + subset_key # Key string to label files.

    ants_initial_alignment(movp,fixp,outputpath,key,type_of_transform = 'antsRegistrationSyN[s]')
    ants_transformation(fixp,movp,transformationpath,outputpath,f'{fname}_auto_to_atlas_rotated',transform_key = 'invtransforms')
    ants_transformation(fixp,sigp,transformationpath,outputpath,f'{sname}_signal_to_atlas_rotated',transform_key = 'invtransforms')
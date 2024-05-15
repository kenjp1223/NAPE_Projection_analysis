# This alignemnt code will run in antspy container

import ants
import numpy as np
import os
import tifffile
import shutil

# Set the input path for the moving and fixed images.
# The moving image will typically be the autofluorescence images
# The fixed image will be the atlas_mri image

def ants_initial_alignment(fixp,movp,outputpath,key = 'auto_to_atlas'):
    # Read the images into ants format
    fix = ants.image_read(fixp)
    mov = ants.image_read(movp)

    # Run ANTS alignment
    type_of_transform = 'antsRegistrationSyNs' #Currently this is the most accurate. It is slow
    #type_of_transform = 'ElasticSyN' # ElasticSyN is accurate and slightly faster.
    mytx = ants.registration(fixed=fix, moving=mov, type_of_transform = type_of_transform )

    # Transform moving image for inspection
    warpedimg = ants.apply_transforms( fixed=fix, moving=mov,transformlist=mytx['fwdtransforms'] )
    warpedimg.to_file(os.path.join(outputpath,f'{key}_fwdtransforms.tif'))

    # Reverse transform fixedi image for inspection
    invwarpedimg = ants.apply_transforms( fixed=mov, moving=fix,transformlist=mytx['invtransforms'] )
    invwarpedimg.to_file(os.path.join(outputpath,f'{key}_invtransforms.tif'))

    # Save the Transformation matrix
    transformationpath = os.path.join(outputpath,f'{key}_transformation')
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
    transforms = [  os.path.join(transformationpath,f'{transform_key}.mat'),
                        os.path.join(transformationpath,f'{transform_key}.nii.gz')]
    
    # change the order if it is a reverse transformation
    if 'inv' is in transform_key:
        transforms = transforms[::-1]

    # apply the transformation
    transformed_img = ants.apply_transforms( fixed=fix, moving=mov,transformlist=transforms )
    transformed_img.to_file(invwarpedimg.to_file(os.path.join(outputpath,f'{fnamekey}_transformed.tif')))


if __name__ == '__main__':
    outputpath = r'\\10.159.50.7\Analysis2\Ken\LSMS\LS_NTS\LS_NTS_F3B'
    movp = os.path.join(outputpath,'resampled_intensity_HRimage.tif')
    fixp = r"\\10.159.50.7\Analysis2\Ken\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations\Kim_ref_adult_v1_brain.tif"
    transformationpath = os.path.join(outputpath,'auto_to_HRatlas')
    key = 'auto_to_atlas' # Key string to label files.

    #ants_initial_alignment(fixp,movp,outputpath,key)
    ants_transformation(fixp,movp,transformationpath,outputpath,'auto_to_HRatlas',transform_key = 'fwdtransforms')
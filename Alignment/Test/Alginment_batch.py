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
    print(fix.shape,mov.shape)
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
            print(mytx[T])
            try:
                names = os.path.basename(mytx[T][idx]).split('.')
                names[0] = T
                name = '.'.join(names)
                #print(mytx[T][idx],name,T)
                shutil.copy(mytx[T][idx], os.path.join(transformationpath,name))
            except:
                print('no transfromation matrix')

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
    start = time.time()

    #print("Hello, World!")
    # input folder
    #imgfolder = input("Where is the probability images...")
    rootimgfolder = r'Y:\SmartSPIM2\Ken\NAcPRJ_with_antiGFP'
    metafile = pd.read_csv(r"\\10.159.50.7\LabCommon\Ken\data\NAc_PRJ\meta\NAcPRJ_meta.csv",)

    # set the atlas you want to annotate on to
    fixp = r"\\10.159.50.7\Analysis2\Ken\LSMS\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations\Kim_ref_adult_v1_brain_MR_R.tif"
    # 
    key = 'auto_to_atlas_slow' # Key string to label files.
    #ch_key = 'Ex_561_Ch1_stitched'
    ch_key = 'Ex_639_Ch2_stitched_resampled'
    #ch_key = 'Ex_639_Ch2_stitched_resampled_signal_to_atlas_transformed'
    ref_key = 'autofluo_resampled'
    #ref_key = 'autofluo_resampled_auto_to_atlas_transformed'
    for imgfname in [f for f in os.listdir(rootimgfolder) if '_DONE' in f][1:]:
        outputpath = os.path.join(rootimgfolder, imgfname)
        print(imgfname)
        #findex = int(imgfname.split('_')[6])
        #Analysis_hemi,FinalOrientation = metafile.loc[metafile.fname == imgfname,['Analysis_hemi','FinalOrientation']].values[0]
        #print(imgfname,Analysis_hemi,FinalOrientation)

        #refimage_key = f"{ref_key}_{Analysis_hemi}"
        #image_key = f"{ch_key}_{Analysis_hemi}_thresholded"

        movp = os.path.join(outputpath,f'{ref_key}.tif')
        sigp = os.path.join(outputpath,f'{ch_key}.tif')

        # outputs
        transformationpath = os.path.join(outputpath,f'{key}_transformation')

        ants_initial_alignment(movp,fixp,outputpath,key,type_of_transform = 'antsRegistrationSyNQuick[s]') # use for the first round
        #ants_initial_alignment(movp,fixp,outputpath,key,type_of_transform = 'antsRegistrationSyN[s]') # this warps the images too much
        ants_transformation(fixp,movp,transformationpath,outputpath,f'{ref_key}_auto_to_atlas',transform_key = 'invtransforms')
        ants_transformation(fixp,sigp,transformationpath,outputpath,f'{ch_key}_signal_to_atlas',transform_key = 'invtransforms')

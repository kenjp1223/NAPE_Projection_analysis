# Simple script to collect images from a root folder, rename them and move them to a new folder
import os
import shutil


if __name__ == "__main__":
    # set user dependent variables
    rootfolder = r"\\10.158.246.229\DataCommon\SmartSPIM2\Ken\MS_TRAPCeA"
    signal_channel_key = 'Ex_639_Ch2_stitched' # this is the key string to identify the signal channel in the folder name
    output_folder = r"\\10.158.246.229\DataCommon\SmartSPIM2\Ken\MS_TRAPCeA\TRAILMAP\raw"
    os.makedirs(output_folder,exist_ok=True)

    # collect all folders with _Desstripe in, this is a string indicator of the folder containing the desstriped images.
    desstripe_folders = [f for f in os.listdir(rootfolder) if '_Destripe' in f]
    
    # loop through the folders and collect the images
    # the image files to collect are stored in signal_channel_key + '_crops'
    for folder in desstripe_folders: # folder is the name of the folder containing the desstriped images
        folderpath = os.path.join(rootfolder,folder)
        print(f"Folder {folderpath} ")
        signal_channel_folder = os.path.join(folderpath,signal_channel_key + '_crops')
        if os.path.exists(signal_channel_folder):
            print(f"Folder {signal_channel_folder} exists")
            # collect the images
            images = [f for f in os.listdir(signal_channel_folder) if f.endswith('.tif')]
            for image in images:
                imagepath = os.path.join(signal_channel_folder,image)
                print("found images", imagepath)
                # rename the image
                new_image_name = folder + '_' + image
                new_image_path = os.path.join(output_folder,new_image_name)
                
                # move the image to the output folder
                shutil.move(imagepath,new_image_path)
        else:
            print(f"Folder {signal_channel_folder} does not exist")








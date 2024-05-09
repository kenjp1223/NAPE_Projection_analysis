This is designed to analyze axonal projection data imaged on the light-sheet microscope.

1. Resampling
The series of images will be resampled and transformed to match the atlas orientation.
It will take in metadata from the SmartSpim2 scope to identify resolution.
User defigned "target_orientation" 
The process will run in  container XX.

2. Alignment
Alignemnt will be conducted using ANTSpy. Currently, this will happen in a antspy container. https://hub.docker.com/r/stnava/antspy
If you are running on the HPCs, build a container with the following configuration.
singularity build /tmp/antspy.sif docker:stnava/antspy
This is currently running on (20,20,50) micron atlas space, but will be moved to (5, 5, 50) micron space for higher resolution.

This process requires 

atlas_resolution: The resolution (um) of the atlas
atlas_mri   : Image file containing MRI image for the atlas

3. Axon segmentation
Axon segmentation will be performed using TRAILMAP. TRAILMAP will be used to generate a probability map of axons in the raw image space. ILASTIK is also useful to generate this map. An idea is to dig up the Clearmap 1.0 function for ilastik. 
Example of command to run ilastik on the command line.
$IlastikBinary --headless --project=$project $iloutputpath $ilinputpath
The probability will be used to calculate a weighted mask of axons. This can be furtherused to determine a threshold, create a mask, and extract intensity/area of axons. The segmented masks and the weighted axon images will be written on the drive.
# TODO 
Options for skeltonization is in the process.

4. (Option) Cell segmentation
Cell segmentation can be performed using the ClearMap container detectCell() function.
You can import a numpy array (x,y,z) x (cells). 

5. Quantification
Quantification of axon area/intensity will be conducted using the transformation matrix and the segmented axon images. Axon images will be resampled and oriented, then warped to atlas space. The labels in atlas space will be used.
This part will run in antspy container.
This process requires
label_img   : Image file containing atlas labels for each coordinate
atlas_df    : CSV file containing meta data of atlas
atlas_mri   : Image file containing MRI image for the atlas


# SAMMed3D_App 

An adaptation of the SAM-Med3D implementation which was submitted to the SegFM challenge, as part of the evaluation stack for the IS_Evaluation_Framework by Esmaeili et al. 

This implementation was used as the guide for the most part, since it provided the logic for performing inference on images with different voxel counts and image spacings.

For our evaluation, we used SAM-Med3D turbo as the checkpoint. As such, the image normalisation preprocessing was left as-is with reference to the main branch of the SAM-Med3D repository. 
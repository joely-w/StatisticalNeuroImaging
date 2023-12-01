# Script to start playing around with NIFTI, and understanding NII datasets.
library(oro.nifti)
(ffd <- readNIfTI(r"(C:\Users\joelw\Documents\Part III\Datasets\Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information\Patient-1\1-LesionSeg-Flair.nii)"))
image(ffd, oma=rep(2,4))
orthographic(ffd, oma=rep(2,4))


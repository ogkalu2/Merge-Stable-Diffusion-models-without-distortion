# Merge-Stable-Diffusion-models-without-latent-space-distortion
I wrote the permutation spec for Stable Diffusion necessary to merge with the git-re-basin method outlined here - https://github.com/samuela/git-re-basin.
This is based on a 3rd-party python implementation of that here - https://github.com/themrzmaster/git-re-basin-pytorch.

The results of a model merge have not been tested yet but I am done with the spec.
To merge, you will need to install flax, jaxlib and pytorch 1.11.0 or lower (1.12.0 will not work). 
To install jax on windows, follow the instructions here - https://github.com/cloudhan/jax-windows-builder

Download the code folder, open cmd in the directory, transfer the desired models to the same folder and run 
"python SD_rebasin_merge.py --model_a nameofmodela.ckpt --model_b nameofmodelb.ckpt"
If not in the same directory then pathofmodela.ckpt and pathofmodelb.ckpt instead

This will not run GPU only so don't bother trying. There are many layers so merging will take a while. That's why i decided to create this before testing myself. 

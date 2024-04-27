# Merge-Stable-Diffusion-models-without-distortion
I wrote the permutation spec for Stable Diffusion necessary to merge with the git-re-basin method outlined here - https://github.com/samuela/git-re-basin.
This is based on a 3rd-party pytorch implementation of that here - https://github.com/themrzmaster/git-re-basin-pytorch.

To merge, you may need to install pytorch 1.11.0 or lower (at some point, 1.12.0 did not work but the latest versions of pytorch may have resolved the issue). 

Download the code folder, open cmd in the directory, transfer the desired models to the same folder and run 
"python SD_rebasin_merge.py --model_a nameofmodela.ckpt --model_b nameofmodelb.ckpt"

If not in the same directory then 
pathofmodela.ckpt and pathofmodelb.ckpt instead

### Notes for SDXL by DammK ###

- This is a "just make it work" version with minimal option support. 
- Tested in A1111 WebUI 1.9.3 and [sd-mecha](https://github.com/ljleb/sd-mecha) ~~obviously I want to move the codes there.~~
- [The only SDXL code only permutate for a few layers.](https://github.com/vladmandic/automatic/blob/dev/modules/merging/merge_rebasin.py)
- [However permutation spec is present as in PR.](https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion/issues/44)
- *Somehow no follow up.* [For example this PR.](https://github.com/wkpark/sd-webui-model-mixer/issues/96)
- Somehow this original implementation looks correct ~~sorry for not fully understanding the paper because there are way too much maths to read.~~
- **No pruning even it may not crash.** WebUI extensions / nodes will break.
- CLIP fix may be applied.
- **Will detect SD1.5 / SD2.1 / SDXL in auto.** However SD2.1 will not be supported. ~~who are making SD2.1?~~
- Then I'll try my best to analysis the effect. Will post to [this article about the algorithm](https://github.com/6DammK9/nai-anime-pure-negative-prompt/blob/main/ch01/rebasin.md) and [my mega mix which is 70+ in 1](https://github.com/6DammK9/nai-anime-pure-negative-prompt/blob/main/ch05/README_XL.MD)
- **Bonus task (probably impossible): Implement Algorithm 3 MERGEMANY**

```sh
python SD_rebasin_merge.py --model_a vbp.safetensors --model_b cbp.safetensors
```

```sh
python SD_rebasin_merge.py --model_a _x14-ponyDiffusionV6XL_v6.safetensors --model_b _x73-kohakuXLEpsilon_rev1.safetensors
```
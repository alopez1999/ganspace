# DeepLearning project: Disentangled Controls for GANs

## Setup
See the [setup instructions](SETUP.md).

## Usage
This repository includes versions of StyleGAN, and StyleGAN2 modified to support per-layer latent vectors.

**Interactive model exploration with KernelPCA**
```
# Explore StyleGAN2 ffhq in W space
python interactive.py --model=StyleGAN2 --est=kpca --class=ffhq --layer=style --use_w -n=1_000 -b=1_000

# Explore StyleGAN2 cars in W space
python interactive.py --model=StyleGAN2 --est=kpca --class=car --layer=style -n=1_000 -b=1_000
```
In order to change the specific kernel used by the PCA, please go to "estimators.py", line 208

### StyleGAN
1. Install TensorFlow: `conda install tensorflow-gpu=1.*`.
2. Modify methods `__init__()`, `load_model()` in `models/wrappers.py` under class StyleGAN.

### StyleGAN2
1. Follow the instructions in [models/stylegan2/stylegan2-pytorch/README.md](https://github.com/harskish/stylegan2-pytorch/blob/master/README.md#convert-weight-from-official-checkpoints). Make sure to use the fork in this specific folder when converting the weights for compatibility reasons.
2. Save the converted checkpoint as `checkpoints/stylegan2/<dataset>_<resolution>.pt`.
3. Modify methods `__init__()`, `download_checkpoint()` in `models/wrappers.py` under class StyleGAN2.

### Base code
The majority of the base code is directly borrowed from the following repository:

[ganspace]: https://github.com/harskish/ganspace

## GANSpace Citation
```
@inproceedings{härkönen2020ganspace,
  title     = {GANSpace: Discovering Interpretable GAN Controls},
  author    = {Erik Härkönen and Aaron Hertzmann and Jaakko Lehtinen and Sylvain Paris},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```

[stylegan_pytorch]: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
[stylegan2_pytorch]: https://github.com/rosinality/stylegan2-pytorch
[pretrained_stylegan]: https://github.com/justinpinkney/awesome-pretrained-stylegan

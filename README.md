# Temporal segmentation of sign language videos

This repository is my entry for the final project of the [Object recognition and computer vision 2021/2022 course](https://www.di.ens.fr/willow/teaching/recvis21/).

This is forked from the [official repository](https://github.com/RenzKa/sign-segmentation) of the following two papers:

- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/) and [Gül Varol](https://www.robots.ox.ac.uk/~gul),
*Sign language segmentation with temporal convolutional networks*, ICASSP 2021.  [[arXiv]](https://arxiv.org/abs/2011.12986)
- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Neil Fox](https://www.ucl.ac.uk/dcal/people/research-staff/neil-fox), [Gül Varol](https://www.robots.ox.ac.uk/~gul) and [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/),
*Sign Segmentation with Changepoint-Modulated Pseudo-Labelling*, CVPRW 2021. [[arXiv]](https://arxiv.org/abs/2104.13817)

Please go there for information on their initial work, as well as all the scenarios and use cases they cover (which I did not test).

This repo focuses on the replacement of the MS-TCN *segmentation model* (see second article above) by the [ASFormer model](https://github.com/chinayi/asformer) for training on pre-extracted features of the BSLCorpus dataset.

## Setup

``` bash
# Clone this repository
git clone git@github.com:arthur-cahu/sign-segmentation.git
cd sign-segmentation/
# Create signseg_env environment
conda env create -f environment.yml
conda activate signseg
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Notably, this has been tested with the following package versions:

- `python=3.9.7`
- `pytorch=1.10.1`
- `numpy=1.21.2`

## Data and models

You can download Renz et al.'s pretrained models (`models.zip [302MB]`) and data (`data.zip [5.5GB]`) used in the experiments [here](https://drive.google.com/drive/folders/17DaatdfD4GRnLJJ0RX5TcSfHGMxMS0Lm?usp=sharing) or by executing `download/download_*.sh`. The unzipped `data/` and `models/` folders should be located on the root directory of the repository (for using the demo downloading the `models` folder is sufficient).


### Data:
Please cite the original datasets when using the data: [BSL Corpus](https://bslcorpusproject.org/cava/acknowledgements-and-citation/) | [Phoenix14](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
Renz et al. provide the pre-extracted features and metadata. See [here](data/README.md) for a detailed description of the data files. 

- Features: `data/features/*/*/features.mat`
- Metadata: `data/info/*/info.pkl`

### Models:
- I3D weights, trained for sign classification: `models/i3d/*.pth.tar`
- MS-TCN weights for the demo (see tables [here](https://github.com/RenzKa/sign-segmentation#training) for links to the other models): `models/ms-tcn/*.model`

The folder structure should be as below:
```
sign-segmentation/models/
  i3d/
    i3d_kinetics_bsl1k_bslcp.pth.tar
    i3d_kinetics_bslcp.pth.tar
    i3d_kinetics_phoenix_1297.pth.tar
  ms-tcn/
    mstcn_bslcp_i3d_bslcp.model
```
## Demo (untested)
The demo folder contains a sample script to estimate the segments of a given sign language video. It is also possible to use pre-extracted I3D features as a starting point, and only apply the MS-TCN model.
`--generate_vtt` generates a `.vtt` file which can be used with [Renz et al.'s modified version of the VIA annotation tool](https://github.com/RenzKa/VIA_sign-language-annotation):

```
usage: demo.py [-h] [--starting_point {video,feature}]
               [--i3d_checkpoint_path I3D_CHECKPOINT_PATH]
               [--mstcn_checkpoint_path MSTCN_CHECKPOINT_PATH]
               [--video_path VIDEO_PATH] [--feature_path FEATURE_PATH]
               [--save_path SAVE_PATH] [--num_in_frames NUM_IN_FRAMES]
               [--stride STRIDE] [--batch_size BATCH_SIZE] [--fps FPS]
               [--num_classes NUM_CLASSES] [--slowdown_factor SLOWDOWN_FACTOR]
               [--save_features] [--save_segments] [--viz] [--generate_vtt]
```

Example usage:
``` bash
# Print arguments
python demo/demo.py -h
# Save features and predictions and create visualization of results in full speed
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 1 --save_features --save_segments --viz
# Save only predictions and create visualization of results slowed down by factor 6
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 6 --save_segments --viz
# Create visualization of results slowed down by factor 6 and .vtt file for VIA tool
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 6 --viz --generate_vtt
```

The demo will: 
1. use the `models/i3d/i3d_kinetics_bslcp.pth.tar` pretrained I3D model to extract features,
2. use the `models/ms-tcn/mstcn_bslcp_i3d_bslcp.model` pretrained MS-TCN model to predict the segments out of the features,
3. save results (depending on which flags are used).

## Training (ICASSP)

Basically, most of [Renz et al.'s use cases](https://github.com/RenzKa/sign-segmentation#training) of `main.py` should still work, but I did not test them all.

My own experiments revolved around training either MS-TCN or ASFormer (see [`transformer.py`](transformer.py)) on pre-extracted features of the BSLCorpus dataset.

Run the corresponding command (see below) to train the model with pre-extracted features on BSL Corpus.
During the training a `.log` file for tensorboard is generated. In addition the metrics get saved in `train_progress.txt`.

|ID | Model | `num_decoders` | `num_layers` | mF1B | mF1S |
|   -   |   -  |   -  |   -   |   -   |   -   |
|1 | MS-TCN (Renz *et al.*) |  | 10 |  |  |
| A | MS-TCN (control) |      | 10 | 70.37 |47.86 |
| B | ASFormer (default) | 3 | 9 | 70.41 |50.85 |
| C | ASFormer | 3 | 7    | **70.80** |49.39 |
| D | ASFormer | 4    | 9    | 70.70 |**52.76** |

Example of training command for MS-TCN:

```bash
python main.py --action train --model mstcn --extract_set train --train_data bslcp --test_data bslcp --num_epochs 10 --seed 0
```

Default training command for ASFormer:

```bash
python main.py --action train --model asformer --extract_set train --train_data bslcp --test_data bslcp --num_epochs 20 --seed 0 --num_layers 9
```

To get an idea of more ASFormer options, you can modify this command, which corresponds to the default parameters:

```bash
python main.py --action train --model asformer --extract_set train --train_data bslcp --test_data bslcp --num_epochs 20 --seed 0 --num_decoders 3 --num_layers 9 --bz 8 --lr 0.0005
```

or simply run `python main.py -h` for an exhaustive list of options.

## Citation

If you use this code and data, please cite the following:

```
@inproceedings{Renz2021signsegmentation_a,
    author       = "Katrin Renz and Nicolaj C. Stache and Samuel Albanie and G{\"u}l Varol",
    title        = "Sign Language Segmentation with Temporal Convolutional Networks",
    booktitle    = "ICASSP",
    year         = "2021",
}
```
```
@inproceedings{Renz2021signsegmentation_b,
    author       = "Katrin Renz and Nicolaj C. Stache and Neil Fox and G{\"u}l Varol and Samuel Albanie",
    title        = "Sign Segmentation with Changepoint-Modulated Pseudo-Labelling",
    booktitle    = "CVPRW",
    year         = "2021",
}
```

## License
The license in this repository only covers the code. For data.zip and models.zip we refer to the terms of conditions of original datasets.


## Acknowledgements
The code builds on the [github.com/yabufarha/ms-tcn](https://github.com/yabufarha/ms-tcn) repository. The demo reuses parts from [github.com/gulvarol/bsl1k](https://github.com/gulvarol/bsl1k).  We like to thank C. Camgoz for the help with the BSLCORPUS data preparation.
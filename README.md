# Ambisonizer: Neural Upmixing as Spherical Harmonics Generation
Yongyi Zang*, Yifan Wang* (* Equal Contribution), Minglun Lee

[arXiv link](https://arxiv.org/abs/2405.13428)

We directly generate the Ambisonic B-format from mono channel to achieve mono-to-any audio upmixing, and use stereo signal as condition to achieve stereo-to-any upmixing.

The model implementation (defined in `/model`) largely references the [EnCodec Implementation](https://github.com/facebookresearch/encodec) of SEANet.

## Updates
- May 2024: We release the model weights, pre-processed ambisonic impulse responses, and relevant scripts for training and inference.

## Getting Started
### Prepare Environment
```bash
conda create --name ambisonizer python=3.10
conda activate ambisonizer
pip install -r requirements.txt
```

### Run Inference
We provide the example embedding [here](https://drive.google.com/file/d/1S9VPkvPs0LI3oZzZRwoeLmdbm6BJpUph/view?usp=sharing). Once this file is downloaded, you can use the `model_inference_test.ipynb` to run inference using the model.

Note that the inference result will be W, X and Y channels of the first-order Ambisonic B-format. We provide a simple script in `synthesize.ipynb` to help you convert the ambisonic signals to stereo signals given two azimuth angles. Feel free to adjust it to your needs.

### Train
To train the model, start by downloading [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) dataset, then use the `generate-data.py` script to generate the training data as needed. For reproducibility, the pre-processed ambisonic impulse responses are available for download [here](https://drive.google.com/file/d/1aGC9pqxZMZPnDjctRqp3wWs-b6HzJYNC/view?usp=sharing).

Once the data is ready, you can train the model using the `train.py` script. You may need to edit the available partitions defined in the `dataset.py` script to your specific needs. This is an example training command:
```bash
python train.py --base_dir [path to the dataset] --epochs 100 --batch_size 16 --lr 1e-4 --num_workers 8 --embed_dim 64 --log_dir [path to the log directory]
```

## Citation
If you find any part of our work useful, please consider citing us:
```bibtex
@article{zang2024ambisonizer,
  title={Ambisonizer: Neural Upmixing as Spherical Harmonics Generation},
  author={Zang, Yongyi and Wang, Yifan and Lee, Minglun},
  journal={arXiv preprint arXiv:2405.13428},
  year={2024}
}
```
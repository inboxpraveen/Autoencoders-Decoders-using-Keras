# Autoencoders with Keras: Image Compression, Denoising, Retrieval and Morphing

<p align="center">
  <img src="docs/images/banner.png" alt="Autoencoders with Keras: faces compressed to a 32-number code and rebuilt by the decoder" width="100%">
</p>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](environment.yml)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B%20(Keras%203)-FF6F00.svg)](requirements.txt)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](tests)
[![Sponsor](https://img.shields.io/badge/sponsor-inboxpraveen-ea4aaa.svg)](https://github.com/sponsors/inboxpraveen)

Three image autoencoders in TensorFlow and Keras (a linear PCA-style model, a deep convolutional model and a denoising model) that share one encoder/decoder interface, plus two things you can do with a trained encoder: find similar images and morph one image into another. One command line tool trains and runs all of them on the Labeled Faces in the Wild dataset or on a folder of your own images.

Use it to learn how autoencoders work, as a baseline for image compression and denoising experiments, or as a small, readable starting point for image embeddings and similarity search.

![Test faces, their PCA reconstructions and their convolutional reconstructions from 32 numbers each](docs/images/reconstructions.png)

*Held-out LFW faces (top), rebuilt from a 32-number code by the linear model (middle) and the convolutional model (bottom). Each 32x32 colour image is 3,072 numbers, so both codes are 96 times smaller.*

## What it does

- **Linear (PCA-style) autoencoder**: one dense layer down to the code and one back up, no activations. A fast baseline that learns a PCA-like subspace.
- **Convolutional autoencoder**: four Conv2D + MaxPool blocks down to a 32-number code, four Conv2DTranspose blocks back up. Saves its encoder and decoder for reuse.
- **Denoising autoencoder**: the same network with a 512-number code, trained on images with fresh Gaussian noise each epoch and the clean image as the target.
- **Image retrieval**: encodes a set of images once and finds the nearest neighbours of any query in code space with scikit-learn.
- **Image morphing**: blends the codes of two images step by step and decodes each step.
- **Your own data**: `--data-dir` trains on any folder of images (subfolders included) instead of LFW.
- **Saved models that remember their settings**: weights are written with a `model_config.json`, so retrieval and morphing rebuild the model with the right code size automatically.

## Quick start

```bash
git clone https://github.com/inboxpraveen/Autoencoders-Decoders-using-Keras.git
cd Autoencoders-Decoders-using-Keras

conda env create -f environment.yml
conda activate autoencoders

python example_usage.py
```

`example_usage.py` downloads LFW on first use (about 200 MB, cached by scikit-learn in `~/scikit_learn_data`), trains the linear model for five epochs and writes a reconstruction to `results/`. After the download it finishes in under 30 seconds on a laptop CPU.

Without Conda, use any Python 3.10 or newer: `pip install -r requirements.txt`.

## Usage

Everything runs through `main.py`. Train a model first, then point retrieval or morphing at it.

| Mode | Command | Writes |
|------|---------|--------|
| Linear (PCA) | `python main.py --mode pca --epochs 15 --code-size 32` | `results/pca/` |
| Convolutional | `python main.py --mode convolutional --epochs 25 --code-size 32` | `results/convolutional/`, `saved_models/` |
| Denoising | `python main.py --mode denoising --epochs 25 --code-size 512 --noise-sigma 0.1` | `results/denoising/`, `saved_models/` |
| Retrieval | `python main.py --mode retrieval --model-path saved_models --n-queries 3 --n-neighbors 5` | `results/retrieval/` |
| Morphing | `python main.py --mode morphing --model-path saved_models --n-pairs 5 --n-steps 7` | `results/morphing/` |

The convolutional and denoising modes both save to `--save-dir` (default `saved_models`), so the second run replaces the first. Give each its own folder to keep both:

```bash
python main.py --mode convolutional --save-dir saved_models/conv
python main.py --mode denoising --code-size 512 --save-dir saved_models/denoising
python main.py --mode retrieval --model-path saved_models/conv
```

Train on your own images (any mix of jpg, png, bmp, gif and webp, in any folder structure):

```bash
python main.py --mode convolutional --data-dir path/to/images --epochs 30
python main.py --mode retrieval --data-dir path/to/images --model-path saved_models
```

Run `python main.py --help` for every flag. The [usage guide](docs/usage.md) covers each mode, all flags, outputs and troubleshooting.

## Use it from Python

Every builder returns three models that share weights: the full autoencoder, the encoder and the decoder.

```python
from models import build_convolutional_autoencoder
from image_retrieval import ImageRetrieval
from image_morphing import interpolate_images
from utils import load_image_folder, prepare_data

X = load_image_folder("path/to/images", dimx=32, dimy=32)   # pixels in [0, 255]
X_train, X_test = prepare_data(X, test_size=0.1)            # scaled to [-0.5, 0.5]

autoencoder, encoder, decoder = build_convolutional_autoencoder((32, 32, 3), code_size=32)
autoencoder.compile(optimizer="adamax", loss="mse")
autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=25, batch_size=32)

codes = encoder.predict(X_test)                  # (n, 32) embeddings

search = ImageRetrieval(encoder)
search.index_images(X_train)
distances, matches = search.find_similar(X_test[0], n_neighbors=5)

frames = interpolate_images(X_test[0], X_test[1], encoder, decoder, n_steps=7)
```

## How it works

![Architecture: data loading, the convolutional encoder and decoder with tensor shapes, the three model variants, and the retrieval and morphing applications](docs/images/architecture.png)

An autoencoder is trained to reproduce its input through a narrow middle layer, the code. Because the code is much smaller than the image, the encoder has to keep what matters and the decoder has to rebuild the rest. Training minimises the mean squared error between input and output, so no labels are needed.

The convolutional encoder halves the image four times (32, 16, 8, 4, 2 pixels) while the filters grow from 32 to 256, flattens the 2x2x256 result to 1,024 numbers and maps it to the code with a dense layer. The decoder mirrors it with a dense layer, a reshape and four stride-2 transposed convolutions. The [architecture guide](docs/architecture.md) has every layer, the parameter counts, the training setup and the maths.

## Results

Measured on all 13,233 LFW faces at 32x32 (11,909 for training, 1,324 held out), on a laptop CPU (Intel Core i7-14650HX, TensorFlow 2.21). MSE is on the held-out faces with pixels scaled to [-0.5, 0.5]; lower is better. Run time is the whole command, data loading and plots included.

| Model | Code size | Parameters | Test MSE | Run time |
|-------|-----------|------------|----------|---------------|
| Linear (PCA) | 32 | 199,712 | 0.0081 | 2 min (15 epochs) |
| Convolutional | 32 | 843,171 | 0.0074 | 7 min (25 epochs) |
| Denoising, σ = 0.1 | 512 | 1,826,691 | 0.0046 | 7 min (25 epochs) |

For denoising, the MSE compares the model's output on noisy faces with the clean originals. The noisy input itself scores 0.0100 against the clean faces.

**Denoising**: clean faces, the same faces with Gaussian noise added, and the model's output.

![Clean, noisy and denoised faces](docs/images/denoising.png)

**Morphing**: two real photos at the ends, decoded blends of their codes in between.

![Two morphing sequences between pairs of faces](docs/images/morphing.png)

**Retrieval**: a held-out query and its five nearest training faces in code space.

![Two query faces and their five nearest neighbours](docs/images/retrieval.png)

A plain autoencoder does not force its code space to be smooth, so some blends in the middle of a morph look less like a face than the ends. A variational autoencoder fixes this and is on the roadmap.

## Project layout

```
main.py                  command line entry point (five modes)
example_usage.py         one minute demo
config.py                default settings as dataclasses
models/
  pca_autoencoder.py           linear model and its training loop
  convolutional_autoencoder.py convolutional model, training, saving and loading
  denoising_autoencoder.py     denoising training with fresh noise each epoch
utils/
  data_loader.py         LFW, image folders, scaling and splitting
  noise.py               Gaussian, salt and pepper, and occlusion noise
  visualization.py       plots for reconstructions, denoising and training curves
image_retrieval.py       k-nearest-neighbour search on codes
image_morphing.py        code interpolation
tests/                   fast pytest suite on random arrays
docs/                    usage and architecture guides, images
```

## Tests

```bash
pytest -q
```

The suite builds every model, checks shapes and parameter counts, data scaling, the folder loader, noise, retrieval, morphing, saving and reloading settings, and that the denoising trainer keeps its best weights. It runs on a CPU in under a minute and never downloads LFW.

## Roadmap

- Variational autoencoder for a smooth code space and sampling new images
- Salt and pepper and occlusion noise as `--noise-type` options (already in `utils/noise.py`)
- Larger images (64x64 and up) with a deeper encoder
- Code space plots with t-SNE or UMAP
- A small Gradio demo for retrieval and morphing

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the setup and the checks to run.

## Sponsor

If this project helps your work or your teaching, you can support it through [GitHub Sponsors](https://github.com/sponsors/inboxpraveen). Sponsorship pays for the time spent on the roadmap above and on keeping the code current with new TensorFlow and Keras releases.

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgments

Labeled Faces in the Wild: G. B. Huang, M. Ramesh, T. Berg and E. Learned-Miller, *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*, University of Massachusetts, Amherst, Technical Report 07-49, 2007. [vis-www.cs.umass.edu/lfw](http://vis-www.cs.umass.edu/lfw/)

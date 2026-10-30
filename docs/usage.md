# Usage guide

Every command below runs from the repository root inside the `autoencoders` Conda environment (see the [README](../README.md#quick-start)).

## The one minute demo

```bash
python example_usage.py
```

Downloads LFW on first use, trains the linear model for five epochs and writes:

- `results/quick_example_samples.png`: six training faces
- `results/quick_example_reconstruction.png`: one test face, its 32-number code and the reconstruction

## Data

By default every mode loads all 13,233 photos of [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) through scikit-learn. The first run downloads about 200 MB into `~/scikit_learn_data`; later runs read the cache. A square 125x125 crop around each face is resized to `--image-size` (32 by default).

To use your own images instead, pass `--data-dir`. Every `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif` and `.webp` file in the folder and its subfolders is loaded, converted to RGB and resized to `--image-size` square. Other files are ignored. Images are stretched, not cropped, so roughly square or centre-cropped pictures work best.

```bash
python main.py --mode convolutional --data-dir D:/datasets/products --epochs 30
```

In both cases pixels are scaled to [-0.5, 0.5] and 10 percent of the images (`--test-size`) are held out for validation, with a fixed seed so the split is repeatable.

If the LFW download fails, the tool prints a warning and falls back to random images so you can still check that the pipeline runs. Results from that fallback are meaningless; fix the download or use `--data-dir`.

## Training modes

### Linear (PCA-style)

```bash
python main.py --mode pca --epochs 15 --code-size 32
```

Flatten, one dense layer to the code, one dense layer back, no activations. It learns the same kind of subspace as PCA and is the fastest way to check your setup. Training stops early when the validation loss stops improving for five epochs. This mode does not save weights.

Writes `results/pca/training_history.png` and `results/pca/reconstruction_1.png` to `_5.png` (original, code, reconstruction).

### Convolutional

```bash
python main.py --mode convolutional --epochs 25 --code-size 32
```

The deep model described in the [architecture guide](architecture.md). Keeps the best validation epoch (early stopping after ten epochs without improvement), halves the learning rate after five, and saves the best full model to `checkpoints/conv_autoencoder_best.keras`.

Writes:

- `results/convolutional/training_history.png` and `reconstruction_1.png` to `_5.png`
- `saved_models/encoder.weights.h5`, `decoder.weights.h5` and `model_config.json`
- `logs/convolutional/` for TensorBoard, only if TensorBoard is installed (`pip install tensorboard`)

### Denoising

```bash
python main.py --mode denoising --epochs 25 --code-size 512 --noise-sigma 0.1
```

Same network, trained to map noisy images to clean ones. Fresh Gaussian noise with standard deviation `--noise-sigma` is drawn every epoch so the model cannot memorise it. The best epoch's weights are kept and also saved to `checkpoints/denoising_autoencoder_best.keras`.

Writes `results/denoising/training_history.png`, `denoising_1.png` to `_5.png` (clean, noisy, denoised) and the same three files in `saved_models/` as the convolutional mode.

A larger code (512) leaves room for detail; with 32 numbers the output is clean but blurry. Pixel values span 1.0 after scaling, so σ = 0.1 is mild noise and σ = 0.3 is heavy.

## Using a trained model

Both modes rebuild the convolutional network, load `encoder.weights.h5` (and `decoder.weights.h5` for morphing) from `--model-path`, and read `model_config.json` to use the same `--code-size` and `--image-size` the model was trained with. For model folders saved before `model_config.json` existed, pass `--code-size` yourself (for example `--code-size 512` for a denoising model).

### Retrieval

```bash
python main.py --mode retrieval --model-path saved_models --n-queries 3 --n-neighbors 5
```

Encodes every training image, fits a scikit-learn `NearestNeighbors` index on the codes (Euclidean distance), picks random held-out images as queries and plots each query with its closest matches and their distances. Writes `results/retrieval/retrieval_query_1.png` and so on.

### Morphing

```bash
python main.py --mode morphing --model-path saved_models --n-pairs 5 --n-steps 7
```

Picks random pairs of held-out images, encodes both, blends the codes as `(1 - α) z1 + α z2` for `--n-steps` values of α from 0 to 1 and decodes each blend. Writes `results/morphing/morphing_pair_1.png` and so on.

## All flags

| Flag | Default | Used by | Meaning |
|------|---------|---------|---------|
| `--mode` | required | all | `pca`, `convolutional`, `denoising`, `retrieval` or `morphing` |
| `--data-dir` | LFW | all | folder of your own images |
| `--image-size` | 32 | all | images are resized to this many pixels square |
| `--test-size` | 0.1 | all | fraction held out for validation and queries |
| `--code-size` | 32 | all | length of the code; use 512 for denoising |
| `--epochs` | 25 | training | maximum epochs (early stopping may end sooner) |
| `--batch-size` | 32 | training | images per step |
| `--noise-sigma` | 0.1 | denoising | standard deviation of the Gaussian noise |
| `--n-queries` | 3 | retrieval | number of query images |
| `--n-neighbors` | 5 | retrieval | matches shown per query |
| `--n-pairs` | 5 | morphing | number of image pairs |
| `--n-steps` | 7 | morphing | frames per morph, ends included |
| `--save-dir` | `saved_models` | convolutional, denoising | where weights and settings are written |
| `--checkpoint-dir` | `checkpoints` | convolutional, denoising | where the best full model is written |
| `--model-path` | `saved_models` | retrieval, morphing | where weights and settings are read from |

`--image-size` must be divisible by 16 for the convolutional models (four 2x poolings). 32 and 64 work; 64 needs roughly four times the compute.

## Output folders

```
results/
  sample_images.png      six training images, written by every mode
  pca/  convolutional/  denoising/  retrieval/  morphing/
saved_models/
  encoder.weights.h5  decoder.weights.h5  model_config.json
checkpoints/
  conv_autoencoder_best.keras  denoising_autoencoder_best.keras
logs/                    TensorBoard logs, if TensorBoard is installed
```

All four folders are in `.gitignore`.

## Hardware

TensorFlow runs on the CPU on native Windows; for GPU training use Linux or WSL2 and install `tensorflow[and-cuda]`. Training times on a laptop CPU are in the [README results](../README.md#results).

## Troubleshooting

**Out of memory.** Lower `--batch-size` (16 or 8). The whole dataset is held in memory as float32: 13,233 images at 32x32 take about 160 MB; at 64x64 about 650 MB.

**Retrieval or morphing cannot load the weights (a shape mismatch error).** The model was trained with a different code or image size and has no `model_config.json`. Pass the same `--code-size` and `--image-size` you trained with.

**Reconstructions are blurry.** That is expected from a 32-number code trained with MSE. Train longer, or raise `--code-size` to 128 or more for sharper output at less compression.

**The LFW download fails.** Check your connection or proxy, delete a partial download from `~/scikit_learn_data/lfw_home`, or train on your own images with `--data-dir`.

**No `logs/` folder after training.** TensorBoard is not installed by recent TensorFlow releases. Run `pip install tensorboard` and train again.

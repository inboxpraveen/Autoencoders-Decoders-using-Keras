# Changelog

## 2026-10-30

### Fixed

- LFW images were scaled twice. scikit-learn already returns pixels in [0, 1] and `prepare_data()` divided by 255 again, so every model trained on images that were almost uniformly -0.5. Models trained before this fix should be retrained.
- LFW now loads all 13,233 photos with a square face crop at the requested size. Before, only the 3,023 photos of people with 20 or more images were used, at 16x12 pixels, stretched to 32x32.
- The models failed to build on Keras 3.15 and newer (`Dense` received a NumPy integer for `units`).
- Convolutional training crashed when TensorBoard is not installed, which is the default with TensorFlow 2.21. TensorBoard logging is now used only when the package is present.
- Denoising training reloaded its best checkpoint into a new model object, so the encoder and decoder that were saved kept the last epoch's weights instead of the best ones. The best weights are now restored in place.
- Running any mode on a fresh clone failed to save `results/sample_images.png` because `results/` did not exist yet.
- The convolutional code layer used ReLU, and 8 of its 32 units (90 of 512 in the denoising model) were zero for every face, so the model lost to the linear baseline. The code layer is now linear. Old weight files still load because the shapes match, but they give wrong results; retrain them.
- Convolutional training wrote a new checkpoint file for every improving epoch; it now keeps one `conv_autoencoder_best.keras`.

### Added

- `--data-dir` to train and run every mode on a folder of your own images (`load_image_folder()` in `utils`).
- `model_config.json` next to saved weights. Retrieval and morphing read it and use the code size and image size the model was trained with, so a 512-number denoising model loads without extra flags.
- A pytest suite in `tests/` that runs on random arrays in under a minute.
- `environment.yml` for Conda, `CONTRIBUTING.md`, this changelog.
- A new logo, banner, architecture diagram and result figures from a real training run, and rewritten README, usage guide and architecture guide with measured numbers.

### Changed

- Requirements: TensorFlow 2.16 or newer (Keras 3), NumPy 2 allowed, Pillow listed explicitly. Tested with Python 3.11 and TensorFlow 2.21.
- `START_HERE.md` and `PROJECT_SUMMARY.md` moved to `docs/usage.md` and `docs/architecture.md`.

## 2025-12

- Rebuilt the original 2019 TensorFlow 1.x notebook as a Python package with a command line interface, three autoencoders, retrieval and morphing.

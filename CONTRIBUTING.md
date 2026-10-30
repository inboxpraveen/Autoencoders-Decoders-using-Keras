# Contributing

Bug reports, questions and pull requests are welcome.

## Reporting a problem

Open an issue with your operating system, Python and TensorFlow versions (`python -c "import tensorflow as tf; print(tf.__version__)"`), the exact command you ran, and the last lines of its output. For training problems, say which dataset you used (LFW or `--data-dir`) and attach `results/<mode>/training_history.png` if you have it.

## Working on the code

```bash
conda env create -f environment.yml
conda activate autoencoders
pytest -q
```

The tests use small random arrays, run on a CPU in under a minute and never download LFW. Before opening a pull request, also run one short real training run, for example:

```bash
python main.py --mode convolutional --epochs 2
```

Keep changes small and focused, and match the style of the surrounding code: type hints, Google style docstrings, and builders that return `(autoencoder, encoder, decoder)` sharing weights. Add or adjust a test in `tests/` when you change a model, a data helper or an application. Update `README.md` and the guides in `docs/` when behaviour or command line flags change.

Good first contributions are listed under [Roadmap](README.md#roadmap) in the README.

## Pull requests

Branch from `master`, describe what the change does and why, and mention how you tested it. One topic per pull request.

"""Fast checks for the models, data helpers and applications.

All tests use small random arrays, so they run on a CPU in well under a
minute and never download LFW.
"""

import numpy as np
import pytest
from PIL import Image

from image_morphing import interpolate_images
from image_retrieval import ImageRetrieval
from models import (
    build_convolutional_autoencoder,
    build_denoising_autoencoder,
    build_pca_autoencoder,
)
from models.convolutional_autoencoder import read_model_config, save_weights
from models.denoising_autoencoder import train_denoising_autoencoder
from utils import apply_gaussian_noise, load_image_folder, prepare_data

SHAPE = (32, 32, 3)


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, size=(24, *SHAPE)).astype('float32')


@pytest.mark.parametrize('build, code_size', [
    (build_pca_autoencoder, 32),
    (build_convolutional_autoencoder, 32),
    (build_denoising_autoencoder, 512),
])
def test_models_round_trip_shapes(build, code_size, images):
    autoencoder, encoder, decoder = build(SHAPE, code_size=code_size)
    codes = encoder.predict(images[:4], verbose=0)
    assert codes.shape == (4, code_size)
    assert decoder.predict(codes, verbose=0).shape == (4, *SHAPE)
    assert autoencoder.predict(images[:4], verbose=0).shape == (4, *SHAPE)


def test_parameter_counts_match_docs():
    pca, _, _ = build_pca_autoencoder(SHAPE, code_size=32)
    conv, _, _ = build_convolutional_autoencoder(SHAPE, code_size=32)
    assert pca.count_params() == 3072 * 32 + 32 + 32 * 3072 + 3072
    assert conv.count_params() == 843_171


def test_prepare_data_scales_to_centered_range():
    raw = np.full((10, *SHAPE), 255.0, dtype='float32')
    raw[:5] = 0.0
    X_train, X_test = prepare_data(raw, test_size=0.2)
    combined = np.concatenate([X_train, X_test])
    assert combined.min() == pytest.approx(-0.5)
    assert combined.max() == pytest.approx(0.5)
    assert len(X_test) == 2


def test_load_image_folder_reads_nested_images(tmp_path):
    (tmp_path / 'sub').mkdir()
    Image.new('RGB', (64, 48), (255, 0, 0)).save(tmp_path / 'a.png')
    Image.new('L', (20, 20), 128).save(tmp_path / 'sub' / 'b.jpg')
    (tmp_path / 'notes.txt').write_text('not an image')

    X = load_image_folder(str(tmp_path), dimx=32, dimy=32)
    assert X.shape == (2, 32, 32, 3)
    assert X.dtype == np.float32
    assert 0.0 <= X.min() and X.max() <= 255.0


def test_load_image_folder_fails_loudly_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_folder(str(tmp_path))


def test_gaussian_noise_is_seeded_and_keeps_dtype(images):
    a = apply_gaussian_noise(images, sigma=0.1, seed=1)
    b = apply_gaussian_noise(images, sigma=0.1, seed=1)
    assert a.dtype == images.dtype
    np.testing.assert_array_equal(a, b)
    assert np.std(a - images) == pytest.approx(0.1, rel=0.1)


def test_retrieval_finds_the_query_itself_first(images):
    _, encoder, _ = build_pca_autoencoder(SHAPE, code_size=32)
    retrieval = ImageRetrieval(encoder)
    retrieval.index_images(images)
    distances, similar = retrieval.find_similar(images[3], n_neighbors=3)
    assert distances[0] == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_array_equal(similar[0], images[3])


def test_morphing_has_requested_steps(images):
    _, encoder, decoder = build_pca_autoencoder(SHAPE, code_size=32)
    morphed = interpolate_images(images[0], images[1], encoder, decoder, n_steps=5)
    assert morphed.shape == (5, *SHAPE)


def test_model_config_round_trip(tmp_path, images):
    _, encoder, decoder = build_convolutional_autoencoder(SHAPE, code_size=32)
    assert read_model_config(str(tmp_path)) == {}
    save_weights(encoder, decoder, save_dir=str(tmp_path),
                 config={'code_size': 32, 'image_size': 32})
    assert read_model_config(str(tmp_path)) == {'code_size': 32, 'image_size': 32}
    assert (tmp_path / 'encoder.weights.h5').exists()


def test_denoising_keeps_best_weights_in_shared_encoder(tmp_path, images):
    autoencoder, encoder, decoder = build_denoising_autoencoder(SHAPE, code_size=16)
    history = train_denoising_autoencoder(
        autoencoder, images[:16], images[16:],
        noise_function=lambda x, **_: x,  # no noise, so validation is repeatable
        noise_params={},
        epochs=4,
        batch_size=8,
        checkpoint_dir=str(tmp_path),
        verbose=0,
    )
    restored = np.mean((decoder.predict(encoder.predict(images[16:], verbose=0),
                                        verbose=0) - images[16:]) ** 2)
    assert restored == pytest.approx(min(history.history['val_loss']), rel=1e-3)

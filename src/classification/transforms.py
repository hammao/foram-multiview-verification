"""The single evaluation preprocessor used for validation, test, calibration, audit,
noise tests, embedding export and deployment.

Before the September 2026 revision the internal evaluation (``eval.py``, the validation
loop in ``train.py`` and ``calibrate_temperature.py``) used timm's default evaluation
transform, which for a 384-pixel input is ``Resize(438)`` followed by ``CenterCrop(384)``,
a 14 % zoom into the already letterboxed image. Deployment (``src/inference/pipeline.py``),
the audit scorer and the noise test fed the letterboxed 384 x 384 image directly. The same
checkpoint therefore scored differently on the same images depending on which script
evaluated it (0.990 under the centre crop, 0.970 without it, on the 202-image test split).

Every evaluation path now goes through :func:`build_eval_transform`, which is exactly the
deployment preprocessor: letterbox to ``input_size`` on a black background, convert to a
tensor and normalise with ImageNet statistics. No crop. The training transform is
unchanged and is documented in :func:`build_train_transform`.
"""

from __future__ import annotations

from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox(im: Image.Image, size: int) -> Image.Image:
    """Aspect-preserving pad to ``size`` x ``size`` on a black background."""
    im = im.convert("RGB")
    return ImageOps.pad(im, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))


def build_eval_transform(input_size: int):
    """Deployment preprocessor: letterbox, tensor, ImageNet normalisation. No crop."""
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Lambda(lambda im: letterbox(im, int(input_size))),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform(input_size: int, augmentation: str):
    """Training augmentation, unchanged from the runs reported in the manuscript.

    timm's ``create_transform(is_training=True)`` applies, in order: RandomResizedCrop to
    ``input_size`` (scale 0.08 to 1.0, ratio 3/4 to 4/3), RandomHorizontalFlip (p = 0.5),
    RandAugment (two operations, magnitude 9, magnitude std 0.5) when ``augmentation`` is
    ``rand_augment``, ToTensor, ImageNet normalisation and RandomErasing (p = 0.25, pixel
    mode). The horizontal flip mirrors apparent coiling direction; none of the 31 catalogue
    species is separated from another catalogue species by chirality, but this must be
    revisited before chiral species pairs are added.
    """
    import timm

    aa = "rand-m9-mstd0.5-inc1" if augmentation == "rand_augment" else None
    return timm.data.create_transform(
        input_size=(3, int(input_size), int(input_size)),
        is_training=True,
        auto_augment=aa,
        re_prob=0.25,
        re_mode="pixel",
    )

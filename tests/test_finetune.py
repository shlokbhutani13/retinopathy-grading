from pathlib import Path

import numpy as np
import torch

from retinopathy.finetune import class_aware_sampler, load_ordinal_checkpoint
from retinopathy.ordinal import OrdinalClassifier


def test_class_aware_sampler_gives_minority_classes_more_weight():
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 3, 4])

    sampler = class_aware_sampler(labels, seed=19)

    weights = sampler.weights.numpy()
    assert weights[0] < weights[4] < weights[6]
    assert sampler.num_samples == len(labels)
    assert sampler.replacement is True
    assert sampler.generator.initial_seed() == 19


def test_class_aware_sampler_requires_every_grade():
    labels = np.array([0, 0, 1, 1])

    try:
        class_aware_sampler(labels, seed=19)
    except ValueError as error:
        assert "every grade" in str(error)
    else:
        raise AssertionError("missing grades must be rejected")


def test_load_ordinal_checkpoint_restores_state(tmp_path: Path):
    source = OrdinalClassifier(pretrained=False)
    first_key = next(iter(source.state_dict()))
    state = source.state_dict()
    state[first_key] = torch.full_like(state[first_key], 0.125)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": state,
            "temperature": 1.4,
            "image_size": 384,
            "architecture": "efficientnet_b0_ordinal",
        },
        path,
    )

    model, metadata = load_ordinal_checkpoint(path, torch.device("cpu"))

    assert torch.allclose(
        model.state_dict()[first_key],
        torch.full_like(model.state_dict()[first_key], 0.125),
    )
    assert metadata["image_size"] == 384
    assert metadata["temperature"] == 1.4

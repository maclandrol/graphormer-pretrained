import warnings
import os
import errno

from loguru import logger
import datamol as dm

import torch
import torch.hub
import torch.distributed as dist
from torch.hub import load_state_dict_from_url
from graphormer_pretrained.utils.common import patch_torchdist_module

dist = patch_torchdist_module(dist)

ORIGINAL_PRETRAINED_MODEL_URLS = {
    "pcqm4mv1_graphormer_base": "https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv1/checkpoint_best_pcqm4mv1.pt",
    "pcqm4mv2_graphormer_base": "https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv2/checkpoint_best_pcqm4mv2.pt",
    "oc20is2re_graphormer3d_base": "https://szheng.blob.core.windows.net/graphormer/modelzoo/oc20is2re/checkpoint_last_oc20_is2re.pt",
    "pcqm4mv1_graphormer_base_for_molhiv": "https://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv1/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
}

PRETRAINED_MODEL_URLS_ALT = {
    "pcqm4mv1_graphormer_base": "gs://molfeat-store-prod/checkpoints/graphormer/pcqm4mv1/checkpoint_best_pcqm4mv1.pt",
    "pcqm4mv2_graphormer_base": "gs://molfeat-store-prod/checkpoints/graphormer/pcqm4mv2/checkpoint_best_pcqm4mv2.pt",
    "oc20is2re_graphormer3d_base": "gs://molfeat-store-prod/checkpoints/graphormer/oc20is2re/checkpoint_last_oc20_is2re.pt",
    "pcqm4mv1_graphormer_base_for_molhiv": "gs://molfeat-store-prod/checkpoints/graphormer/pcqm4mv1/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
}

PRETRAINED_MODEL_URLS = PRETRAINED_MODEL_URLS_ALT


def load_state_dict_from_fsspec_url(
    url,
    model_dir=None,
    map_location=None,
    progress=True,
    file_name=None,
):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        file_name (string, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn("TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead")

    if model_dir is None:
        model_dir = dm.fs.get_cache_dir(app_name="graphformer-pretrained", suffix="models")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = dm.fs.get_basename(url)
    if file_name is not None:
        filename = file_name

    cached_file = dm.fs.join(model_dir, filename)
    if not os.path.exists(cached_file):
        logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        dm.fs.copy_file(url, cached_file, progress=progress)

    # NOTE(hadim): since those are private, it might be fragile with respect to future torch versions...
    if torch.hub._is_legacy_zip_format(cached_file):
        return torch.hub._legacy_zip_load(cached_file, model_dir, map_location)

    return torch.load(cached_file, map_location=map_location)


def load_pretrained_model(pretrained_model_name, verbose=False):
    pretrained_model_path = PRETRAINED_MODEL_URLS.get(pretrained_model_name, pretrained_model_name)
    if verbose and pretrained_model_path not in PRETRAINED_MODEL_URLS:
        logger.warning(
            f"Pretrained model: {pretrained_model_name} isnot a recognized Graphormer pretrained model !"
        )
    if not dist.is_initialized():
        return load_state_dict_from_fsspec_url(pretrained_model_path, progress=True)["model"]
    else:
        pretrained_model = load_state_dict_from_fsspec_url(
            pretrained_model_path,
            progress=True,
            file_name=f"{pretrained_model_name}_{dist.get_rank()}",
        )["model"]
        dist.barrier()
        return pretrained_model

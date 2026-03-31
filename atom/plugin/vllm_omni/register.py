from typing import Optional
import logging

from atom.plugin.prepare import _set_framework_backbone
from atom.utils import envs
from atom.plugin.vllm.mla_patch import patch_vllm_mla_attention
from atom.plugin.vllm.register import (
    _patch_vllm_attention_process_weights_after_loading,
    _VLLM_MODEL_REGISTRY_OVERRIDES,
)

logger = logging.getLogger("atom")

# this flag is used to enable the vllm-omni plugin mode
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN


def register_omni_platform() -> Optional[str]:

    if disable_vllm_plugin:
        logger.info("Disable ATOM OOT plugin platforms (vllm-omni)")
        return None

    _set_framework_backbone("vllm")

    # return the ATOM omni platform to vllm-omni
    return "atom.plugin.vllm_omni.platform.ATOMOmniPlatform"


def register_omni_model() -> None:
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register (vllm-omni)")
        return

    from vllm_omni.model_executor.models.registry import OmniModelRegistry
    import vllm.model_executor.models.registry as vllm_model_registry

    any_updated = False
    for arch, qual in _VLLM_MODEL_REGISTRY_OVERRIDES.items():
        module_name, class_name = qual.split(":", 1)
        existing = OmniModelRegistry.models.get(arch)
        if existing is not None:
            # If already overridden to the same target, skip re-registering.
            if (
                getattr(existing, "module_name", None) == module_name
                and getattr(existing, "class_name", None) == class_name
            ):
                continue

        logger.info(f"Register model {arch} to vLLM-Omni with {qual}")
        OmniModelRegistry.register_model(arch, qual)
        any_updated = True

    # clear lru cache
    if any_updated:
        vllm_model_registry._try_load_model_cls.cache_clear()
        vllm_model_registry._try_inspect_model_cls.cache_clear()

    patch_vllm_mla_attention()
    # patch attention process weights after loading
    try:
        from vllm.attention.layer import Attention, MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import Attention, MLAAttention

    _patch_vllm_attention_process_weights_after_loading(Attention)
    _patch_vllm_attention_process_weights_after_loading(MLAAttention)

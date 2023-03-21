def safe_hasattr(obj, k):
    """Returns True if the given key exists and is not None."""
    return getattr(obj, k, None) is not None


def patch_torchdist_module(mod):
    """Patch imported torch distributed module"""
    if mod.is_available():
        return mod
    mod.is_initialized = lambda: False
    return mod

"""Shared pytest configuration.

Installs lightweight stubs for optional runtime dependencies (decord, psutil)
that are not available in the plain CI/dev environment.  The stubs satisfy
module-level imports so that pure-logic functions can be tested without
needing GPU libraries or system-monitoring packages.
"""
import sys
import types


def _stub(name, attrs=None):
    """Register an empty module stub in sys.modules if not already present."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod


# decord: used by Video_loader and Function_prepare_images for GPU-accelerated
# video reading.  Tests that call filter_cnts don't exercise these paths.
_stub("decord")

# psutil: used by security_settings_track.check_memory_overload() only.
# Stub provides the one function actually called at module scope (none).
_stub("psutil")

# cupy: GPU array library.  Stub ensures gpu_utils.CUPY_AVAILABLE is False in
# local/CI environments without a CUDA GPU, so CPU fallback paths are exercised.
_stub("cupy")

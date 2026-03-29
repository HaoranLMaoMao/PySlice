"""Unit tests for pyslice/backend.py"""
import os
import tempfile

import numpy as np
import pytest

from pyslice.backend import (
    NumpyBackend,
    Backend,
    make_backend,
    to_cpu,
    to_numpy,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch
    from pyslice.backend import TorchBackend


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numpy_backend():
    return NumpyBackend()


@pytest.fixture(params=["numpy", pytest.param("torch", marks=pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not available"))])
def backend(request):
    if request.param == "numpy":
        return NumpyBackend()
    return TorchBackend(device="cpu")


def to_np(x):
    """Coerce any array-like to a plain numpy array for assertions."""
    return to_numpy(x)


# ---------------------------------------------------------------------------
# to_cpu / to_numpy
# ---------------------------------------------------------------------------

class TestStandaloneConversions:
    def test_to_cpu_numpy_passthrough(self):
        a = np.array([1.0, 2.0])
        assert to_cpu(a) is a

    def test_to_cpu_scalar(self):
        assert to_cpu(3.14) == 3.14

    def test_to_numpy_list(self):
        result = to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_to_numpy_numpy_array(self):
        a = np.array([1.0, 2.0])
        result = to_numpy(a)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, a)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_to_cpu_torch_tensor(self):
        t = torch.tensor([1.0, 2.0])
        result = to_cpu(t)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_to_numpy_torch_tensor(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0])


# ---------------------------------------------------------------------------
# make_backend factory
# ---------------------------------------------------------------------------

class TestMakeBackend:
    def test_returns_numpy_when_forced(self, monkeypatch):
        monkeypatch.setenv("PYSLICE_BACKEND", "numpy")
        b = make_backend()
        assert isinstance(b, NumpyBackend)

    def test_returns_backend_instance(self):
        b = make_backend()
        assert isinstance(b, Backend)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_returns_torch_when_available(self, monkeypatch):
        monkeypatch.delenv("PYSLICE_BACKEND", raising=False)
        b = make_backend(device="cpu")
        assert isinstance(b, TorchBackend)


# ---------------------------------------------------------------------------
# NumpyBackend — array creation
# ---------------------------------------------------------------------------

class TestNumpyBackendCreation:
    def test_asarray_default_dtype(self, numpy_backend):
        a = numpy_backend.asarray([1, 2, 3])
        assert a.dtype == np.float64

    def test_asarray_explicit_dtype(self, numpy_backend):
        a = numpy_backend.asarray([1, 2, 3], dtype=np.float32)
        assert a.dtype == np.float32

    def test_asarray_strips_imag_for_float_dtype(self, numpy_backend):
        c = np.array([1 + 2j, 3 + 4j])
        a = numpy_backend.asarray(c, dtype=np.float64)
        np.testing.assert_array_equal(a, [1.0, 3.0])

    def test_zeros_shape(self, numpy_backend):
        a = numpy_backend.zeros((3, 4))
        assert a.shape == (3, 4)
        assert np.all(a == 0)

    def test_zeros_dtype_string_float(self, numpy_backend):
        a = numpy_backend.zeros((2,), dtype="float")
        assert a.dtype == np.float64

    def test_zeros_dtype_string_complex(self, numpy_backend):
        a = numpy_backend.zeros((2,), dtype="complex")
        assert a.dtype == np.complex128

    def test_zeros_dtype_string_int(self, numpy_backend):
        a = numpy_backend.zeros((2,), dtype="int")
        assert a.dtype == np.int64

    def test_zeros_type_match(self, numpy_backend):
        ref = np.zeros(1, dtype=np.complex64)
        a = numpy_backend.zeros((3,), type_match=ref)
        assert a.dtype == np.complex64

    def test_ones_shape_and_values(self, numpy_backend):
        a = numpy_backend.ones((2, 3))
        assert a.shape == (2, 3)
        assert np.all(a == 1)

    def test_fftfreq_length(self, numpy_backend):
        f = numpy_backend.fftfreq(8, d=1.0)
        expected = np.fft.fftfreq(8, 1.0)
        np.testing.assert_allclose(f, expected)

    def test_randfloats_shape_and_range(self, numpy_backend):
        r = numpy_backend.randfloats(100)
        assert r.shape == (100,)
        assert r.min() >= 0.0
        assert r.max() < 1.0

    def test_memmap_no_filename_returns_zeros(self, numpy_backend):
        a = numpy_backend.memmap((4, 4))
        assert a.shape == (4, 4)
        assert np.all(a == 0)

    def test_memmap_with_filename(self, numpy_backend):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            fname = f.name
        try:
            mm = numpy_backend.memmap((3, 3), dtype=np.float32, filename=fname)
            assert mm.shape == (3, 3)
            assert mm.dtype == np.float32
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# Backend — shared operations (parametrised over both backends)
# ---------------------------------------------------------------------------

class TestBackendOperations:
    def test_astype(self, backend):
        a = backend.asarray([1.0, 2.0])
        b = backend.astype(a, backend.complex_dtype)
        np_b = to_np(b)
        assert np_b.dtype in (np.complex64, np.complex128)

    def test_ones_like(self, backend):
        a = backend.asarray([3.0, 4.0])
        b = backend.ones_like(a)
        np.testing.assert_array_equal(to_np(b), [1.0, 1.0])

    def test_clone_is_independent(self, backend):
        a = backend.asarray([1.0, 2.0, 3.0])
        b = backend.clone(a)
        np_a = to_np(a).copy()
        # Mutate original (numpy); check b is unaffected
        if isinstance(a, np.ndarray):
            a[0] = 99.0
            assert to_np(b)[0] == np_a[0]

    # ------------------------------------------------------------------
    # FFT family
    # ------------------------------------------------------------------

    def test_fft_roundtrip(self, backend):
        data = backend.asarray(np.random.rand(8))
        result = backend.ifft(backend.fft(data))
        np.testing.assert_allclose(to_np(result).real, to_np(data), atol=1e-10)

    def test_fft_with_axis(self, backend):
        # axes kwarg must be translated correctly (axis= for numpy, dim= for torch)
        data = backend.asarray(np.random.rand(4, 8))
        result = backend.ifft(backend.fft(data, axes=1), axes=1)
        np.testing.assert_allclose(to_np(result).real, to_np(data), atol=1e-10)

    def test_fft2_roundtrip(self, backend):
        data = backend.asarray(np.random.rand(8, 8))
        result = backend.ifft2(backend.fft2(data))
        np.testing.assert_allclose(to_np(result).real, to_np(data), atol=1e-10)

    def test_fft2_with_axes(self, backend):
        data = backend.asarray(np.random.rand(4, 8, 8))
        result = backend.ifft2(backend.fft2(data, axes=(-2, -1)), axes=(-2, -1))
        np.testing.assert_allclose(to_np(result).real, to_np(data), atol=1e-10)

    def test_fftshift_ifftshift_roundtrip(self, backend):
        data = backend.asarray(np.arange(8, dtype=np.float64))
        shifted = backend.fftshift(data)
        restored = backend.ifftshift(shifted)
        np.testing.assert_array_equal(to_np(restored), to_np(data))

    def test_fftshift_with_axes(self, backend):
        data = backend.asarray(np.random.rand(4, 8))
        shifted = backend.fftshift(data, axes=1)
        restored = backend.ifftshift(shifted, axes=1)
        np.testing.assert_allclose(to_np(restored), to_np(data), atol=1e-10)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def test_sum_all(self, backend):
        a = backend.asarray([1.0, 2.0, 3.0])
        assert float(to_np(backend.sum(a))) == pytest.approx(6.0)

    def test_sum_axis(self, backend):
        a = backend.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        s = to_np(backend.sum(a, axis=0))
        np.testing.assert_allclose(s, [4.0, 6.0])

    def test_sum_keepdims(self, backend):
        a = backend.asarray(np.ones((3, 4)))
        s = backend.sum(a, axis=1, keepdims=True)
        assert to_np(s).shape == (3, 1)

    def test_mean_all(self, backend):
        a = backend.asarray([2.0, 4.0, 6.0])
        assert float(to_np(backend.mean(a))) == pytest.approx(4.0)

    def test_cumsum(self, backend):
        a = backend.asarray([1.0, 2.0, 3.0])
        cs = to_np(backend.cumsum(a, axis=0))
        np.testing.assert_allclose(cs, [1.0, 3.0, 6.0])

    def test_any_true(self, backend):
        a = backend.asarray([0.0, 1.0, 0.0])
        assert bool(backend.any(a > 0.5))

    def test_any_false(self, backend):
        a = backend.asarray([0.0, 0.0, 0.0])
        assert not bool(backend.any(a > 0.5))

    # ------------------------------------------------------------------
    # Shape manipulation
    # ------------------------------------------------------------------

    def test_reshape(self, backend):
        a = backend.asarray(np.arange(6, dtype=np.float64))
        b = backend.reshape(a, (2, 3))
        assert to_np(b).shape == (2, 3)

    def test_expand_dims(self, backend):
        a = backend.asarray([1.0, 2.0, 3.0])
        b = backend.expand_dims(a, 0)
        assert to_np(b).shape == (1, 3)

    def test_stack(self, backend):
        a = backend.asarray([1.0, 2.0])
        b = backend.asarray([3.0, 4.0])
        s = to_np(backend.stack([a, b], axis=0))
        assert s.shape == (2, 2)
        np.testing.assert_allclose(s[0], [1.0, 2.0])
        np.testing.assert_allclose(s[1], [3.0, 4.0])

    def test_roll(self, backend):
        a = backend.asarray([1.0, 2.0, 3.0, 4.0])
        r = to_np(backend.roll(a, 1, axis=0))
        np.testing.assert_array_equal(r, [4.0, 1.0, 2.0, 3.0])

    # ------------------------------------------------------------------
    # Elementwise math
    # ------------------------------------------------------------------

    def test_absolute(self, backend):
        a = backend.asarray([-1.0, 2.0, -3.0])
        result = to_np(backend.absolute(a))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_sqrt(self, backend):
        a = backend.asarray([4.0, 9.0, 16.0])
        result = to_np(backend.sqrt(a))
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_exp(self, backend):
        a = backend.asarray([0.0, 1.0])
        result = to_np(backend.exp(a))
        np.testing.assert_allclose(result, [1.0, np.e], rtol=1e-6)

    def test_log(self, backend):
        a = backend.asarray([1.0, np.e])
        result = to_np(backend.log(a))
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_real(self, backend):
        c = backend.asarray(np.array([1 + 2j]), dtype=backend.complex_dtype)
        result = to_np(backend.real(c))
        np.testing.assert_allclose(result, [1.0])

    def test_cos(self, backend):
        a = backend.asarray([0.0])
        np.testing.assert_allclose(to_np(backend.cos(a)), [1.0], atol=1e-10)

    def test_isnan(self, backend):
        a = backend.asarray([float("nan"), 1.0])
        mask = to_np(backend.isnan(a))
        assert mask[0] == True
        assert mask[1] == False

    def test_ceil_scalar(self, backend):
        assert backend.ceil(2.3) == 3
        assert isinstance(backend.ceil(2.3), int)

    # ------------------------------------------------------------------
    # Array construction helpers
    # ------------------------------------------------------------------

    def test_arange(self, backend):
        a = to_np(backend.arange(5))
        np.testing.assert_array_equal(a, [0, 1, 2, 3, 4])

    def test_linspace(self, backend):
        a = to_np(backend.linspace(0.0, 1.0, num=5))
        np.testing.assert_allclose(a, np.linspace(0.0, 1.0, 5))

    def test_amin_amax(self, backend):
        a = backend.asarray([3.0, 1.0, 4.0, 1.0, 5.0])
        assert float(to_np(backend.amin(a))) == pytest.approx(1.0)
        assert float(to_np(backend.amax(a))) == pytest.approx(5.0)

    def test_argwhere(self, backend):
        a = backend.asarray([0.0, 1.0, 0.0, 2.0])
        idx = to_np(backend.argwhere(a > 0.5))
        assert set(idx.flatten().tolist()) == {1, 3}

    # ------------------------------------------------------------------
    # Einsum
    # ------------------------------------------------------------------

    def test_einsum_matrix_vector(self, backend):
        A = backend.asarray(np.eye(3))
        v = backend.asarray([1.0, 2.0, 3.0])
        result = to_np(backend.einsum("ij,j->i", A, v))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    # ------------------------------------------------------------------
    # Histogram (always numpy-backed)
    # ------------------------------------------------------------------

    def test_histogram(self, backend):
        a = backend.asarray(np.array([0.5, 1.5, 2.5, 3.5]))
        counts, edges = backend.histogram(a, bins=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        assert isinstance(counts, np.ndarray)
        np.testing.assert_array_equal(counts, [1, 1, 1, 1])

    # ------------------------------------------------------------------
    # pi property
    # ------------------------------------------------------------------

    def test_pi(self, backend):
        assert backend.pi == pytest.approx(np.pi)

    # ------------------------------------------------------------------
    # chunk_ids static method
    # ------------------------------------------------------------------

    def test_chunk_ids_exact_division(self):
        chunks = Backend.chunk_ids(9, chunksize=3)
        assert len(chunks) == 3
        np.testing.assert_array_equal(chunks[0], [0, 1, 2])
        np.testing.assert_array_equal(chunks[2], [6, 7, 8])

    def test_chunk_ids_partial_last_chunk(self):
        chunks = Backend.chunk_ids(10, chunksize=3)
        assert len(chunks) == 4
        np.testing.assert_array_equal(chunks[-1], [9])

    def test_chunk_ids_empty(self):
        assert Backend.chunk_ids(0) == []


# ---------------------------------------------------------------------------
# TorchBackend-specific tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTorchBackend:
    def test_device_defaults_to_cpu(self, monkeypatch):
        monkeypatch.delenv("PYSLICE_DEVICE", raising=False)
        b = TorchBackend(device="cpu")
        assert b.device == torch.device("cpu")

    def test_device_env_override(self, monkeypatch):
        monkeypatch.setenv("PYSLICE_DEVICE", "cpu")
        b = TorchBackend()
        assert b.device.type == "cpu"

    def test_float_dtype_cpu(self, monkeypatch):
        monkeypatch.delenv("PYSLICE_DEVICE", raising=False)
        b = TorchBackend(device="cpu")
        assert b.float_dtype == torch.float64

    def test_complex_dtype_cpu(self, monkeypatch):
        monkeypatch.delenv("PYSLICE_DEVICE", raising=False)
        b = TorchBackend(device="cpu")
        assert b.complex_dtype == torch.complex128

    def test_asarray_returns_tensor(self):
        b = TorchBackend(device="cpu")
        t = b.asarray([1.0, 2.0])
        assert isinstance(t, torch.Tensor)

    def test_asarray_strips_imag(self):
        b = TorchBackend(device="cpu")
        c = np.array([1 + 2j, 3 + 4j])
        t = b.asarray(c, dtype=torch.float64)
        np.testing.assert_array_equal(to_np(t), [1.0, 3.0])

    def test_zeros_returns_tensor(self):
        b = TorchBackend(device="cpu")
        z = b.zeros((2, 3))
        assert isinstance(z, torch.Tensor)
        assert z.shape == torch.Size([2, 3])

    def test_zeros_type_match_numpy_returns_numpy(self):
        b = TorchBackend(device="cpu")
        ref = np.zeros(1, dtype=np.float64)
        z = b.zeros((4,), type_match=ref)
        assert isinstance(z, np.ndarray)

    def test_fftfreq_returns_tensor(self):
        b = TorchBackend(device="cpu")
        f = b.fftfreq(8)
        assert isinstance(f, torch.Tensor)

    def test_randfloats_returns_tensor(self):
        b = TorchBackend(device="cpu")
        r = b.randfloats(50)
        assert isinstance(r, torch.Tensor)
        assert r.shape == torch.Size([50])

    def test_resolve_dtype_string_float(self):
        b = TorchBackend(device="cpu")
        dtype, _ = b._resolve_dtype_device("float", None, None)
        assert dtype == torch.float64

    def test_resolve_dtype_string_complex(self):
        b = TorchBackend(device="cpu")
        dtype, _ = b._resolve_dtype_device("complex", None, None)
        assert dtype == torch.complex128

    def test_resolve_dtype_string_int(self):
        b = TorchBackend(device="cpu")
        dtype, _ = b._resolve_dtype_device("int", None, None)
        assert dtype == torch.int64

    def test_not_available_raises_without_torch(self, monkeypatch):
        import pyslice.backend as bmod
        orig = bmod.TORCH_AVAILABLE
        monkeypatch.setattr(bmod, "TORCH_AVAILABLE", False)
        with pytest.raises(RuntimeError):
            TorchBackend.__new__(TorchBackend)
            # The __init__ check fires
            b = TorchBackend.__new__(TorchBackend)
            b.__init__()
        monkeypatch.setattr(bmod, "TORCH_AVAILABLE", orig)

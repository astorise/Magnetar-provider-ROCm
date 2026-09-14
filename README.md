# magnetar-provider-rocm

## Purpose

An AMD ROCm/HIP execution Provider for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime,
implementing Magnetar's `Provider` contract for AMD Devices -- the ROCm
counterpart to
[`providers/cuda`](https://github.com/astorise/Magnetar-provider-CUDA)'s
CUDA Provider.

## Status

**Real device-discovery skeleton, not a full Provider.** `RocmProvider::new()`
dynamically loads the real HIP runtime shared library (`amdhip64.dll` on
Windows, `libamdhip64.so` on Linux -- no build-time dependency on either,
the same `libloading`-based dynamic-loading posture `providers/cuda`'s own
`cudarc` dependency already establishes for CUDA) and calls the real,
public `hipInit`/`hipGetDeviceCount` HIP API entry points to discover
whether a usable ROCm runtime and device exist. When they do not --
including on every machine and CI runner this crate has actually been
developed and tested on, none of which has a real AMD GPU or ROCm
installation -- it gracefully reports
[`ProviderHealth::Unavailable`](https://github.com/astorise/Magnetar) and
zero Devices, exactly like `CudaProvider` does for CUDA when no compatible
GPU is present.

**What is real and verified**: the graceful-unavailability path itself.
`RocmProvider::new()` constructs successfully and `is_available()`/
`health()` correctly report "no ROCm here" on a genuinely ROCm-less
machine -- confirmed on this crate's own development environment (Windows,
no ROCm) and will be confirmed again by this repository's CI (`ubuntu-latest`,
also genuinely ROCm-less), the same two-environments-agree verification
`providers/cuda`'s own README already establishes for the equivalent CUDA
case.

**What is explicitly not implemented, and why**: no compute Kernels
(`ProviderExecutionApi`), no per-device enumeration beyond a raw count, no
Kernel advertisements. Writing real HIP/hipRTC compute kernels (matmul,
rmsnorm, rope, attention, ...) and claiming they are correct without ever
running them on a real AMD GPU would be unverifiable guesswork -- exactly
the category of confident-but-untested work this repository's own
development practice avoids everywhere else (every other Provider/Kernel
in this workspace is verified against real hardware before being
considered done). This crate stops at the boundary of what can actually be
verified from a ROCm-less development environment: real device discovery,
honestly reporting unavailability when that is the truth.

## What a future contributor with real AMD/ROCm hardware would need to do

1. Extend `src/hip_sys.rs` with the real per-device enumeration API
   (`hipDeviceGet`/`hipGetDeviceProperties` or `hipDeviceGetName`) and
   build real `magnetar_runtime::device::DeviceDescriptor` values from it,
   mirroring `providers/cuda`'s own `device.rs`.
2. Implement `hipModuleLoadData`/hipRTC-based Kernel compilation and
   `ProviderExecutionApi`, mirroring `providers/cuda`'s own
   `CudaKernels`/`CudaExecutor` -- HIP's own API is deliberately CUDA-API-
   compatible in shape, so that existing implementation is the natural
   reference to port from, not a green-field design.
3. Verify every ported Kernel against `providers/cpu`'s reference
   implementation on real AMD hardware, the same `provider-compute`
   conformance profile `providers/cuda` already passes for CUDA.

## Governing contract

Implements Magnetar's generic `Provider`/`Device`/`ProviderExecutionApi`
contracts from the main [Magnetar](https://github.com/astorise/Magnetar)
repository's `magnetar-runtime` crate. No dedicated OpenSpec capability
exists yet for this crate specifically; one should be scoped in the main
repository's `openspec/specs/` once real Kernel execution work here
begins, the same way `providers/cuda`'s own `cuda-provider` capability was.

## Relationship to magnetar-runtime

This Provider is loaded and driven by the Runtime's own Provider registry,
never the reverse -- `magnetar-runtime` has zero compile-time dependency
on this crate (the same externalization invariant every Provider/Component/
Format module in this workspace observes). It is pinned into the main
Magnetar repository as a git submodule at `providers/rocm`.

//! Minimal, hand-declared FFI surface for AMD ROCm's HIP runtime driver
//! API -- only the two stable, well-documented entry points this
//! baseline's device discovery actually calls, matching the real public
//! HIP API signatures (`hip_runtime_api.h`): `hipInit`, `hipGetDeviceCount`.
//! HIP's own driver-style API is deliberately CUDA-driver-API-compatible
//! in shape, matching the same convention `providers/cuda`'s own `cudarc`
//! dependency wraps. Real per-device enumeration (`hipDeviceGet`/
//! `hipDeviceGetName`, or the full `hipGetDeviceProperties` struct) is
//! real future work, not declared here yet -- see this crate's own
//! README for why a device *count* alone is this baseline's whole scope.
//!
//! # Honesty about verification
//!
//! These declarations are believed correct against HIP's real, public,
//! documented API and are exercised for real by [`crate::provider::
//! RocmProvider::new`]'s device-discovery path on every machine this
//! repository's own tests and CI actually run on -- but no machine this
//! crate has been developed or tested on has a real AMD GPU or ROCm
//! runtime installed. The only path genuinely exercised so far is
//! `hipGetDeviceCount`'s absence (the dynamic library fails to load at
//! all, or a real call fails/reports zero devices) -- never a real,
//! successful call against real hardware. See this crate's own README
//! for what would be needed to close that gap.

use std::os::raw::{c_int, c_uint};

/// `hipError_t`'s `hipSuccess` value (`0`) -- the only variant this
/// baseline's device-discovery path distinguishes; any other value is
/// treated uniformly as "discovery failed", matching
/// [`super::provider::RocmProvider`]'s own "no partial credit" graceful-
/// unavailability posture (the same one `providers/cuda`'s `CudaProvider`
/// already establishes for the CUDA driver API).
pub const HIP_SUCCESS: c_int = 0;

/// `hipInit(unsigned int flags)` -- real HIP API, `flags` must currently
/// be `0` (reserved for future use, matching the CUDA driver API's own
/// `cuInit` convention HIP mirrors).
pub type HipInitFn = unsafe extern "C" fn(flags: c_uint) -> c_int;

/// `hipGetDeviceCount(int* count)`.
pub type HipGetDeviceCountFn = unsafe extern "C" fn(count: *mut c_int) -> c_int;

/// Real HIP runtime shared library file names this crate attempts to
/// dynamically load, in order, per platform -- the real, documented
/// install-time artifact names a real ROCm installation provides. No
/// build-time dependency on any of them (`libloading` resolves purely at
/// runtime), matching `providers/cuda`'s own dynamic-loading posture.
#[cfg(target_os = "windows")]
pub const HIP_LIBRARY_CANDIDATES: &[&str] = &["amdhip64.dll"];
#[cfg(target_os = "linux")]
pub const HIP_LIBRARY_CANDIDATES: &[&str] =
    &["libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"];
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
pub const HIP_LIBRARY_CANDIDATES: &[&str] = &[];

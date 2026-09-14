//! `RocmProvider`: a real device-discovery skeleton for AMD ROCm/HIP,
//! deliberately not yet an optimized, kernel-executing Provider.
//!
//! # Status: device discovery only, no compute kernels
//!
//! Unlike `providers/cuda`'s `CudaProvider` (a full, real-hardware-
//! verified Provider), this crate implements only the "does a compatible
//! ROCm runtime and device exist" half of that same pattern --
//! `RocmProvider::new()` always constructs successfully and gracefully
//! reports [`ProviderHealth::Unavailable`] when no ROCm runtime is found,
//! exactly like `CudaProvider` does for CUDA, but it advertises zero
//! Kernels and zero real compute capability even when a device *is*
//! found. Real Kernel execution (the Operator dispatch surface
//! `providers/cpu`/`providers/cuda` both implement) needs real AMD GPU
//! hardware to write and verify against, which no machine this crate has
//! been developed on has. Implementing that is real, tracked future work
//! for whoever has that hardware -- not attempted here as unverifiable
//! guesswork.
//!
//! # Graceful unavailability
//!
//! `RocmProvider::new()` never fails Runtime initialization for the
//! absence of hardware or the ROCm runtime shared library: it attempts to
//! dynamically load the real HIP runtime library (`hip_sys.rs`'s
//! `HIP_LIBRARY_CANDIDATES`) and call `hipInit`/`hipGetDeviceCount`;
//! any failure at any step (library not found, symbol not found, a real
//! HIP call reporting an error, or zero devices found) is treated
//! uniformly as "no usable ROCm device" -- [`ProviderHealth::Unavailable`],
//! not a construction error. This is also what keeps this crate's
//! `cargo test` green on every machine and CI runner this repository
//! currently has (all genuinely ROCm-less): the only thing that can fail
//! is the runtime `dlopen` attempt, which this module already treats as a
//! normal, expected outcome, the same posture `providers/cuda`'s
//! `CudaProvider` established first.

use magnetar_runtime::affinity::ProviderHealth;
use magnetar_runtime::provider::{Provider, ProviderError, ProviderMetadata, ProviderRegistry};

use crate::hip_sys::{HIP_LIBRARY_CANDIDATES, HIP_SUCCESS, HipGetDeviceCountFn, HipInitFn};

const ROCM_PROVIDER_NAME: &str = "rocm";
const ROCM_PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");
const ROCM_PROVIDER_VENDOR: &str = "AMD";

pub fn rocm_provider_metadata() -> ProviderMetadata {
    ProviderMetadata::new(
        ROCM_PROVIDER_NAME,
        ROCM_PROVIDER_VERSION,
        ROCM_PROVIDER_VENDOR,
        "Device-discovery skeleton for the AMD ROCm/HIP runtime -- no compute \
         Kernels implemented yet (see this crate's own README)",
    )
}

/// The ROCm Provider itself. Registers as a built-in Provider, matching
/// `CudaProvider`'s own posture (no dynamic-library Provider ABI in this
/// baseline).
pub struct RocmProvider {
    metadata: ProviderMetadata,
    /// `Some(count)` only when a real HIP runtime was dynamically loaded,
    /// `hipInit` succeeded, and `hipGetDeviceCount` reported at least one
    /// device. `count` itself is currently unused beyond "is it
    /// positive" -- no per-device descriptor is built yet (see this
    /// module's own doc comment: device discovery only, not full Device
    /// enumeration).
    device_count: Option<i32>,
}

impl RocmProvider {
    pub fn new() -> Self {
        Self {
            metadata: rocm_provider_metadata(),
            device_count: Self::discover_device_count(),
        }
    }

    /// Attempts to dynamically load a real HIP runtime library and query
    /// its real device count. Returns `None` for *any* failure along the
    /// way (library not found, symbol not found, a real HIP call
    /// reporting an error, or zero devices) -- this baseline draws no
    /// distinction between those cases; they are all "no usable ROCm
    /// device here" to a caller.
    fn discover_device_count() -> Option<i32> {
        let library = HIP_LIBRARY_CANDIDATES
            .iter()
            .find_map(|name| unsafe { libloading::Library::new(name).ok() })?;

        // SAFETY: `hipInit`/`hipGetDeviceCount` are real, stable, publicly
        // documented HIP runtime API entry points (`hip_sys.rs`); the
        // symbol lookups below only succeed against a real HIP runtime
        // library actually exporting them under these exact names, and
        // the function-pointer types declared in `hip_sys.rs` match HIP's
        // own real C signatures.
        unsafe {
            let hip_init: libloading::Symbol<HipInitFn> = library.get(b"hipInit\0").ok()?;
            if hip_init(0) != HIP_SUCCESS {
                return None;
            }
            let hip_get_device_count: libloading::Symbol<HipGetDeviceCountFn> =
                library.get(b"hipGetDeviceCount\0").ok()?;
            let mut count: std::os::raw::c_int = 0;
            if hip_get_device_count(&mut count) != HIP_SUCCESS {
                return None;
            }
            if count <= 0 { None } else { Some(count) }
        }
    }

    /// Whether this Provider found a usable ROCm runtime and at least one
    /// device.
    pub fn is_available(&self) -> bool {
        self.device_count.is_some()
    }
}

impl Default for RocmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for RocmProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        // No Devices to register yet -- this baseline does not build a
        // real `DeviceDescriptor` even when a device is found (see this
        // module's own "Status" doc comment); `devices()` keeps the
        // `Provider` trait's own empty-`Vec` default.
        Ok(())
    }

    fn health(&self) -> ProviderHealth {
        if self.is_available() {
            ProviderHealth::Available
        } else {
            ProviderHealth::Unavailable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one thing genuinely verified on every machine this crate has
    /// actually run on (this development environment, and CI's own
    /// ROCm-less `submodule-integration` runner): a machine with no real
    /// ROCm runtime installed gets a Provider that constructs
    /// successfully and honestly reports itself unavailable, never a
    /// panic or a false "available".
    #[test]
    fn reports_unavailable_without_a_real_rocm_runtime() {
        let provider = RocmProvider::new();
        assert!(
            !provider.is_available(),
            "this test environment has no real ROCm runtime installed"
        );
        assert_eq!(provider.health(), ProviderHealth::Unavailable);
    }

    #[test]
    fn metadata_is_well_formed() {
        let provider = RocmProvider::new();
        let metadata = provider.metadata();
        assert_eq!(metadata.name, ROCM_PROVIDER_NAME);
        assert_eq!(metadata.vendor, ROCM_PROVIDER_VENDOR);
        assert!(!metadata.description.is_empty());
    }

    #[test]
    fn devices_are_empty_without_a_real_rocm_runtime() {
        let provider = RocmProvider::new();
        assert!(provider.devices().is_empty());
    }
}

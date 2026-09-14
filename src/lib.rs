//! AMD ROCm/HIP execution Provider for the Magnetar local AI Runtime.
//!
//! **Status: real device-discovery skeleton, no compute Kernels yet.**
//! See [`provider::RocmProvider`]'s own module doc comment and this
//! crate's README for what that means and why.

mod hip_sys;
mod provider;

pub use provider::{RocmProvider, rocm_provider_metadata};

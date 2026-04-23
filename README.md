# Basis-space Spectral Path Tracing (BsSPT)

This repository contains a high-performance research engine for spectral path tracing in coefficient space. The system represents spectral data as coefficients of a discretized basis, allowing for efficient light transport simulations using matrix-vector operations.

## Core Architecture

The engine is built on the concept of projecting continuous spectral distributions into a discrete basis. This approach transforms complex spectral interactions into linear or affine operators in coefficient space.

### Spectral Bases
The primary basis used is the Gaussian-Hermite Global Spectral Function (GHGSF).
*   Standard Basis: A multi-lobe basis with uniform bandwidth.
*   Dual Domain Basis: A bleeding-edge implementation supporting independent wide and narrow lobe domains with configurable scaling schedules.

### Operators and States
Spectral interactions are modeled as affine transformations: O(alpha) = A * alpha + b.
*   Absorption: Implements Beer-Lambert law integration in basis space.
*   Reflectance: Handles wavelength-dependent BSDF interactions.
*   Emission: Projects light source spectra into the coefficient domain.
*   Localization: Extracts wavelength-localized energy packets for dispersive effects.

## Project Structure

The codebase is organized into the research package:

*   research/engine: Contains the core mathematical logic for spectral domains, bases, operators, and states.
*   research/plot: A production-grade scientific plotting system for research dark-themed visualizations.
*   tests: Verification scripts to ensure numerical stability and refactor integrity.

## Usage

Ensure the Spectral environment is active before running scripts. Use the included check_env.py script to verify that PyTorch and CUDA are correctly configured.

```bash
python check_env.py
python tests/verify_refactor.py
```

All implementation follows a professional naming convention with PascalCase for classes and camelCase for methods and members.

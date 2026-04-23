# Phase 1: Spectral Stability Formalization

This document outlines the formal architecture and execution pipeline for Phase 1 of the Basis-space Spectral Path Tracing research. The primary objective of this phase is to identify and formalize the optimal basis configurations for high-performance spectral reconstruction.

## 1. Core Objectives
The research focuses on the stability and expressiveness of the GHGSF Dual Domain basis across a massive configuration space. Key goals include:
- Mapping the stability landscape across varying lobe counts and polynomial orders.
- Investigating the effect of spectral boundary margins on Hermite ringing artifacts.
- Identifying rescued configurations where the whitening transform enables extreme resolution.
- Establishing a golden zone of parameters for production path tracer integration.

## 2. Technical Architecture
The pipeline is implemented as a modular Python package located in the research directory.

### 2.1 Research Engine
- SpectralDomain: Centralized domain management with persistent integration weights.
- GHGSFDualDomainBasis: The frontier basis implementation supporting wide and narrow lobe scheduling.
- SpectralOperatorFactory: Unified logic for Cholesky decomposition and operator construction.

### 2.2 Numerical Configuration
- Precision: Pure Float64 (Reference Mode) for all stability benchmarks.
- Domain: 4096 samples across the 380nm to 830nm visible spectrum.
- Scaling: Support for constant, linear, square root, and power scaling laws.

## 3. Execution Pipeline
The formalization is executed through a five stage automated pipeline managed by phase1_master.py.

### Stage 1: Stability Sweep
The sweep script generates a 10.5 million configuration dataset. It computes both raw and whitened stability metrics in a single fused pass to minimize computational overhead.

### Stage 2: Result Partitioning
The master dataset is partitioned into three subsets based on the spectral margin (0nm, 10nm, and 20nm). This allows for isolated study of boundary condition effects.

### Stage 3: SPD Categorization
Successful configurations are filtered into two distinct research sets:
- Naturally Stable: Configurations that are mathematically sound in their raw state.
- Rescued: Configurations that require a whitening transform to achieve stability.

### Stage 4: Visual Analysis
The plotting suite generates 2D heatmaps of Lobe Count vs. Polynomial Order. These maps visualize the log-condition number and spectral entropy to identify regions of high stability.

### Stage 5: Golden Zone Selection
A two-pass analysis script ranks candidates based on their spectral resolution and information density. This stage identifies the final parameters for engine integration.

## 4. Usage
To execute the complete Phase 1 formalization, run the following command from the project root:

python phase1_master.py

Results and visualizations are stored in the results and plots directories respectively.

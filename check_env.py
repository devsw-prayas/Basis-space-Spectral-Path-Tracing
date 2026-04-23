import sys
import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
try:
    import pyarrow
except ImportError:
    pyarrow = None
try:
    import pandas
except ImportError:
    pandas = None

def checkEnv():
    print("═══════════════════════════════════════════════════════════════")
    print(" BsSPT Environment Check")
    print("═══════════════════════════════════════════════════════════════")

    # 1. System Info
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Platform       : {sys.platform}")

    # 2. Dependencies
    print("\n[Dependencies]")
    print(f"PyTorch        : {torch.__version__} {'✓' if torch.__version__ else '✗'}")
    print(f"NumPy          : {np.__version__} {'✓' if np.__version__ else '✗'}")
    print(f"Matplotlib     : {matplotlib.__version__} {'✓' if matplotlib.__version__ else '✗'}")
    print(f"PyArrow        : {pyarrow.__version__ if pyarrow else 'MISSING'} {'✓' if pyarrow else '✗'}")
    print(f"Pandas         : {pandas.__version__ if pandas else 'MISSING'} {'✓' if pandas else '✗'}")

    # 3. Hardware Acceleration
    print("\n[Hardware]")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available : {'YES ✓' if cuda_available else 'NO ✗'}")
    if cuda_available:
        print(f"GPU Device     : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version   : {torch.version.cuda}")

    # 4. Local Package Integrity
    print("\n[Package Integrity]")
    try:
        from research.engine.domain import SpectralDomain
        from research.engine.basis import GHGSFDualDomainBasis
        from research.plot.engine import PlotEngine
        print("research.engine : OK ✓")
        print("research.plot   : OK ✓")
    except ImportError as e:
        print(f"Package Error  : {e} ✗")
        print("Make sure you are running from the projectSo root.")

    print("\n═══════════════════════════════════════════════════════════════")
    if cuda_available:
        print(" Environment is ready for high-performance spectral R&D.")
    else:
        print(" Environment is ready (CPU-only).")
    print("═══════════════════════════════════════════════════════════════")

if __name__ == "__main__":
    checkEnv()

import torch

class TorchConfig:
    @staticmethod
    def resolveDevice() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def setMode(
        mode: str,
        device: torch.device = None,
        verbose: bool = False
    ) -> dict:
        """
        Configure torch for either 'performance' (float32) or 'reference' (float64).
        """
        if mode not in ["performance", "reference"]:
            raise ValueError("Mode must be 'performance' or 'reference'.")

        if device is None:
            device = TorchConfig.resolveDevice()

        torch.set_grad_enabled(False)

        if mode == "performance":
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            dtype = torch.float32
            if verbose:
                print(f"Torch set to PERFORMANCE mode (TF32) on {device}.")
        else:
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            dtype = torch.float64
            if verbose:
                print(f"Torch set to REFERENCE mode (FP64) on {device}.")

        return {
            "device": device,
            "dtype": dtype
        }

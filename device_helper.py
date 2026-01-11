def get_device():
    try:
        import torch_directml
        dev = torch_directml.device()
        print("[DEVICE] Using DirectML device:", dev)
        return dev, "directml"
    except Exception:
        import torch
        if torch.cuda.is_available():
            print("[DEVICE] Using CUDA device")
            return torch.device("cuda"), "cuda"
        else:
            print("[DEVICE] Using CPU")
            return torch.device("cpu"), "cpu"
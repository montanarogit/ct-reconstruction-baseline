from pathlib import Path
import numpy as np
import pydicom
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

def load_ct_slice(filepath: str | None = None, size: int = 512) -> np.ndarray:
    if filepath and Path(filepath).exists():
        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        if img.shape != (size, size):
            img = resize(img, (size, size), anti_aliasing=True).astype(np.float32)
        return img
    else:
        print("[INFO] Nessun file DICOM trovato. Uso Shepp-Logan phantom 2D.")
        phantom = shepp_logan_phantom()
        return resize(phantom, (size, size), anti_aliasing=True).astype(np.float32)

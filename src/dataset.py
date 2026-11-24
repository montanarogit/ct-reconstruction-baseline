from pathlib import Path
import numpy as np
import pydicom
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

def load_ct_slice(filepath: str | None = None, size: int = 512) -> tuple[np.ndarray, float]:
    if filepath and Path(filepath).exists():
        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array.astype(np.float32)

        # Applica RescaleSlope e RescaleIntercept se presenti nei metadati DICOM per avere i veri HU
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        hu_min, hu_max = float(np.min(img)), float(np.max(img))
        hu_range = hu_max - hu_min + 1e-8

        # Normalizzazione [0, 1]
        img_norm = (img - hu_min) / hu_range

        if img_norm.shape != (size, size):
            img_norm = resize(img_norm, (size, size), anti_aliasing=True).astype(np.float32)
        return img_norm, hu_range
    else:
        print("[INFO] Nessun file DICOM trovato. Uso Shepp-Logan phantom 2D.")
        phantom = shepp_logan_phantom()
        return resize(phantom, (size, size), anti_aliasing=True).astype(np.float32), 1000.0

from pathlib import Path
import numpy as np
import pydicom
from skimage.data import shepp_logan_phantom
from skimage.transform import resize



#questo modulo invece gestisce la preparazione dei dati;
#legge la scansione medica con pydicom la normalizza in una scala numerica
#tra 0 e 1 -->
# (schiacciare o nero o bianco i pixel)
# e la ridimensiona a 512x512 pixel, se necessario. Se non trova il file DICOM, 
# genera un fantoccio sintetico Shepp-Logan per i test.
def load_ct_slice(filepath: str | None = None, size: int = 512) -> np.ndarray:
    # 1. LETTURA: Se il file DICOM esiste, lo apre e estrae la matrice grezza dei pixel in float32
    if filepath and Path(filepath).exists():
        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array.astype(np.float32)

        # 2. NORMALIZZAZIONE: Schiaccia tutti i valori tra 0 e 1 (il +1e-8 evita divisioni per zero)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        # 3. RESIZE: Uniforma tutte le immagini alla stessa dimensione (es. 512x512) con anti-aliasing
        if img.shape != (size, size):
            img = resize(img, (size, size), anti_aliasing=True).astype(np.float32)
        return img
    else:
        # 4. FALLBACK DI SICUREZZA: Se mancano i file reali, crea un fantoccio sintetico standard per i test
        print("[INFO] Nessun file DICOM trovato. Uso Shepp-Logan phantom 2D.")
        phantom = shepp_logan_phantom()
        return resize(phantom, (size, size), anti_aliasing=True).astype(np.float32)

import io
import zipfile
from pathlib import Path
import requests

SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
TCIA_API_URL = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={SERIES_UID}"

def download_lidc_sample(output_dir: str = "data") -> str:
    data_path = Path(output_dir)
    data_path.mkdir(exist_ok=True)
    target_file = data_path / "sample_slice.dcm"

    if target_file.exists():
        print(f"[INFO] File DICOM già presente in {target_file}")
        return str(target_file)

    print("[INFO] Download di una scansione LIDC-IDRI da TCIA in corso (circa 60MB)...")
    response = requests.get(TCIA_API_URL, stream=True, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        dcm_files = sorted([f for f in z.namelist() if f.endswith(".dcm")])
        middle_slice = dcm_files[len(dcm_files) // 2]
        print(f"[INFO] Trovate {len(dcm_files)} fette 2D. Estraggo la fetta centrale: {middle_slice}")
        
        with z.open(middle_slice) as source, open(target_file, "wb") as target:
            target.write(source.read())

    print(f"[OK] Fetta CT 2D salvata con successo in: {target_file}")
    return str(target_file)

if __name__ == "__main__":
    download_lidc_sample()

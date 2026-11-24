from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.dataset import load_ct_slice
from src.astra_sim import simulate_and_reconstruct
from src.metrics import CTMetricsEvaluator

def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_torch_device()
    print(f"[INFO] PyTorch in esecuzione su device: {device}")

    Path("outputs").mkdir(exist_ok=True)
    
    # Lavoriamo su un singolo file DICOM di riferimento
    dicom_path = "data/paziente_1.dcm"
    if not Path(dicom_path).exists():
        print(f"[ERRORE] File non trovato: {dicom_path}")
        return

    evaluator = CTMetricsEvaluator(device=device)

    # Tutte le combinazioni della to-do list:
    # (Tipo_Sparsità, Geometria, N_Angoli, Arco_Rotazione_Gradi)
    configurazioni = [
        ("Full-View", "parallel", 360, 180.0),
        ("Sparse-View", "parallel", 60, 180.0),
        ("Limited-Angle", "parallel", 60, 90.0),
        ("Full-View", "fanbeam", 360, 360.0),
        ("Sparse-View", "fanbeam", 60, 360.0),
        ("Limited-Angle", "fanbeam", 60, 180.0),
    ]

    tipi_rumore = ["none", "gaussian", "poisson"]
    risultati_excel = []

    nome_paziente = os.path.basename(dicom_path).replace(".dcm", "")
    ground_truth, hu_range = load_ct_slice(dicom_path, size=512)
    print(f"\n=== Benchmark su {nome_paziente} (Range HU: {hu_range:.1f}) ===")

    for tipo_sparsita, geom_type, num_angles, max_angle in configurazioni:
        for noise in tipi_rumore:
            print(f"-> {geom_type.upper()} | {tipo_sparsita} ({num_angles}v, {max_angle}°) | Rumore: {noise}")

            sinogram, recon_fbp = simulate_and_reconstruct(
                ground_truth,
                geom_type=geom_type,
                num_angles=num_angles,
                max_angle_deg=max_angle,
                noise_type=noise,
                algorithm="FBP"
            )

            metriche = evaluator.evaluate(ground_truth, recon_fbp, hu_range=hu_range)

            risultati_excel.append({
                "Paziente": nome_paziente,
                "Tipo Geometria": geom_type,
                "Sparsità": tipo_sparsita,
                "N. Angoli": num_angles,
                "Arco (°)": max_angle,
                "Rumore": noise,
                "MSE (HU^2)": metriche["MSE (HU^2)"],
                "RMSE (HU)": metriche["RMSE (HU)"],
                "PSNR (dB)": metriche["PSNR (dB)"],
                "SSIM": metriche["SSIM"],
                "LPIPS": metriche["LPIPS"]
            })

            # Creazione grafico comparativo a 3 pannelli
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(ground_truth, cmap="gray")
            axes[0].set_title(f"Ground Truth ({nome_paziente})")
            axes[0].axis("off")

            axes[1].imshow(sinogram, cmap="gray", aspect="auto")
            axes[1].set_title(f"Sino ({geom_type}, {num_angles}v, {int(max_angle)}°, {noise})")

            axes[2].imshow(np.clip(recon_fbp, 0, 1), cmap="gray")
            axes[2].set_title(
                f"RMSE:{metriche['RMSE (HU)']}HU | PSNR:{metriche['PSNR (dB)']}dB\n"
                f"SSIM:{metriche['SSIM']} | LPIPS:{metriche['LPIPS']}"
            )
            axes[2].axis("off")

            plt.tight_layout()
            
            # Salva in sottocartelle ordinate (outputs/parallel/ e outputs/fanbeam/) con nomi brevi
            cartella_out = Path(f"outputs/{geom_type}")
            cartella_out.mkdir(parents=True, exist_ok=True)
            nome_file = f"{tipo_sparsita}_{num_angles}v_{int(max_angle)}deg_{noise}.png"
            plt.savefig(cartella_out / nome_file, dpi=200)
            plt.close(fig)

    # Salva la tabella Excel finale
    df = pd.DataFrame(risultati_excel)
    excel_path = "outputs/tabella_benchmark_fbp.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"\n[OK] Tabella Excel creata in: {excel_path}\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

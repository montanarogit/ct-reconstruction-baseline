from pathlib import Path
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset import load_ct_slice
from src.astra_sim import simulate_and_reconstruct

def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_torch_device()
    print(f"[INFO] PyTorch in esecuzione su device: {device}")

    # Assicuriamoci che la cartella outputs esista
    Path("outputs").mkdir(exist_ok=True)

    # Trova automaticamente tutti i file .dcm dentro la cartella data/
    tutti_i_pazienti = sorted(glob.glob("data/*.dcm"))

    if not tutti_i_pazienti:
        print("[ERRORE] Nessun file .dcm trovato nella cartella data/!")
        return

    num_angles = 360

    # Ciclo che elabora ogni paziente trovato
    for dicom_path in tutti_i_pazienti:
        nome_paziente = os.path.basename(dicom_path).replace(".dcm", "")
        print(f"\n--- Elaborazione {nome_paziente} ({dicom_path}) ---")

        # 1. Carica la fetta DICOM
        ground_truth = load_ct_slice(dicom_path, size=512)

        # 2. Simula acquisizione e ricostruzione con ASTRA
        print(f"[INFO] Simulazione acquisizione CT (Parallel Beam, {num_angles} angoli)...")
        sinogram, recon_fbp = simulate_and_reconstruct(ground_truth, num_angles=num_angles, algorithm="FBP")

        # 3. Calcola MSE su PyTorch (MPS)
        recon_tensor = torch.from_numpy(recon_fbp).unsqueeze(0).unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0).unsqueeze(0).to(device)

        mse_loss = torch.nn.functional.mse_loss(recon_tensor, gt_tensor)
        print(f"[RISULTATO] MSE per {nome_paziente} ({num_angles} viste): {mse_loss.item():.6f}")

        # 4. Crea e salva il grafico specifico per questo paziente
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(ground_truth, cmap="gray")
        axes[0].set_title(f"Ground Truth ({nome_paziente})")
        axes[0].axis("off")

        axes[1].imshow(sinogram, cmap="gray", aspect="auto")
        axes[1].set_title(f"Sinogramma ({num_angles} angoli)")
        axes[1].set_xlabel("Posizione Detector")
        axes[1].set_ylabel("Angolo di proiezione")

        axes[2].imshow(np.clip(recon_fbp, 0, 1), cmap="gray")
        axes[2].set_title(f"Ricostruzione FBP (MSE: {mse_loss.item():.4f})")
        axes[2].axis("off")

        plt.tight_layout()
        output_file = f"outputs/risultato_{nome_paziente}.png"
        plt.savefig(output_file, dpi=300)
        plt.close(fig)  # Chiude la figura per liberare la memoria RAM tra un paziente e l'altro
        print(f"[INFO] Grafico salvato in {output_file}")

if __name__ == "__main__":
    main()
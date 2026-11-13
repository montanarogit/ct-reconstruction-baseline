#prende in automatico tutte le scansioni CT dalla cartella data/
#le passa al soimulatore fisico ASTRA e ricostruisce le immagini con FBP algoritmo
#calcola l'errore matematico mse sfruttando la gpu del mio mac tramite pytorch 
#pytorch è un framework di deep learning che permette di fare calcoli numerici su gpu e cpu in maniera semplice e veloce
#infine salva un grafico a 3 pannelli (originale | sinogramma | ricostruzione fbp) per ogni scansione CT



#FBP èp un algoritmo classico di ricostruzione ct che prende i dati raccolit
#da ogni angolazione e li spalma lungo la direzione di acquisizione, poi applica un filtro per rimuovere 
# gli artefatti e ricostruire l'immagine finale



from pathlib import Path
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset import load_ct_slice
from src.astra_sim import simulate_and_reconstruct

def get_torch_device() -> torch.device:
    # 1. HARDWARE: Sceglie in automatico la GPU del Mac (mps) o NVIDIA (cuda) per velocizzare PyTorch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_torch_device()
    print(f"[INFO] PyTorch in esecuzione su device: {device}")

    Path("outputs").mkdir(exist_ok=True)

    # 2. DATASET: Pesca automaticamente tutti i file medici (.dcm) dalla cartella data/
    tutti_i_pazienti = sorted(glob.glob("data/*.dcm"))

    if not tutti_i_pazienti:
        print("[ERRORE] Nessun file .dcm trovato nella cartella data/!")
        return

    # 3. PARAMETRO CHIAVE: 360 = alta dose (ottima qualità) | 30/60 = bassa dose (sparse-view con artefatti)
    num_angles = 360

    for dicom_path in tutti_i_pazienti:
        nome_paziente = os.path.basename(dicom_path).replace(".dcm", "")
        print(f"\n--- Elaborazione {nome_paziente} ({dicom_path}) ---")

        # Carica la fetta originale normalizzata tra 0 e 1 (il nostro riferimento perfetto)
        ground_truth = load_ct_slice(dicom_path, size=512)

        # 4. SIMULAZIONE FISICA (ASTRA): Genera il sinogramma (raggi X) e ricostruisce con l'algoritmo classico FBP
        print(f"[INFO] Simulazione acquisizione CT (Parallel Beam, {num_angles} angoli)...")
        sinogram, recon_fbp = simulate_and_reconstruct(ground_truth, num_angles=num_angles, algorithm="FBP")

        # 5. VALUTAZIONE (PyTorch): Trasforma in tensori su GPU e calcola l'errore (MSE) rispetto all'originale
        recon_tensor = torch.from_numpy(recon_fbp).unsqueeze(0).unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0).unsqueeze(0).to(device)

        mse_loss = torch.nn.functional.mse_loss(recon_tensor, gt_tensor)
        print(f"[RISULTATO] MSE per {nome_paziente} ({num_angles} viste): {mse_loss.item():.6f}")

        # Salva il confronto visivo a 3 pannelli (Originale | Sinogramma | Ricostruzione FBP)
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
        plt.close(fig)
        print(f"[INFO] Grafico salvato in {output_file}")

if __name__ == "__main__":
    main()

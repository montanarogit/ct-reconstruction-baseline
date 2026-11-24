import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class CTMetricsEvaluator:
    def __init__(self, device: torch.device):
        self.device = device
        # Carica la rete AlexNet pre-addestrata per la metrica percettiva LPIPS
        self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)

    def evaluate(self, gt: np.ndarray, recon: np.ndarray, hu_range: float = 1000.0) -> dict:
        recon_clipped = np.clip(recon, 0.0, 1.0)

        # 1. MSE (HU^2) e RMSE (HU) riportati alla scala fisica Hounsfield
        gt_hu = gt * hu_range
        recon_hu = recon_clipped * hu_range

        gt_t = torch.from_numpy(gt_hu).to(self.device)
        recon_t = torch.from_numpy(recon_hu).to(self.device)

        mse_hu = torch.nn.functional.mse_loss(recon_t, gt_t).item()
        rmse_hu = float(np.sqrt(mse_hu))

        # 2. PSNR (dB) e 3. SSIM (calcolati sull'immagine normalizzata [0, 1])
        psnr_val = float(peak_signal_noise_ratio(gt, recon_clipped, data_range=1.0))
        ssim_val = float(structural_similarity(gt, recon_clipped, data_range=1.0))

        # 4. LPIPS (richiede tensori in [-1, 1] con 3 canali [1, 3, H, W])
        gt_norm_t = torch.from_numpy(gt).to(self.device)
        recon_norm_t = torch.from_numpy(recon_clipped).to(self.device)

        gt_lpips = (gt_norm_t * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        recon_lpips = (recon_norm_t * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

        with torch.no_grad():
            lpips_val = self.lpips_fn(gt_lpips, recon_lpips).item()

        return {
            "MSE (HU^2)": round(mse_hu, 2),
            "RMSE (HU)": round(rmse_hu, 2),
            "PSNR (dB)": round(psnr_val, 2),
            "SSIM": round(ssim_val, 4),
            "LPIPS": round(lpips_val, 4)
        }

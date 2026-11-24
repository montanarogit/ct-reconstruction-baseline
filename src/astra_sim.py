import astra
import numpy as np

def add_sinogram_noise(sinogram: np.ndarray, noise_type: str = "none") -> np.ndarray:
    """Applica rumore Gaussiano (elettrico) o di Poisson (scattering) al sinogramma y = K(x) + e"""
    if noise_type == "none":
        return sinogram
    
    sino_noisy = sinogram.copy()
    
    if noise_type == "gaussian":
        sigma = 0.02 * (np.max(sino_noisy) - np.min(sino_noisy))
        noise = np.random.normal(0, sigma, sino_noisy.shape).astype(np.float32)
        return sino_noisy + noise
        
    elif noise_type == "poisson":
        sino_max = np.max(sino_noisy) + 1e-8
        sino_norm = np.clip(sino_noisy / sino_max, 0, None)
        I0 = 1e4
        photons = np.random.poisson(I0 * np.exp(-sino_norm)).astype(np.float32)
        photons = np.maximum(photons, 1.0)
        return (-np.log(photons / I0) * sino_max).astype(np.float32)
        
    return sino_noisy

def simulate_and_reconstruct(
    image: np.ndarray,
    geom_type: str = "parallel",
    num_angles: int = 360,
    max_angle_deg: float = 180.0,
    noise_type: str = "none",
    algorithm: str = "FBP"
):
    rows, cols = image.shape
    vol_geom = astra.create_vol_geom(rows, cols)

    max_angle_rad = np.deg2rad(max_angle_deg)
    angles = np.linspace(0, max_angle_rad, num_angles, endpoint=False)
    num_detectors = int(np.ceil(np.sqrt(rows**2 + cols**2)))

    if geom_type == "parallel":
        proj_geom = astra.create_proj_geom("parallel", 1.0, num_detectors, angles)
        proj_id = astra.create_projector("linear", proj_geom, vol_geom)
        actual_alg = algorithm
        iterations = 1 if actual_alg == "FBP" else 30
    elif geom_type == "fanbeam":
        source_origin = rows * 2.0
        origin_det = rows * 1.0
        proj_geom = astra.create_proj_geom("fanflat", 1.0, num_detectors, angles, source_origin, origin_det)
        proj_id = astra.create_projector("line_fanflat", proj_geom, vol_geom)
        # Su ASTRA CPU (macOS), FBP supporta solo parallel; per fanbeam usiamo CGLS (o SIRT)
        actual_alg = "CGLS" if algorithm == "FBP" else algorithm
        iterations = 25
    else:
        raise ValueError(f"Geometria non supportata: {geom_type}")

    # Forward Projection: K(x) = y
    sinogram_id, sinogram = astra.create_sino(image, proj_id)

    # Aggiunta rumore: y = K(x) + e
    sinogram_noisy = add_sinogram_noise(sinogram, noise_type=noise_type)
    astra.data2d.store(sinogram_id, sinogram_noisy)

    # Ricostruzione
    rec_id = astra.data2d.create("-vol", vol_geom)
    cfg = astra.astra_dict(actual_alg)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ProjectorId"] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)
    reconstruction = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return sinogram_noisy, reconstruction

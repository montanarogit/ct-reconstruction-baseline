import astra
import numpy as np

def simulate_and_reconstruct(image: np.ndarray, num_angles: int = 180, algorithm: str = "FBP"):
    rows, cols = image.shape
    vol_geom = astra.create_vol_geom(rows, cols)

    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    num_detectors = int(np.ceil(np.sqrt(rows**2 + cols**2)))
    proj_geom = astra.create_proj_geom("parallel", 1.0, num_detectors, angles)

    proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    sinogram_id, sinogram = astra.create_sino(image, proj_id)

    rec_id = astra.data2d.create("-vol", vol_geom)

    cfg = astra.astra_dict(algorithm)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ProjectorId"] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1 if algorithm == "FBP" else 50)

    reconstruction = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return sinogram, reconstruction

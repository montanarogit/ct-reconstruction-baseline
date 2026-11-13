import astra
import numpy as np

#questo e il simulatore fisico vero e proprii:
#imposta una geometria a fasci paralleli,spara i raggi x
#sull'immagine originale generando il sinogramma,
#poi ricostruisce l'immagine con l'algoritmo classico FBP

def simulate_and_reconstruct(image: np.ndarray, num_angles: int = 180, algorithm: str = "FBP"):
    # 1. GEOMETRIA DEL PAZIENTE: Definisce lo spazio 2D dell'immagine (512x512)
    rows, cols = image.shape
    vol_geom = astra.create_vol_geom(rows, cols)

    # 2. GEOMETRIA DEL MACCHINARIO TAC: Crea gli angoli (da 0 a 180 gradi) e calcola la larghezza del sensore
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    num_detectors = int(np.ceil(np.sqrt(rows**2 + cols**2)))
    proj_geom = astra.create_proj_geom("parallel", 1.0, num_detectors, angles)

    # 3. ACQUISIZIONE (Forward Projection): Spara i raggi sull'immagine e genera il Sinogramma
    proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    sinogram_id, sinogram = astra.create_sino(image, proj_id)

    # 4. CONFIGURAZIONE RICOSTRUZIONE: Prepara lo spazio in memoria e imposta l'algoritmo (es. FBP)
    rec_id = astra.data2d.create("-vol", vol_geom)

    cfg = astra.astra_dict(algorithm)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ProjectorId"] = proj_id

    # 5. ESECUZIONE: Lancia il calcolo (1 singolo passaggio per FBP analitica, 50 iterazioni se iterativo)
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1 if algorithm == "FBP" else 50)

    # Estrae l'immagine ricostruita dalla memoria di ASTRA verso Python
    reconstruction = astra.data2d.get(rec_id)

    # 6. PULIZIA RAM: Libera gli oggetti creati nella memoria C++ di ASTRA per evitare memory leak
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return sinogram, reconstruction

import os
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = {
    "cnn_best.pt":    "1soKrd3OY7Geyb_ew723u4Ssn4CFAqG2k",
    "mlp_best.pt":    "1aZGPYSh7qSDhOZ8doTeQf7RIS5TDG4aj",
    "fusion_best.pt": "1mkw2s4OMbgMsHyH5wR25pijGw31olaKF",
}

for filename, file_id in models.items():
    path = os.path.join(BASE_DIR, filename)
    # Always delete and redownload to ensure fresh files
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted old {filename}", flush=True)
    print(f"Downloading {filename}...", flush=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, path, quiet=False)
    print(f"Done: {filename} — {os.path.getsize(path)} bytes", flush=True)
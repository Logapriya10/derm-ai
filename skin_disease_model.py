import os
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORCE = os.environ.get("FORCE_REDOWNLOAD", "0") == "1"

models = {
    "cnn_best.pt":    "1soKrd3OY7Geyb_ew723u4Ssn4CFAqG2k",
    "mlp_best.pt":    "1aZGPYSh7qSDhOZ8doTeQf7RIS5TDG4aj",
    "fusion_best.pt": "1mkw2s4OMbgMsHyH5wR25pijGw31olaKF",
}

for filename, file_id in models.items():
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path) or FORCE:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted old {filename}")
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, path, quiet=False)
        print(f"Done: {filename} — {os.path.getsize(path)} bytes")
    else:
        print(f"Already exists: {filename}")
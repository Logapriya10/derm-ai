import os, io, base64, traceback
import numpy as np
from PIL import Image
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

from flask import Flask, request, jsonify
from flask_cors import CORS

import threading
def load_models():
    global cnn, mlp, fusion, gradcam_engine
    try:
        from skin_disease_model import ImageBranch, SymptomMLP, FusionHead

        cnn_path    = os.path.join(BASE_DIR, "cnn_best.pt")
        mlp_path    = os.path.join(BASE_DIR, "mlp_best.pt")
        fusion_path = os.path.join(BASE_DIR, "fusion_best.pt")

        print(f"[LOAD] Loading CNN from {cnn_path}", flush=True)
        cnn = ImageBranch().to(DEVICE)
        cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE, weights_only=True))
        cnn.eval()
        print("[LOAD] CNN OK", flush=True)

        print(f"[LOAD] Loading MLP from {mlp_path}", flush=True)
        mlp = SymptomMLP().to(DEVICE)
        mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE, weights_only=True))
        mlp.eval()
        print("[LOAD] MLP OK", flush=True)

        print(f"[LOAD] Loading Fusion from {fusion_path}", flush=True)
        fusion = FusionHead().to(DEVICE)
        fusion.load_state_dict(torch.load(fusion_path, map_location=DEVICE, weights_only=True))
        fusion.eval()
        print("[LOAD] Fusion OK", flush=True)

        gradcam_engine = GradCAM(cnn)
        print(f"[OK] All 3 models + GradCAM loaded | device={DEVICE}", flush=True)
        return True
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}", flush=True)
        traceback.print_exc()
        return False

load_dotenv()

app    = Flask(__name__)
CORS(app, origins="*")

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSES     = ["melanoma", "vitiligo", "psoriasis", "ringworm", "acne", "normal"]
NUM_CLASSES = 6
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

SYMPTOM_LABELS = [
    "Itching", "Burning sensation", "Skin redness", "White/light patches",
    "Silvery scales", "Blisters", "Circular ring lesion", "Pus-filled spots",
    "Fever", "Dark irregular mole", "Waxy lesion", "Facial flushing",
    "Dry/cracked skin", "Bleeding/oozing", "Pain/tenderness",
]

DISEASE_INFO = {
    "melanoma": {
        "title": "Melanoma",
        "description": "The most serious type of skin cancer. Early detection is critical.",
        "symptoms": ["Changing mole", "Irregular borders", "Multiple colors", "Larger than 6mm"],
        "causes": ["UV radiation", "Fair skin", "Family history", "Weakened immune system"],
        "treatment": ["Surgical removal", "Immunotherapy", "Targeted therapy", "Radiation"],
        "when_doctor": "URGENT — See a doctor immediately if you suspect melanoma.",
        "disclaimer": "URGENT: Immediate consultation with a dermatologist is strongly recommended.",
        "urgency": "urgent",
    },
    "vitiligo": {
        "title": "Vitiligo",
        "description": "A disease causing loss of skin color in patches.",
        "symptoms": ["White patches on skin", "Hair whitening", "No pain or itch", "Slowly spreading"],
        "causes": ["Autoimmune condition", "Genetics", "Stress", "Skin trauma"],
        "treatment": ["Topical corticosteroids", "Light therapy", "Depigmentation", "Skin grafting"],
        "when_doctor": "If patches are spreading rapidly or affecting visible areas.",
        "disclaimer": "This is an AI-assisted analysis. Please consult a dermatologist.",
        "urgency": "normal",
    },
    "psoriasis": {
        "title": "Psoriasis",
        "description": "A skin disease causing red, itchy, scaly patches.",
        "symptoms": ["Silvery scales", "Dry cracked skin", "Nail pitting", "Joint stiffness"],
        "causes": ["Immune system dysfunction", "Genetics", "Stress triggers", "Infections"],
        "treatment": ["Topical treatments", "Light therapy", "Systemic medications", "Biologics"],
        "when_doctor": "If plaques are spreading rapidly or joints are painful.",
        "disclaimer": "This is an AI-assisted analysis. Please consult a dermatologist.",
        "urgency": "normal",
    },
    "ringworm": {
        "title": "Ringworm (Tinea Corporis)",
        "description": "A fungal infection causing a ring-shaped rash. Highly contagious but treatable.",
        "symptoms": ["Ring-shaped rash", "Itching", "Scaly patches", "Hair loss in affected area"],
        "causes": ["Fungal infection", "Direct contact", "Shared items", "Weakened immunity"],
        "treatment": ["Antifungal creams", "Oral antifungals", "Keep area dry", "Avoid sharing items"],
        "when_doctor": "If rash spreads or does not improve with antifungal treatment.",
        "disclaimer": "This is an AI-assisted analysis. Please consult a dermatologist.",
        "urgency": "normal",
    },
    "acne": {
        "title": "Acne Vulgaris",
        "description": "A skin condition caused by clogged hair follicles.",
        "symptoms": ["Pimples and pustules", "Oily skin", "Redness around spots", "Scarring"],
        "causes": ["Excess oil production", "Bacteria", "Hormonal changes", "Diet"],
        "treatment": ["Topical retinoids", "Benzoyl peroxide", "Antibiotics", "Isotretinoin for severe cases"],
        "when_doctor": "If acne is severe, painful, or causing emotional distress.",
        "disclaimer": "This is an AI-assisted analysis. Please consult a dermatologist.",
        "urgency": "normal",
    },
    "normal": {
        "title": "Normal Skin",
        "description": "No significant skin disease detected in the image.",
        "symptoms": [],
        "causes": [],
        "treatment": ["Maintain regular skincare routine", "Use sunscreen daily"],
        "when_doctor": "If you notice any new or changing skin features.",
        "disclaimer": "This is an AI-assisted analysis. Please consult a dermatologist if concerned.",
        "urgency": "normal",
    },
    "unknown": {
        "title": "Unknown / Inconclusive",
        "description": "The AI could not confidently identify a condition. Confidence was below 40%.",
        "symptoms": [],
        "causes": ["Poor image quality", "Uncommon condition"],
        "treatment": ["Consult a dermatologist directly", "Try uploading a clearer image"],
        "when_doctor": "Please consult a dermatologist for proper diagnosis.",
        "disclaimer": "This result is inconclusive. Always consult a qualified medical professional.",
        "urgency": "normal",
    },
}

# ── Transforms ─────────────────────────────────────────────────
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
S    = 300

TTA_TF = [
    transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(), NORM]),
    transforms.Compose([transforms.Resize((S, S)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), NORM]),
    transforms.Compose([transforms.Resize((S, S)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), NORM]),
    transforms.Compose([transforms.Resize((S, S)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), NORM]),
    transforms.Compose([transforms.Resize((S+40, S+40)), transforms.CenterCrop(S), transforms.ToTensor(), NORM]),
    transforms.Compose([transforms.Resize((S+40, S+40)), transforms.CenterCrop(S), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), NORM]),
    transforms.Compose([transforms.Resize((S, S)), transforms.RandomRotation((15, 15)), transforms.ToTensor(), NORM]),
    transforms.Compose([transforms.Resize((S, S)), transforms.RandomRotation((-15, -15)), transforms.ToTensor(), NORM]),
]

EVAL_TF = transforms.Compose([
    transforms.Resize((S, S)),
    transforms.ToTensor(),
    NORM,
])

# ── GradCAM ────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        # Hook last conv block of EfficientNet B3 = features[8]
        target_layer = model.features[8]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        img_tensor = img_tensor.clone().requires_grad_(True)
        logits     = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = cv2.resize(cam, (S, S))
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def apply_heatmap(pil_img, cam):
    img_np  = np.array(pil_img.resize((S, S)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)
    buf     = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Load models ────────────────────────────────────────────────
cnn = mlp = fusion = None
gradcam_engine = None

def load_models():
    global cnn, mlp, fusion, gradcam_engine
    try:
        from skin_disease_model import ImageBranch, SymptomMLP, FusionHead

        cnn_path    = os.path.join(BASE_DIR, "cnn_best.pt")
        mlp_path    = os.path.join(BASE_DIR, "mlp_best.pt")
        fusion_path = os.path.join(BASE_DIR, "fusion_best.pt")

        for path in [cnn_path, mlp_path, fusion_path]:
            if not os.path.exists(path):
                print(f"[ERROR] Missing: {path}")
                return False

        cnn = ImageBranch().to(DEVICE)
        cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE, weights_only=True))
        cnn.eval()

        mlp = SymptomMLP().to(DEVICE)
        mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE, weights_only=True))
        mlp.eval()

        fusion = FusionHead().to(DEVICE)
        fusion.load_state_dict(torch.load(fusion_path, map_location=DEVICE, weights_only=True))
        fusion.eval()

        gradcam_engine = GradCAM(cnn)

        print(f"[OK] All 3 models + GradCAM loaded | device={DEVICE}")
        return True
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        traceback.print_exc()
        return False

load_models()


# ── Inference ──────────────────────────────────────────────────
def run_predict(pil_img, symptom_values=None):
    probs = []
    with torch.no_grad():
        for tf in TTA_TF:
            t   = tf(pil_img).unsqueeze(0).to(DEVICE)
            out = cnn(t)
            probs.append(torch.softmax(out, dim=1).squeeze(0))
    cnn_prob = torch.stack(probs).mean(0)

    symptoms_used = symptom_values and any(v != 0 for v in symptom_values)

    if symptoms_used:
        symp_tensor = torch.tensor(symptom_values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mlp_prob = torch.softmax(mlp(symp_tensor), dim=1).squeeze(0)
        mode = "fusion"
    else:
        mlp_prob = torch.zeros(NUM_CLASSES, device=DEVICE)
        mode     = "image_only"

    fusion_input = torch.cat([cnn_prob, mlp_prob]).unsqueeze(0)
    with torch.no_grad():
        final_logits = fusion(fusion_input).squeeze(0)
    final_prob = torch.softmax(final_logits, dim=0)

    pred_idx   = final_prob.argmax().item()
    confidence = final_prob[pred_idx].item()

    # ── All probs dict — use final_prob so top predictions match hero ──
    all_probs  = {CLASSES[i]: round(final_prob[i].item() * 100, 1) for i in range(NUM_CLASSES)}

    # ── Unknown threshold ──
    if confidence * 100 < 40:
        pred_class = "unknown"
    else:
        pred_class = CLASSES[pred_idx]

    return pred_class, pred_idx, confidence, all_probs, mode


# ── Health ─────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": cnn is not None,
        "device":       str(DEVICE),
        "classes":      CLASSES,
        "model_type":   "CNN + MLP + Fusion + GradCAM",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if cnn is None:
        return jsonify({"error": "Models are still loading, please wait 1-2 minutes and try again."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400

    img_file  = request.files["image"]
    extension = (img_file.filename or "").rsplit(".", 1)[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '.{extension}'."}), 400

    try:
        pil_img = Image.open(img_file.stream).convert("RGB")

        # Parse symptoms
        symptom_values = None
        raw = request.form.get("symptoms", "")
        if raw:
            try:
                symptom_values = [int(x) for x in raw.split(",")]
            except:
                symptom_values = None

        pred_class, pred_idx, confidence, all_probs, mode = run_predict(pil_img, symptom_values)

        # ── GradCAM ──
        gradcam_b64 = ""
        try:
            img_tensor  = EVAL_TF(pil_img).unsqueeze(0).to(DEVICE)
            cam, _      = gradcam_engine.generate(img_tensor, class_idx=pred_idx)
            gradcam_b64 = apply_heatmap(pil_img, cam)
            print("[OK] GradCAM generated")
        except Exception as e:
            print(f"[WARN] GradCAM failed: {e}")
            traceback.print_exc()

        # Original image → base64
        buf = io.BytesIO()
        pil_img.resize((S, S)).save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # If confidence is too low, return unknown
        if confidence * 100 < 40:
            pred_class = "unknown"
        return jsonify({
            "prediction":    pred_class,
            "confidence":    round(confidence * 100, 1),
            "all_probs":     all_probs,
            "disease_info":  DISEASE_INFO[pred_class],
            "image":         img_b64,
            "gradcam":       gradcam_b64,
            "symptoms_used": bool(symptom_values and any(v != 0 for v in symptom_values)),
            "mode":          mode,
            "symptoms":      SYMPTOM_LABELS,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
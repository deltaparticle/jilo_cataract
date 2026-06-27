import os
import pickle
import torch
# Set PyTorch threads to 1 to prevent severe CPU throttling/contention in containers
torch.set_num_threads(1)
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as T

app = FastAPI(title="Cataract Diagnostics AI")

# Mount static files folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------------------
# 1. Model Structure Re-creation & Quantization Helpers
# -------------------------------------------------------------
def _get_attr(obj, *candidates):
    for name in candidates:
        if hasattr(obj, name):
            return name, getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of the expected attributes: {candidates}")

def _resolve_layer_norms(layer):
    before_name, _ = _get_attr(layer, "layernorm_before", "norm1", "layer_norm1")
    after_name,  _ = _get_attr(layer, "layernorm_after",  "norm2", "layer_norm2")
    return before_name, after_name

def _resolve_mlp_parts(mlp):
    _, fc1 = _get_attr(mlp, "fc1")
    _, act = _get_attr(mlp, "activation", "act_fn", "intermediate_act_fn")
    _, fc2 = _get_attr(mlp, "fc2")
    return fc1, act, fc2

class QuantizableMLP(nn.Module):
    def __init__(self, mlp_module):
        super().__init__()
        self.fc1, self.act, self.fc2 = _resolve_mlp_parts(mlp_module)
        self.quant   = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

def patch_encoder_mlps(encoder):
    for layer in encoder.layer:
        norm_before_name, norm_after_name = _resolve_layer_norms(layer)
        layer.mlp = QuantizableMLP(layer.mlp)
        def make_forward(blk, nb, na):
            def forward(hidden_states):
                norm_before = getattr(blk, nb)
                norm_after  = getattr(blk, na)
                hidden_states_norm = norm_before(hidden_states)
                attn_out = blk.attention(hidden_states_norm)
                attn_out = blk.layer_scale1(attn_out)
                hidden_states = hidden_states + attn_out
                hidden_states_norm = norm_after(hidden_states)
                mlp_out = blk.mlp(hidden_states_norm)
                mlp_out = blk.layer_scale2(mlp_out)
                hidden_states = hidden_states + mlp_out
                return hidden_states
            return forward
        layer.forward = make_forward(layer, norm_before_name, norm_after_name)
    return encoder

class QATDinoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = AutoModel.from_pretrained(
            "facebook/dinov2-small",
            attn_implementation="eager"
        )
        self.embeddings = base_model.embeddings
        self.encoder    = base_model.encoder
        self.encoder.layer = nn.ModuleList(list(self.encoder.layer)[:8])
        patch_encoder_mlps(self.encoder)
        self.layernorm  = base_model.layernorm
        self.classifier = nn.Linear(384, 2)
    def _encode(self, pixel_values):
        x = self.embeddings(pixel_values)
        encoder_out = self.encoder(x)
        x = self.layernorm(encoder_out[0])
        return x[:, 0, :]
    def forward(self, pixel_values):
        return self.classifier(self._encode(pixel_values))
    def extract_features(self, pixel_values):
        return self._encode(pixel_values)

# -------------------------------------------------------------
# 2. Loading Pipeline Globals
# -------------------------------------------------------------
MODEL_PATH = "dinov2_8layer_qat_int8.pth"
PCA_PATH = "pca_transformer.pkl"
SVM_PATH = "svm_qat_pca.pkl"

device = torch.device("cpu") # Run on CPU for deployment

print("Initializing backbone model layout...")
backbone = QATDinoClassifier()
backbone.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
backbone.embeddings.qconfig = None
backbone.layernorm.qconfig   = None
backbone.classifier.qconfig  = None
for layer in backbone.encoder.layer:
    layer.attention.qconfig    = None
    layer.layer_scale1.qconfig = None
    layer.layer_scale2.qconfig = None
    nb, na = _resolve_layer_norms(layer)
    getattr(layer, nb).qconfig = None
    getattr(layer, na).qconfig = None

backbone.train()
torch.ao.quantization.prepare_qat(backbone, inplace=True)
backbone.eval()
backbone.to("cpu")
torch.ao.quantization.convert(backbone, inplace=True)

# Load state dict
if os.path.exists(MODEL_PATH):
    backbone.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Quantized backbone weights loaded successfully.")
else:
    print(f"Warning: {MODEL_PATH} not found.")

# Load preprocessor & scikit-learn models
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)
with open(SVM_PATH, "rb") as f:
    svm = pickle.load(f)
print("Pipeline models loaded.")

val_transform = T.Compose([T.Resize((224, 224))])

# -------------------------------------------------------------
# 3. Routes
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_cataract(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # Open and preprocess the image
        img = Image.open(file.file).convert("RGB")
        img_resized = val_transform(img)
        
        # Prepare inputs for DINOv2
        inputs = processor(images=img_resized, return_tensors="pt")
        
        # Extract features
        with torch.inference_mode():
            embeddings_384 = backbone.extract_features(inputs["pixel_values"]).numpy()
            
        # PCA Dimensionality Reduction (384 -> 64)
        embeddings_64 = pca.transform(embeddings_384)
        
        # SVM Classifier Inference
        prediction = svm.predict(embeddings_64)[0]
        probabilities = svm.predict_proba(embeddings_64)[0]
        
        classes = ["Normal", "Cataract"]
        result_label = classes[prediction]
        confidence = float(probabilities[prediction])
        
        return {
            "prediction": result_label,
            "confidence": f"{confidence * 100:.2f}%",
            "probabilities": {
                "Normal": float(probabilities[0]),
                "Cataract": float(probabilities[1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

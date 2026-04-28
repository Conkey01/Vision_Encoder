# app.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import base64
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, datasets

# --- 1. MODEL DEFINITIONS ---
class PatchEmbed(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=8, stride=8)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, heads=3):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MiniViT(nn.Module):
    def __init__(self, dim=192, depth=6):
        super().__init__()
        self.patch = PatchEmbed(dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, 65, dim))
        self.blocks = nn.Sequential(*[Block(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.patch(x)
        B, N, _ = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :N + 1]
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

class DINOHead(nn.Module):
    def __init__(self, in_dim=192, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, out_dim))
    def forward(self, x):
        return self.mlp(x)

# --- 2. INITIALIZE APP & MODEL ---
app = FastAPI()

# CORS: Allow your frontend website to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(MiniViT(), DINOHead())
model.load_state_dict(torch.load("encoder.pth", map_location=device))
model.to(device)
model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# --- 3. LOAD CIFAR-10 & PRE-COMPUTE EMBEDDINGS ---
# CRITICAL: Hugging Face only allows writes to /tmp
print("Downloading CIFAR-10 to /tmp/data ...")
cifar_raw = datasets.CIFAR10(root="/tmp/data", train=True, download=True)
CIFAR_CLASSES = cifar_raw.classes

cifar_transformed = datasets.CIFAR10(root="/tmp/data", train=True, transform=inference_transform)

GALLERY_SIZE = 10000
subset = torch.utils.data.Subset(cifar_transformed, list(range(GALLERY_SIZE)))
loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)

print(f"Extracting embeddings for {GALLERY_SIZE} images...")
gallery_embs = []
with torch.no_grad():
    for batch_x, _ in loader:
        gallery_embs.append(model(batch_x.to(device)))

gallery_tensor = torch.cat(gallery_embs, dim=0)
print("API ready!")

# --- Helper: Convert PIL to base64 ---
def pil_to_base64(img):
    buf = io.BytesIO()
    img = img.resize((128, 128), Image.NEAREST)
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# --- 4. API ENDPOINTS ---
@app.get("/")
def health():
    return {"status": "running", "gallery_size": GALLERY_SIZE}

@app.post("/search")
async def search(file: UploadFile = File(...)):
    image_bytes = await file.read()
    user_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = inference_transform(user_image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model(tensor)

    sims = F.cosine_similarity(query_emb, gallery_tensor)
    top_scores, top_indices = torch.topk(sims, k=5)

    results = []
    for i in range(5):
        idx = top_indices[i].item()
        score = top_scores[i].item()
        match_img, label_idx = cifar_raw[idx]
        results.append({
            "class": CIFAR_CLASSES[label_idx].capitalize(),
            "score": round(score, 4),
            "image": pil_to_base64(match_img)
        })

    return {
        "query_image": pil_to_base64(user_image),
        "results": results
    }

import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

# Best general purpose anime/illustration tagger
REPO_ID = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
MODEL_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"

class WD14Tagger:
    def __init__(self):
        self.model = None
        self.tags = None
        self.model_path = None
        self.tags_path = None
        # FIX: ConvNextV2 requires 448x448 input (SwinV2 used 446)
        self.target_size = 448 

    def load_model(self):
        """Downloads and loads the ONNX model."""
        try:
            # Download/Cache Model
            self.model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
            self.tags_path = hf_hub_download(repo_id=REPO_ID, filename=TAGS_FILE)

            # Load Tags
            df = pd.read_csv(self.tags_path)
            self.tags = df["name"].tolist()

            # Load ONNX Session (CPU is fast enough for tagging)
            self.model = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            return True
        except Exception as e:
            print(f"Tagger Load Error: {e}")
            return False

    def tag_image(self, image_path, threshold=0.35, max_tags=50, blacklist=None):
        if not self.model:
            if not self.load_model():
                return []

        try:
            # Preprocess
            img = Image.open(image_path).convert("RGB")
            # Resize to expected dim (squash is standard for WD14)
            img = img.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)
            
            # Convert to numpy array (NHWC)
            # The model expects BGR float32
            img_np = np.array(img).astype(np.float32)
            # Flip RGB to BGR
            img_np = img_np[:, :, ::-1] 
            # Add batch dimension: (1, 448, 448, 3)
            input_tensor = np.expand_dims(img_np, 0)

            # Inference
            input_name = self.model.get_inputs()[0].name
            probs = self.model.run(None, {input_name: input_tensor})[0][0]

            # Pair tags with probabilities
            tag_probs = list(zip(self.tags, probs))
            
            # 1. Filter by Threshold & System Tags
            filtered = []
            sys_tags = ["general", "sensitive", "questionable", "explicit"]
            blacklist = blacklist or []
            
            for tag, prob in tag_probs:
                if prob > threshold:
                    if tag not in sys_tags and tag not in blacklist:
                        filtered.append((tag, prob))
            
            # 2. Sort by Confidence
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            # 3. Cutoff Max Tags
            final_tags = [t[0] for t in filtered[:max_tags]]

            return final_tags

        except Exception as e:
            print(f"Inference Error {image_path}: {e}")
            return []
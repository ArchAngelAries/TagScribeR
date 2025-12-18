import os
import torch
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download
from PySide6.QtCore import QObject, Signal

class QwenWorker(QObject):
    finished = Signal(str, str) # file_path, caption
    error = Signal(str)
    progress = Signal(str)      # Log messages
    
    def __init__(self, model_path, file_paths, prompt, params=None):
        super().__init__()
        self.model_path = model_path
        self.file_paths = file_paths
        self.prompt = prompt
        self.params = params or {} # {max_tokens, temperature, top_p}
        self.model = None
        self.processor = None
        self.running = True
        self.device = "cpu"

    def load_model(self):
        try:
            # --- HARDWARE DETECTION ---
            if torch.cuda.is_available():
                # ComfyUI Fix for AMD/Windows
                self.device_map = {"": 0}
                self.dtype = torch.float16
                self.device = "cuda"
                self.progress.emit(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device_map = "cpu"
                self.dtype = torch.float32
                self.device = "cpu"
                self.progress.emit("⚠️ GPU NOT DETECTED! Falling back to CPU.")

            self.progress.emit(f"📂 Loading model from: {self.model_path}")

            if os.path.isfile(self.model_path):
                 self.model_path = os.path.dirname(self.model_path)

            # --- LOAD PROCESSOR ---
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # --- LOAD MODEL ---
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            
            self.model.eval()
            self.progress.emit("✅ Model loaded successfully!")
            return True
        except Exception as e:
            self.error.emit(f"Failed to load model: {str(e)}")
            return False

    def run(self):
        if not self.model:
            if not self.load_model():
                return

        total = len(self.file_paths)
        
        # Extract params
        max_tokens = self.params.get('max_tokens', 256)
        temp = self.params.get('temperature', 0.7)
        top_p = self.params.get('top_p', 0.9)
        
        # Logic: If temp is 0, we can't 'sample', we must be deterministic
        do_sample = True if temp > 0 else False

        for i, fpath in enumerate(self.file_paths):
            if not self.running: break
            try:
                # Prepare Inputs
                messages = [{
                    "role": "user",
                    "content": [{"type": "image", "image": fpath}, {"type": "text", "text": self.prompt}]
                }]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text], 
                    images=image_inputs, 
                    videos=video_inputs, 
                    padding=True, 
                    return_tensors="pt"
                )
                
                inputs = inputs.to(self.model.device)

                # Generate with User Params
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=max_tokens,
                        do_sample=do_sample,
                        temperature=temp,
                        top_p=top_p
                    )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                self.finished.emit(fpath, output_text)
                self.progress.emit(f"({i+1}/{total}) Processed")
                
            except Exception as e:
                self.error.emit(f"Error on {os.path.basename(fpath)}: {str(e)}")
    
    def stop(self):
        self.running = False
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- SEPARATE WORKER FOR DOWNLOADS ---
class DownloadWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    
    def __init__(self, repo_id, target_dir):
        super().__init__()
        self.repo_id = repo_id
        self.target_dir = target_dir
        
    def run(self):
        try:
            self.progress.emit(f"📥 Starting download for {self.repo_id}...")
            self.progress.emit("This may take a while (several GBs)...")
            
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.target_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            self.progress.emit("✅ Download Complete!")
            self.finished.emit()
        except Exception as e:
            self.progress.emit(f"❌ Download Failed: {str(e)}")
            self.finished.emit()
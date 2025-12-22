import os
import platform
import torch
import gc
from PySide6.QtCore import QObject, Signal
from core.image_utils import image_to_base64

class QwenWorker(QObject):
    finished = Signal(str, str) # file_path, caption
    error = Signal(str)
    progress = Signal(str)      # Log messages
    
    def __init__(self, model_path, file_paths, prompt, params=None, api_config=None):
        super().__init__()
        self.file_paths = file_paths
        self.prompt = prompt
        self.params = params or {} 
        self.api_config = api_config 
        
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.running = True
        self.device = "cpu"

    def load_local_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        try:
            current_os = platform.system()
            if torch.cuda.is_available():
                self.progress.emit(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
                if current_os == "Linux":
                    self.device_map = "auto"
                    self.device = "cuda" 
                    self.dtype = torch.float16
                    self.progress.emit("✅ Mode: Linux Optimized (Accelerate Auto-Map)")
                else:
                    self.device_map = None 
                    self.device = "cuda"
                    self.dtype = torch.float16
                    self.progress.emit("✅ Mode: Windows Forced GPU")
            else:
                self.device_map = "cpu"
                self.dtype = torch.float32
                self.device = "cpu"
                self.progress.emit("⚠️ GPU NOT DETECTED! Falling back to CPU.")

            self.progress.emit(f"📂 Loading model: {self.model_path}")

            if os.path.isfile(self.model_path):
                 self.model_path = os.path.dirname(self.model_path)

            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            
            if current_os == "Windows" and self.device == "cuda":
                self.model.to(self.device)
            
            self.model.eval()
            self.progress.emit("✅ Model loaded!")
            return True
        except Exception as e:
            self.error.emit(f"Failed to load local model: {str(e)}")
            return False

    def run_api_inference(self, client, fpath):
        b64_img = image_to_base64(fpath)
        if not b64_img: raise Exception("Failed to encode image to Base64")

        try:
            # Added timeout=60 to prevent infinite hanging
            response = client.chat.completions.create(
                model=self.api_config['model_name'],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        ],
                    }
                ],
                max_tokens=self.params.get('max_tokens', 512),
                temperature=self.params.get('temperature', 0.7),
                top_p=self.params.get('top_p', 0.9),
                timeout=60 # <--- PREVENTS HANGS
            )
            
            if not response or not response.choices:
                raise Exception("API returned empty response.")
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"API Call Failed: {e}")

    def run_local_inference(self, fpath):
        import torch
        from qwen_vl_utils import process_vision_info
        
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": fpath}, {"type": "text", "text": self.prompt}]
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.params.get('max_tokens', 256),
                do_sample=True if self.params.get('temperature', 0.7) > 0 else False,
                temperature=self.params.get('temperature', 0.7),
                top_p=self.params.get('top_p', 0.9)
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def run(self):
        is_api = self.api_config is not None
        
        if is_api:
            from openai import OpenAI
            url = self.api_config['base_url'].strip()
            self.progress.emit(f"🌐 Connecting to API: {url}")
            try:
                client = OpenAI(base_url=url, api_key=self.api_config['api_key'])
            except Exception as e:
                self.error.emit(f"API Init Error: {e}")
                return
        else:
            if not self.model:
                if not self.load_local_model(): return

        total = len(self.file_paths)
        for i, fpath in enumerate(self.file_paths):
            if not self.running: break
            try:
                if is_api:
                    output_text = self.run_api_inference(client, fpath)
                else:
                    output_text = self.run_local_inference(fpath)
                
                # Check running again after potentially long blocking call
                if self.running: 
                    self.finished.emit(fpath, output_text)
                    self.progress.emit(f"({i+1}/{total}) Processed")
                
            except Exception as e:
                self.error.emit(f"Error on {os.path.basename(fpath)}: {str(e)}")
    
    def stop(self):
        self.running = False
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except: pass
        self.model = None
        gc.collect()

# --- DOWNLOAD WORKER (Unchanged) ---
class DownloadWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    
    def __init__(self, repo_id, target_dir):
        super().__init__()
        self.repo_id = repo_id
        self.target_dir = target_dir
        
    def run(self):
        from huggingface_hub import snapshot_download
        try:
            self.progress.emit(f"📥 Starting download for {self.repo_id}...")
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
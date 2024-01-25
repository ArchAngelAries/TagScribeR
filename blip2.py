import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

class BLIP2:
    def __init__(self, model_type):
        # setup device to use
        self.device = torch.device("directml") if torch.directml.is_available() else torch.device("cpu")
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2", model_type=model_type, is_eval=True, device=self.device)

    def generate_caption(
            self, 
            image: Image, 
            num_beams: int = 3,  # number of beams for beam search. 1 means no beam search.
            use_nucleus_sampling: bool = False,  # if False, use top-k sampling
            max_length: int = 30,  # maximum length of the generated caption
            min_length: int = 10,  # minimum length of the generated caption
            top_p: float = 0.9,  # cumulative probability for nucleus sampling
            repetition_penalty: float = 1.0,  # parameter for repetition penalty. 1.0 means no penalty.
        ):
        # Prepare the image
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        # Generate captions
        captions = self.model.generate(
            {"image": image},
            num_beams=num_beams,
            use_nucleus_sampling=use_nucleus_sampling,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        # Decode captions
        caption = self.processor.decode(captions[0], skip_special_tokens=True)
        return caption

    def unload(self):
        del self.model
        del self.vis_processors
        torch.cuda.empty_cache()  # Change to torch.directml.empty_cache() if using DirectML

import bentoml
from PIL.Image import Image
from bentoml.models import HuggingFaceModel


sample_prompt = (
    "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
)


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    workers=1,
    resources={
        "gpu": 1,
    },
)
class Flux:
    model_path = HuggingFaceModel("black-forest-labs/FLUX.1-schnell")

    def __init__(self) -> None:
        from diffusers import FluxPipeline, FluxImg2ImgPipeline
        import torch
        import os
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        torch.set_float32_matmul_precision("high")

        self.pipe = FluxPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).to(device="cuda")
        # self.pipe.transformer.to(memory_format=torch.channels_last)
        # self.pipe.transformer = torch.compile(
        #     self.pipe.transformer, mode="max-autotune"
        # )
        
        self.img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).to(device="cuda")
        self.img2img_pipe.transformer.to(memory_format=torch.channels_last)
        self.img2img_pipe.transformer = torch.compile(
            self.img2img_pipe.transformer, mode="max-autotune"
        )

    @bentoml.api
    def txt2img(
        self,
        prompt: str = sample_prompt,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
    ) -> Image:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image
    
    @bentoml.api
    def img2img(self, image: Image, prompt: str) -> Image:
        image = self.img2img_pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=4,
            guidance_scale=0.,
        ).images[0]
        return image

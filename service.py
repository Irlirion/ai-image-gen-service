import bentoml
from bentoml.models import HuggingFaceModel
from PIL.Image import Image as PILImage

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
class FluxText2Image:
    model_path = HuggingFaceModel("black-forest-labs/FLUX.1-dev")

    def __init__(self) -> None:
        import torch
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

        torch.set_float32_matmul_precision("high")

        self.pipe = FluxPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).to(device="cuda")

        self.pipe = optimize(self.pipe)

    @bentoml.api
    async def predict(
        self,
        prompt: str = sample_prompt,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
    ) -> PILImage:
        images: list[PILImage] = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=512,
            num_images_per_prompt=1,
        ).images
        return images[0]


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    workers=1,
    resources={
        "gpu": 1,
    },
)
class Text2Text:
    model_path = HuggingFaceModel("google/gemma-2-9b-it")

    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        torch.set_float32_matmul_precision("high")

        self.pipe = pipeline(
            "text-generation",
            model=self.model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )

    @bentoml.api
    async def predict(
        self,
        prompt: str = sample_prompt,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        assistant_response = self.pipe(messages, max_new_tokens=512)[0][
            "generated_text"
        ][-1]["content"].strip()  # type: ignore
        return assistant_response


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    workers=1,
)
class Flux:
    text2img_service = bentoml.depends(FluxText2Image)
    text2text_service = bentoml.depends(Text2Text)

    @bentoml.api
    async def txt2img(
        self,
        prompt: str = sample_prompt,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
    ) -> PILImage:
        return await self.text2img_service.predict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )

    @bentoml.api
    async def txt2txt(
        self,
        prompt: str = sample_prompt,
    ) -> str:
        return await self.text2text_service.predict(prompt=prompt)


def optimize(pipe, compile=True):
    import torch

    # fuse QKV projections in Transformer and VAE
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # trigger torch compilation
    print("🔦 running torch compiliation (may take up to 20 minutes)...")
    pipe(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=50,  # use ~50 for [dev], smaller for [schnell]
        height=1024,
        width=1024,
        num_images_per_prompt=1,
        max_sequence_length=512,
    ).images[0]

    print("🔦 finished torch compilation")

    return pipe

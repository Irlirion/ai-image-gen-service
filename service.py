import bentoml
import PIL
import PIL.Image
from bentoml.models import HuggingFaceModel
from PIL.Image import Image

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
class FluxImage2Image:
    model_path = HuggingFaceModel("black-forest-labs/FLUX.1-dev")

    def __init__(self) -> None:
        import torch
        from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline

        torch.set_float32_matmul_precision("high")

        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).to(device="cuda")

        self.pipe = optimize(self.pipe)

    @bentoml.api
    async def predict(
        self,
        image: Image,
        prompt: str,
        num_inference_steps: int = 50,
        strength: float = 0.6,
        guidance_scale: float = 3.5,
    ) -> Image:
        image = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            height=image.height,
            width=image.width,
        ).images[0]
        return image


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
    ) -> Image:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]
        return image


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    workers=1,
)
class Flux:
    text2img_service = bentoml.depends(FluxText2Image)
    img2img_service = bentoml.depends(FluxImage2Image)

    @bentoml.api
    async def txt2img(
        self,
        prompt: str = sample_prompt,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
    ) -> Image:
        image = await self.text2img_service.predict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )
        return image

    @bentoml.api
    async def img2img(
        self,
        prompt: str,
        image: Image,
        num_inference_steps: int = 50,
        strength: float = 0.6,
        guidance_scale: float = 3.5,
    ) -> Image:
        image = await self.img2img_service.predict(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
        )
        return image


def optimize(pipe, compile=True):
    import torch
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
    from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline

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
    if isinstance(pipe, FluxImg2ImgPipeline):
        pipe(
            "dummy prompt to trigger torch compilation",
            image=PIL.Image.new("RGB", (1024, 1024)),
            output_type="pil",
            num_inference_steps=50,  # use ~50 for [dev], smaller for [schnell]
        ).images[0]
    elif isinstance(pipe, FluxPipeline):
        pipe(
            "dummy prompt to trigger torch compilation",
            output_type="pil",
            num_inference_steps=50,  # use ~50 for [dev], smaller for [schnell]
        ).images[0]
    else:
        raise ValueError("Unsupported pipeline type, received %s" % type(pipe))

    print("🔦 finished torch compilation")

    return pipe

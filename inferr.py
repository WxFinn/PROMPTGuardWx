from diffusers import StableDiffusionPipeline
import torch
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# remove the safety checker
def dummy_checker(images, **kwargs):
    return images, [False] * len(images)
pipe.safety_checker = dummy_checker

safety_embedding_list = ["./safe_embeddings/YOUR_SAFE_EMBEDDINGS_FOLDER/learned_embeds.safetensors"] # the save paths of your embeddings
token1 = "agressive"


token_list = [token1] # the corresponding tokens of your embeddings

pipe.load_textual_inversion(pretrained_model_name_or_path=safety_embedding_list, token=token_list)

origin_prompt = "a photo of a dog"
prompt_with_system = origin_prompt + " " + token1
image = pipe(prompt_with_system).images[0]
image.save("example01.png")
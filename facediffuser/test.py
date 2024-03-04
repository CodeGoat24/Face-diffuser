from transforms import (
    get_test_transforms_with_segmap,
    get_test_object_transforms,
    get_object_processor,
    get_object_transforms
)
from data import DemoDataset
from model import FaceDiffuserModel
from model_text_only import FaceDiffuserModel as FaceDiffuserModel_text_only
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from accelerate.utils import set_seed
from util import parse_args
from accelerate import Accelerator
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import os
from tqdm.auto import tqdm
from pipeline import (
    stable_diffusion_call_with_references_delayed_conditioning,
)
import types
import itertools
import os
from data import get_data_loader, FaceDiffuserDataset
from torchvision.transforms import ToPILImage

@torch.no_grad()
def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision='fp16',
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

    model = FaceDiffuserModel.from_pretrained(args)

    ckpt_name = "pytorch_model.bin"

    model.load_state_dict(
        torch.load(Path(args.finetuned_model_path) / ckpt_name, map_location="cpu")
    )

    model = model.to(device=accelerator.device, dtype=weight_dtype)

    pipe.unet = model.unet

    if args.enable_xformers_memory_efficient_attention:
        pipe.unet.enable_xformers_memory_efficient_attention()

    pipe.text_encoder = model.text_encoder
    pipe.image_encoder = model.image_encoder

    pipe.postfuse_module = model.postfuse_module

    pipe.inference = types.MethodType(
        stable_diffusion_call_with_references_delayed_conditioning, pipe
    )

    model = FaceDiffuserModel_text_only.from_pretrained(args)

    model.load_state_dict(
        torch.load(Path(args.finetuned_model_text_only_path) / ckpt_name, map_location="cpu")
    )
    model = model.to(device=accelerator.device, dtype=weight_dtype)
    pipe.unet_text_only = model.unet
    del model

    pipe = pipe.to(accelerator.device)

    # Set up the dataset
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    test_transforms = get_test_transforms_with_segmap(args)
    object_transforms = get_test_object_transforms(args)
    object_processor = get_object_processor(args)

    test_dataset = FaceDiffuserDataset(
        args.dataset_name,
        tokenizer,
        train_transforms = test_transforms,
        object_transforms = object_transforms,
        object_processor = object_processor,
        device=accelerator.device,
        max_num_objects=args.max_num_objects,
        num_image_tokens=args.num_image_tokens,
        object_appear_prob=args.object_appear_prob,
        uncondition_prob=args.uncondition_prob,
        text_only_prob=args.text_only_prob,
        object_types= None,
        split="all",
        min_num_objects=args.min_num_objects,
        balance_num_objects=args.balance_num_objects,
        mode='train'
    )


    test_dataloader = get_data_loader(test_dataset, args.test_batch_size)


    os.makedirs(args.output_dir, exist_ok=True)

    for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(accelerator.device)
            text = tokenizer.batch_decode(input_ids)[0]
            # print(input_ids)
            image_token_mask = batch["image_token_mask"].to(accelerator.device)

            # print(image_token_mask)
            all_object_pixel_values = (
                batch["object_pixel_values"].to(accelerator.device)
            )
            num_objects = batch["num_objects"].to(accelerator.device)

            all_object_pixel_values = all_object_pixel_values.to(
                dtype=weight_dtype, device=accelerator.device
            )

            object_pixel_values = all_object_pixel_values  # [:, 0, :, :, :]
            if pipe.image_encoder is not None:
                object_embeds = pipe.image_encoder(object_pixel_values)
            else:
                object_embeds = None

            encoder_hidden_states = pipe.text_encoder(
                input_ids, image_token_mask, object_embeds, num_objects
            )[0]

            encoder_hidden_states_text_only = pipe._encode_prompt(
                batch["caption"],
                accelerator.device,
                args.num_images_per_prompt,
                do_classifier_free_guidance=False,
            )

            encoder_hidden_states = pipe.postfuse_module(
                encoder_hidden_states,
                object_embeds,
                image_token_mask,
                num_objects,
            )

            cross_attention_kwargs = {}

            images = pipe.inference(
                prompt_embeds=encoder_hidden_states,
                num_inference_steps=args.inference_steps,
                height=args.generate_height,
                width=args.generate_width,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                cross_attention_kwargs=cross_attention_kwargs,
                prompt_embeds_text_only=encoder_hidden_states_text_only,
                start_merge_step=args.start_merge_step,
                final_step=args.final_step,
            ).images


            for index, image_id in enumerate(batch['image_ids']):
                folder_path = Path(args.output_dir+f'/{image_id}')
                
                folder_path.mkdir(parents=True, exist_ok=True)
                for i, object in enumerate(batch["object_pixel_values"][index]):
                    if batch['image_token_idx_mask'][index][i] > 0:
                        to_pil = ToPILImage()
                        # 使用 to_pil() 方法将张量转换为 PIL 图片
                        pil_image = to_pil(object)
                        pil_image.save(
                            os.path.join(
                                folder_path,
                                f"object_{image_id}_{i}.png",
                            )
                        )
                    else:
                        break
                for instance_id in range(args.num_images_per_prompt):
                    if len(batch['caption']) > 70:
                        batch['caption'] = batch['caption'][:70]
                    images[index].save(
                        os.path.join(
                            folder_path,
                            f"output_{image_id}_{batch['caption'][index]}_{instance_id}.png",
                        )
                    )
                    


if __name__ == "__main__":
    main()

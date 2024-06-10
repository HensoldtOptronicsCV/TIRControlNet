"""
Script to perform IR image synthesis using the method presented in: 
        
    Diffusion based Thermal Semantic Segmentation - CVPR PBVS Workshop 2024

Author: Christian Mayr
Date: 15.05.2024
"""
# pylint: disable=E0401
import argparse
import random
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image


def parse_arguments():
    """Define argument parser"""
    parser = argparse.ArgumentParser(description="Inference script")
    # Adding positional argument
    parser.add_argument(
        "-b",
        "--base_model_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Stable Diffusion backbone",
    )
    parser.add_argument(
        "-c",
        "--contronlet_path",
        type=str,
        default="/path/to/controlnet",
        help="Path to ControlNet checkpoint",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Urban automotive scene containing vegetation, vehicles, roads, buildings, and persons",
        help="Prompt for Stable Diffusion",
    )
    parser.add_argument(
        "-i",
        "--inference_steps",
        type=int,
        default=90,
        help="Amount of steps inference steps",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=Path,
        default=(Path(__file__).resolve().parent / "results"),
        help="Default output folder - if set to None images won\'t be saved",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        type=bool,
        default=True,
        help="If true a random seed will be used, if False the seed is set to 1337",
    )
    # Parse the command-line arguments
    arguments = parser.parse_args()

    return arguments


def pipeline_setup(base_model_path, contronlet_path):
    """Controlnet pipeline setup
    
        Input:
            base_model_path - Path or link to the StableDiffusion base model.
            controlnet_path - Path to the pretrained ControlNet for IR Image Synthesis
        Output:
            pipe - ControlNet pipeline object for image synthesis
    """
    controlnet = ControlNetModel.from_pretrained(
        contronlet_path,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    return pipe


def synthesize_ir_images(semseg_paths, args):
    """Function for the IR image synthesis using semantic maps as input

        Input:
            semseg_paths - List of paths to the semantic maps used as synthesis input
            args - Argument parser arguments
        Output:
            synthesized_ir_imgs - List containing all synthesized images as objects
    
    """
    if args.random_seed:
        seed_lst = [random.randint(0, 1000000) for _ in range(len(semseg_paths))]
    else:
        seed_lst = [1337] * len(semseg_paths)

    synthesized_ir_imgs = [None] * len(semseg_paths)

    # Setup pipeline for inference
    pipeline = pipeline_setup(args.base_model_path, args.contronlet_path)

    # Synthesize an IR image for each provided semantic map
    for idx, semantic_map_path in enumerate(semseg_paths):
        if not Path(semantic_map_path).is_absolute():
            semantic_map_path = (Path(__file__).resolve().parent / semantic_map_path).resolve()
        control_image = load_image(str(semantic_map_path))
        seed = seed_lst[idx]
        generator = torch.manual_seed(seed)
        synthesized_ir_img = pipeline(
            args.prompt,
            num_inference_steps=args.inference_steps,
            generator=generator,
            image=control_image,
        ).images[0]

        if args.output_folder:
            synthesized_ir_img.save(
                Path(args.output_folder)
                / (Path(semantic_map_path).stem + "__" + str(seed) + ".png")
            )

        synthesized_ir_imgs[idx] = synthesized_ir_img

    return synthesized_ir_imgs


if __name__ == "__main__":
    input_args = parse_arguments()

    if input_args.output_folder:
        input_args.output_folder.mkdir(parents=True, exist_ok=True)

    # Provided semantic maps HAVE to be grayscale
    semantic_map_paths = [
        "demo/input/00001_4.png",
        "demo/input/00011_4.png",
        "demo/input/00015_4.png",
        "demo/input/00018_4.png",
        "demo/input/00023_4.png",
    ]

    synthesized_images = synthesize_ir_images(
        semantic_map_paths,
        input_args
    )

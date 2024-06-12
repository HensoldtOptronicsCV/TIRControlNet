# MIT License
#
# Copyright (c) 2024 HENSOLDT Optronics Computer Vision
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Inspired by https://github.com/avaapm/marveldataset2016/blob/master/MARVEL_Download.py
"""
Script to perform TIR image synthesis using the method presented in: 
        
    Narrowing the Synthetic-to-Real Gap for Thermal Infrared Semantic Image Segmentation Using Diffusion-based Conditional Image Synthesis
    CVPR PBVS Workshop 2024

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
    # HuggingFace link to StableDiffusion 2.1, or path to downloaded copy
    parser.add_argument(
        "-b",
        "--base_model_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Stable Diffusion backbone",
    )

    # HuggingFace link to the provided pretrained ControlNet, or path to downloaded copy
    parser.add_argument(
        "-c",
        "--controlnet_path",
        type=str,
        default="0x434D/TIR_ControlNet",
        help="Path to ControlNet checkpoint",
    )

    # Prompt to be used by ControlNet's synthesis backbone
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Urban automotive scene containing vegetation, vehicles, roads, buildings, and persons",
        help="Prompt for Stable Diffusion",
    )

    # The amount of inference steps ControlNet's synthesis backbone performs to synthesize an image
    parser.add_argument(
        "-i",
        "--inference_steps",
        type=int,
        default=90,
        help="Amount of inference steps performed by ControlNet",
    )

    # Folder in which the synthesized images will be saved to
    parser.add_argument(
        "-o",
        "--output_folder",
        type=Path,
        default=(Path(__file__).resolve().parent / "results"),
        help="Default output folder - if set to None images won\'t be saved",
    )

    # Flag wether to use a random or a set seed for image synthesis
    parser.add_argument(
        "-r",
        "--random_seed",
        type=bool,
        default=True,
        help="If true a random seed will be used, if False the seed is set to 1337",
    )
    arguments = parser.parse_args()

    return arguments


def pipeline_setup(base_model_path, controlnet_path):
    """ControlNet pipeline setup
    
        Input:
            base_model_path - Path or link to the StableDiffusion base model.
            controlnet_path - Path to the pretrained ControlNet for TIR Image Synthesis
        Output:
            pipe - ControlNet pipeline object for image synthesis
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    return pipe


def synthesize_tir_images(semseg_paths, args):
    """Function for the TIR image synthesis using semantic maps as input

        Input:
            semseg_paths - List of paths to the semantic maps used as synthesis input
            args - Argument parser arguments
        Output:
            synthesized_tir_imgs - List containing all synthesized TIR images as objects
    
    """
    if args.random_seed:
        seed_lst = [random.randint(0, 1000000) for _ in range(len(semseg_paths))]
    else:
        seed_lst = [1337] * len(semseg_paths)

    synthesized_tir_imgs = [None] * len(semseg_paths)

    # Setup pipeline for inference
    pipeline = pipeline_setup(args.base_model_path, args.controlnet_path)

    # Synthesize a TIR image for each provided semantic map
    for idx, semantic_map_path in enumerate(semseg_paths):
        if not Path(semantic_map_path).is_absolute():
            semantic_map_path = (Path(__file__).resolve().parent / semantic_map_path).resolve()
        control_image = load_image(str(semantic_map_path))
        seed = seed_lst[idx]
        generator = torch.manual_seed(seed)
        synthesized_tir_img = pipeline(
            args.prompt,
            num_inference_steps=args.inference_steps,
            generator=generator,
            image=control_image,
        ).images[0]

        if args.output_folder:
            synthesized_tir_img.save(
                Path(args.output_folder)
                / (Path(semantic_map_path).stem + "__" + str(seed) + ".png")
            )

        synthesized_tir_imgs[idx] = synthesized_tir_img

    return synthesized_tir_imgs


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

    synthesized_images = synthesize_tir_images(
        semantic_map_paths,
        input_args
    )

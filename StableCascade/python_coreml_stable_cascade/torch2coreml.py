#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# Default size of the image is 512 and Stable Cascade achieves a compression factor of 42,
# meaning that it is possible to encode a 512x512 image to 12x12.
# https://github.com/huggingface/diffusers/blob/1c60e094de0b4f86e7bf13009f4d49a27073b9f5/src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade_combined.py#L157
# https://huggingface.co/stabilityai/stable-cascade

COMPRESSION_FACTOR = 42
SAMPLE_SIZE = 512 / COMPRESSION_FACTOR

from python_coreml_stable_cascade import (
    unet, controlnet, chunk_mlprogram
)

import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import coremltools as ct
from diffusers import (
    StableCascadeCombinedPipeline
)
from diffusers.models import StableCascadeUNet
import gc

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os
import requests
import shutil
import time
import re
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

from types import MethodType


def _get_coreml_inputs(sample_inputs, args):
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        ) for k, v in sample_inputs.items()
    ]


def compute_psnr(a, b):
    """ Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects
    """
    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    eps = 1e-5
    eps2 = 1e-10
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr


ABSOLUTE_MIN_PSNR = 35


def report_correctness(original_outputs, final_outputs, log_prefix):
    """ Report PSNR values across two compatible tensors
    """
    original_psnr = compute_psnr(original_outputs, original_outputs)
    final_psnr = compute_psnr(original_outputs, final_outputs)

    dB_change = final_psnr - original_psnr
    logger.info(
        f"{log_prefix}: PSNR changed by {dB_change:.1f} dB ({original_psnr:.1f} -> {final_psnr:.1f})"
    )

    if final_psnr < ABSOLUTE_MIN_PSNR:
        raise ValueError(f"{final_psnr:.1f} dB is too low!")
    else:
        logger.info(
            f"{final_psnr:.1f} dB > {ABSOLUTE_MIN_PSNR} dB (minimum allowed) parity check passed"
        )
    return final_psnr

def _get_out_path(args, submodule_name):
    fname = f"Stable_Cascade_version_{args.model_version}_{submodule_name}.mlpackage"
    fname = fname.replace("/", "_")
    return os.path.join(args.o, fname)


def _convert_to_coreml(submodule_name, torchscript_module, sample_inputs,
                       output_names, args, out_path=None, precision=None, compute_unit=None):

    if out_path is None:
        out_path = _get_out_path(args, submodule_name)

    compute_unit = compute_unit or ct.ComputeUnit[args.compute_unit]

    if os.path.exists(out_path):
        logger.info(f"Skipping export because {out_path} already exists")
        logger.info(f"Loading model from {out_path}")

        start = time.time()
        # Note: Note that each model load will trigger a model compilation which takes up to a few minutes.
        # The Swifty CLI we provide uses precompiled Core ML models (.mlmodelc) which incurs compilation only
        # upon first load and mitigates the load time in subsequent runs.
        coreml_model = ct.models.MLModel(
            out_path, compute_units=compute_unit)
        logger.info(
            f"Loading {out_path} took {time.time() - start:.1f} seconds")

        coreml_model.compute_unit = compute_unit
    else:
        logger.info(f"Converting {submodule_name} to CoreML..")
        coreml_model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
            inputs=_get_coreml_inputs(sample_inputs, args),
            outputs=[ct.TensorType(name=name, dtype=np.float32) for name in output_names],
            compute_units=compute_unit,
            compute_precision=precision,
            skip_model_load=not args.check_output_correctness,
        )

        del torchscript_module
        gc.collect()

    return coreml_model, out_path


def quantize_weights(args):
    """ Quantize weights to args.quantize_nbits using a palette (look-up table)
    """
    for model_name in ["text_encoder", "text_encoder_2", "unet", "refiner", "control-unet"]:
        logger.info(f"Quantizing {model_name} to {args.quantize_nbits}-bit precision")
        out_path = _get_out_path(args, model_name)
        _quantize_weights(
            out_path,
            model_name,
            args.quantize_nbits
        )

    if args.convert_controlnet:
        for controlnet_model_version in args.convert_controlnet:
            controlnet_model_name = controlnet_model_version.replace("/", "_")
            logger.info(f"Quantizing {controlnet_model_name} to {args.quantize_nbits}-bit precision")
            fname = f"ControlNet_{controlnet_model_name}.mlpackage"
            out_path = os.path.join(args.o, fname)
            _quantize_weights(
                out_path,
                controlnet_model_name,
                args.quantize_nbits
            )

def _quantize_weights(out_path, model_name, nbits):
    if os.path.exists(out_path):
        logger.info(f"Quantizing {model_name}")
        mlmodel = ct.models.MLModel(out_path,
                                    compute_units=ct.ComputeUnit.CPU_ONLY)

        op_config = ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
        )

        config = ct.optimize.coreml.OptimizationConfig(
            global_config=op_config,
            op_type_configs={
                "gather": None # avoid quantizing the embedding table
            }
        )

        model = ct.optimize.coreml.palettize_weights(mlmodel, config=config).save(out_path)
        logger.info("Done")
    else:
        logger.info(
            f"Skipped quantizing {model_name} (Not found at {out_path})")


def _compile_coreml_model(source_model_path, output_dir, final_name):
    """ Compiles Core ML models using the coremlcompiler utility from Xcode toolchain
    """
    target_path = os.path.join(output_dir, f"{final_name}.mlmodelc")
    if os.path.exists(target_path):
        logger.warning(
            f"Found existing compiled model at {target_path}! Skipping..")
        return target_path

    logger.info(f"Compiling {source_model_path}")
    source_model_name = os.path.basename(
        os.path.splitext(source_model_path)[0])

    os.system(f"xcrun coremlcompiler compile {source_model_path} {output_dir}")
    compiled_output = os.path.join(output_dir, f"{source_model_name}.mlmodelc")
    shutil.move(compiled_output, target_path)

    return target_path


def bundle_resources_for_swift_cli(args):
    """
    - Compiles Core ML models from mlpackage into mlmodelc format
    - Download tokenizer resources for the text encoder
    """
    resources_dir = os.path.join(args.o, "Resources")
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir, exist_ok=True)
        logger.info(f"Created {resources_dir} for Swift CLI assets")

    # Compile model using coremlcompiler (Significantly reduces the load time for unet)
    for source_name, target_name in [("text_encoder", "TextEncoder"),
                                     ("text_encoder_2", "TextEncoder2"),
                                     ("vae_decoder", "VAEDecoder"),
                                     ("vae_encoder", "VAEEncoder"),
                                     ("unet", "Unet"),
                                     ("unet_chunk1", "UnetChunk1"),
                                     ("unet_chunk2", "UnetChunk2"),
                                     ("refiner", "UnetRefiner"),
                                     ("refiner_chunk1", "UnetRefinerChunk1"),
                                     ("refiner_chunk2", "UnetRefinerChunk2"),
                                     ("control-unet", "ControlledUnet"),
                                     ("control-unet_chunk1", "ControlledUnetChunk1"),
                                     ("control-unet_chunk2", "ControlledUnetChunk2"),
                                     ("safety_checker", "SafetyChecker")]:
        source_path = _get_out_path(args, source_name)
        if os.path.exists(source_path):
            target_path = _compile_coreml_model(source_path, resources_dir,
                                                target_name)
            logger.info(f"Compiled {source_path} to {target_path}")
        else:
            logger.warning(
                f"{source_path} not found, skipping compilation to {target_name}.mlmodelc"
            )
            
    if args.convert_controlnet:
        for controlnet_model_version in args.convert_controlnet:
            controlnet_model_name = controlnet_model_version.replace("/", "_")
            fname = f"ControlNet_{controlnet_model_name}.mlpackage"
            source_path = os.path.join(args.o, fname)
            controlnet_dir = os.path.join(resources_dir, "controlnet")
            target_name = "".join([word.title() for word in re.split('_|-', controlnet_model_name)])

            if os.path.exists(source_path):
                target_path = _compile_coreml_model(source_path, controlnet_dir,
                                                    target_name)
                logger.info(f"Compiled {source_path} to {target_path}")
            else:
                logger.warning(
                    f"{source_path} not found, skipping compilation to {target_name}.mlmodelc"
                )

    # Fetch and save vocabulary JSON file for text tokenizer
    logger.info("Downloading and saving tokenizer vocab.json")
    with open(os.path.join(resources_dir, "vocab.json"), "wb") as f:
        f.write(requests.get(args.text_encoder_vocabulary_url).content)
    logger.info("Done")

    # Fetch and save merged pairs JSON file for text tokenizer
    logger.info("Downloading and saving tokenizer merges.txt")
    with open(os.path.join(resources_dir, "merges.txt"), "wb") as f:
        f.write(requests.get(args.text_encoder_merges_url).content)
    logger.info("Done")

    return resources_dir


from transformers.models.clip import modeling_clip

# Copied from https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/clip/modeling_clip.py#L677C1-L692C1
def patched_make_causal_mask(input_ids_shape, dtype, device, past_key_values_length: int = 0):
    """ Patch to replace torch.finfo(dtype).min with -1e4
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(-1e4, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

modeling_clip._make_causal_mask = patched_make_causal_mask

def convert_text_encoder(text_encoder, tokenizer, submodule_name, args):
    """ Converts the text encoder component of Stable Cascade
    """
    text_encoder = text_encoder.to(dtype=torch.float32)
    out_path = _get_out_path(args, submodule_name)
    if os.path.exists(out_path):
        logger.info(
            f"`{submodule_name}` already exists at {out_path}, skipping conversion."
        )
        return

    # Create sample inputs for tracing, conversion and correctness verification
    text_encoder_sequence_length = tokenizer.model_max_length

    sample_text_encoder_inputs = {
        "input_ids":
        torch.randint(
            text_encoder.config.vocab_size,
            (1, text_encoder_sequence_length),
            # https://github.com/apple/coremltools/issues/1423
            dtype=torch.float32,
        )
    }
    sample_text_encoder_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_text_encoder_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_text_encoder_inputs_spec}")

    class TextEncoder(nn.Module):

        def __init__(self, with_hidden_states_for_layer=None):
            super().__init__()
            self.text_encoder = text_encoder
            self.with_hidden_states_for_layer = with_hidden_states_for_layer

        def forward(self, input_ids):
            if self.with_hidden_states_for_layer is not None:
                output = self.text_encoder(input_ids, output_hidden_states=True)
                hidden_embeds = output.hidden_states[self.with_hidden_states_for_layer]
                if "text_embeds" in output:
                    return (hidden_embeds, output.text_embeds)
                else:
                    return (hidden_embeds, output.pooler_output)
            else:
                return self.text_encoder(input_ids, return_dict=False)

    reference_text_encoder = TextEncoder(with_hidden_states_for_layer=None).eval()

    logger.info(f"JIT tracing {submodule_name}..")
    reference_text_encoder = torch.jit.trace(
        reference_text_encoder,
        (sample_text_encoder_inputs["input_ids"].to(torch.int32), ),
    )
    logger.info("Done.")

    # if args.xl_version:
    #     output_names = ["hidden_embeds", "pooled_outputs"]
    output_names = ["last_hidden_state", "pooled_outputs"]
    coreml_text_encoder, out_path = _convert_to_coreml(
        submodule_name, reference_text_encoder, sample_text_encoder_inputs,
        output_names, args)

    # Set model metadata
    coreml_text_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    # if args.xl_version:
    #     coreml_text_encoder.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
    coreml_text_encoder.license = "OpenRAIL (https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE)"
    coreml_text_encoder.version = args.model_version
    coreml_text_encoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://openreview.net/forum?id=gU58d5QeGv for details."

    # Set the input descriptions
    coreml_text_encoder.input_description[
        "input_ids"] = "The token ids that represent the input text"

    # Set the output descriptions
    # if args.xl_version:
    #     coreml_text_encoder.output_description[
    #         "hidden_embeds"] = "Hidden states after the encoder layers"
    coreml_text_encoder.output_description[
            "last_hidden_state"] = "The token embeddings as encoded by the Transformer model"
    coreml_text_encoder.output_description[
        "pooled_outputs"] = "The version of the `last_hidden_state` output after pooling"

    coreml_text_encoder.save(out_path)

    logger.info(f"Saved text_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = text_encoder(
            sample_text_encoder_inputs["input_ids"].to(torch.int32),
            # output_hidden_states=args.xl_version,
            return_dict=True,
        )
        # if args.xl_version:
        #     # TODO: maybe check pooled_outputs too
        #     baseline_out = baseline_out.hidden_states[hidden_layer].numpy()
        baseline_out = baseline_out.last_hidden_state.numpy()

        coreml_out = coreml_text_encoder.predict(
            {k: v.numpy() for k, v in sample_text_encoder_inputs.items()}
        )
        coreml_out = coreml_out["hidden_embeds" if args.xl_version else "last_hidden_state"]
        report_correctness(
            baseline_out, coreml_out,
            "text_encoder baseline PyTorch to reference CoreML")

    del reference_text_encoder, coreml_text_encoder
    gc.collect()


def modify_coremltools_torch_frontend_badbmm():
    """
    Modifies coremltools torch frontend for baddbmm to be robust to the `beta` argument being of non-float dtype:
    e.g. https://github.com/huggingface/diffusers/blob/v0.8.1/src/diffusers/models/attention.py#L315
    """
    from coremltools.converters.mil import register_torch_op
    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
    if "baddbmm" in _TORCH_OPS_REGISTRY:
        del _TORCH_OPS_REGISTRY["baddbmm"]

    @register_torch_op
    def baddbmm(context, node):
        """
        baddbmm(Tensor input, Tensor batch1, Tensor batch2, Scalar beta=1, Scalar alpha=1)
        output = beta * input + alpha * batch1 * batch2
        Notice that batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, then input must be broadcastable with a (b×n×p) tensor
        and out will be a (b×n×p) tensor.
        """
        assert len(node.outputs) == 1
        inputs = _get_inputs(context, node, expected=5)
        bias, batch1, batch2, beta, alpha = inputs

        if beta.val != 1.0:
            # Apply scaling factor beta to the bias.
            if beta.val.dtype == np.int32:
                beta = mb.cast(x=beta, dtype="fp32")
                logger.warning(
                    f"Casted the `beta`(value={beta.val}) argument of `baddbmm` op "
                    "from int32 to float32 dtype for conversion!")
            bias = mb.mul(x=beta, y=bias, name=bias.name + "_scaled")

            context.add(bias)

        if alpha.val != 1.0:
            # Apply scaling factor alpha to the input.
            batch1 = mb.mul(x=alpha, y=batch1, name=batch1.name + "_scaled")
            context.add(batch1)

        bmm_node = mb.matmul(x=batch1, y=batch2, name=node.name + "_bmm")
        context.add(bmm_node)

        baddbmm_node = mb.add(x=bias, y=bmm_node, name=node.name)
        context.add(baddbmm_node)


def convert_vqgan_decoder(pipe, args):
    """ Converts the vqgan Decoder component of Stable Cascade
    """
    out_path = _get_out_path(args, "vae_decoder")
    if os.path.exists(out_path):
        logger.info(
            f"`vqgan_decoder` already exists at {out_path}, skipping conversion."
        )
        return

    if not hasattr(pipe, "decoder"):
        raise RuntimeError(
            "convert_unet() deletes pipe.unet to save RAM. "
            "Please use convert_vae_decoder() before convert_unet()")

    z_shape = (
        1,  # B
        pipe.vqgan.config.latent_channels,  # C
        args.latent_h or round(SAMPLE_SIZE),  # H
        args.latent_w or round(SAMPLE_SIZE),  # W
    )

    # if args.custom_vae_version is None and args.xl_version:
    #     inputs_dtype = torch.float32
    #     compute_precision = ct.precision.FLOAT32
    #     # FIXME: Hardcoding to CPU_AND_GPU since ANE doesn't support FLOAT32
    #     compute_unit = ct.ComputeUnit.CPU_AND_GPU
    # else:
    inputs_dtype = torch.float16
    compute_precision = None
    compute_unit = None

    sample_vqgan_decoder_inputs = {
        "z": torch.rand(*z_shape, dtype=inputs_dtype)
    }

    class VQGANDecoder(nn.Module):
        """ Wrapper nn.Module wrapper for pipe.decode() method
        """

        def __init__(self):
            super().__init__()
            self.up_blocks = pipe.vqgan.up_blocks.to(dtype=torch.float32)
            self.out_block = pipe.vqgan.out_block.to(dtype=torch.float32)

        def forward(self, x):
            return self.out_block(self.up_blocks(x))

    baseline_decoder = VQGANDecoder().eval()

    # No optimization needed for the VQGAN Decoder as it is a pure ConvNet
    traced_vqgan_decoder = torch.jit.trace(
        baseline_decoder, (sample_vqgan_decoder_inputs["z"].to(torch.float32), ))

    modify_coremltools_torch_frontend_badbmm()
    coreml_vqgan_decoder, out_path = _convert_to_coreml(
        "vqgan_decoder", traced_vqgan_decoder, sample_vqgan_decoder_inputs,
        ["image"], args, precision=compute_precision, compute_unit=compute_unit)

    # Set model metadata
    coreml_vqgan_decoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    # if args.xl_version:
    #     coreml_vqgan_decoder.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
    coreml_vqgan_decoder.license = "OpenRAIL (https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE)"
    coreml_vqgan_decoder.version = args.model_version
    coreml_vqgan_decoder.short_description = \
        "Stable Cascade generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://openreview.net/forum?id=gU58d5QeGv for details."

    # Set the input descriptions
    coreml_vqgan_decoder.input_description["z"] = \
        "The denoised latent embeddings from the unet model after the last step of reverse diffusion"

    # Set the output descriptions
    coreml_vqgan_decoder.output_description["image"] = "Generated image normalized to range [-1, 1]"

    coreml_vqgan_decoder.save(out_path)

    logger.info(f"Saved vqgan_decoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_decoder(
            z=sample_vqgan_decoder_inputs["z"].to(torch.float32)).numpy()
        coreml_out = list(
            coreml_vqgan_decoder.predict(
                {k: v.numpy()
                 for k, v in sample_vqgan_decoder_inputs.items()}).values())[0]
        report_correctness(baseline_out, coreml_out,
                           "vqgan_decoder baseline PyTorch to baseline CoreML")

    del traced_vqgan_decoder, pipe.vqgan.out_block, coreml_vqgan_decoder
    gc.collect()

# def convert_decoder(pipe, args, model_name = None):
#     """ Converts the UNet component of Stable Diffusion
#     """
#     if args.unet_support_controlnet:
#         decoder_name = "control-unet"
#     else:
#         decoder_name = model_name or "decoder"

#     out_path = _get_out_path(args, decoder_name)

#     # Check if Unet was previously exported and then chunked
#     decoder_chunks_exist = all(
#         os.path.exists(out_path.replace(".mlpackage", f"_chunk{idx+1}.mlpackage"))
#         for idx in range(2))

#     if args.chunk_decoder and decoder_chunks_exist:
#         logger.info("`decoder` chunks already exist, skipping conversion.")
#         del pipe.decoder
#         gc.collect()
#         return

#     # If original Unet does not exist, export it from PyTorch+diffusers
#     if not os.path.exists(out_path):
#         # Prepare sample input shapes and values
#         batch_size = 2  # for classifier-free guidance
#         sample_shape = (
#             batch_size,                       # B
#             pipe.decoder.config.in_channels,  # C
#             args.latent_h or round(SAMPLE_SIZE),     # H
#             args.latent_w or round(SAMPLE_SIZE),     # W
#         )

#         if not hasattr(pipe, "text_encoder"):
#             raise RuntimeError(
#                 "convert_text_encoder() deletes pipe.text_encoder to save RAM. "
#                 "Please use convert_unet() before convert_text_encoder()")

#         if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
#             text_token_sequence_length = pipe.text_encoder.config.max_position_embeddings
#             hidden_size = pipe.text_encoder.config.hidden_size

#         encoder_hidden_states_shape = (
#             batch_size,
#             args.text_encoder_hidden_size or round(SAMPLE_SIZE * COMPRESSION_FACTOR) or hidden_size,
#             1,
#             args.text_token_sequence_length or text_token_sequence_length,
#         )

#         # Create the scheduled timesteps for downstream use
#         DEFAULT_NUM_INFERENCE_STEPS = 50
#         pipe.scheduler.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS)

#         sample_decoder_inputs = OrderedDict([
#             ("sample", torch.rand(*sample_shape)),
#             ("timestep",
#              torch.tensor([pipe.scheduler.timesteps[0].item()] *
#                           (batch_size)).to(torch.float32)),
#             ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape))
#         ])

#         # Prepare inputs
#         baseline_sample_decoder_inputs = deepcopy(sample_decoder_inputs)
#         baseline_sample_decoder_inputs[
#             "encoder_hidden_states"] = baseline_sample_decoder_inputs[
#                 "encoder_hidden_states"].squeeze(2).transpose(1, 2)

#         # Initialize reference unet
#         # unet_cls = unet.UNet2DConditionModel
#         decoder_cls = StableCascadeUNet

#         reference_decoder = decoder_cls(**pipe.decoder.config).eval()

#         # load_state_dict_summary = reference_unet.load_state_dict(
#         #     pipe.unet.state_dict())

#         # if args.unet_support_controlnet:
#         #     from .unet import calculate_conv2d_output_shape
#         #     additional_residuals_shapes = []

#         #     # conv_in
#         #     out_h, out_w = calculate_conv2d_output_shape(
#         #         (args.latent_h or pipe.unet.config.sample_size),
#         #         (args.latent_w or pipe.unet.config.sample_size),
#         #         reference_unet.conv_in,
#         #     )
#         #     additional_residuals_shapes.append(
#         #         (batch_size, reference_unet.conv_in.out_channels, out_h, out_w))
            
#         #     # down_blocks
#         #     for down_block in reference_unet.down_blocks:
#         #         additional_residuals_shapes += [
#         #             (batch_size, resnet.out_channels, out_h, out_w) for resnet in down_block.resnets
#         #         ]
#         #         if hasattr(down_block, "downsamplers") and down_block.downsamplers is not None:
#         #             for downsampler in down_block.downsamplers:
#         #                 out_h, out_w = calculate_conv2d_output_shape(out_h, out_w, downsampler.conv)
#         #             additional_residuals_shapes.append(
#         #                 (batch_size, down_block.downsamplers[-1].conv.out_channels, out_h, out_w))
            
#         #     # mid_block
#         #     additional_residuals_shapes.append(
#         #         (batch_size, reference_unet.mid_block.resnets[-1].out_channels, out_h, out_w)
#         #     )

#         #     baseline_sample_unet_inputs["down_block_additional_residuals"] = ()
#         #     for i, shape in enumerate(additional_residuals_shapes):
#         #         sample_residual_input = torch.rand(*shape)
#         #         sample_decoder_inputs[f"additional_residual_{i}"] = sample_residual_input
#         #         if i == len(additional_residuals_shapes) - 1:
#         #             baseline_sample_unet_inputs["mid_block_additional_residual"] = sample_residual_input
#         #         else:
#         #             baseline_sample_unet_inputs["down_block_additional_residuals"] += (sample_residual_input, )

#         sample_decoder_inputs_spec = {
#             k: (v.shape, v.dtype)
#             for k, v in sample_decoder_inputs.items()
#         }
#         logger.info(f"Sample decoder inputs spec: {sample_decoder_inputs_spec}")

#         # JIT trace
#         logger.info("JIT tracing..")
#         reference_decoder = torch.jit.trace(reference_decoder,
#                                          list(sample_decoder_inputs.values()))
#         logger.info("Done.")

#         if args.check_output_correctness:
#             baseline_out = pipe.decoder.to(torch.float32)(**baseline_sample_decoder_inputs,
#                                      return_dict=False)[0].numpy()
#             reference_out = reference_decoder(*sample_decoder_inputs.values())[0].numpy()
#             report_correctness(baseline_out, reference_out,
#                                "decoder baseline to reference PyTorch")

#         del pipe.decoder
#         gc.collect()

#         coreml_sample_decoder_inputs = {
#             k: v.numpy().astype(np.float16)
#             for k, v in sample_decoder_inputs.items()
#         }


#         coreml_decoder, out_path = _convert_to_coreml(decoder_name, reference_decoder,
#                                                    coreml_sample_decoder_inputs,
#                                                    ["noise_pred"], args)
#         del reference_decoder
#         gc.collect()

#         # Set model metadata
#         coreml_decoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
#         # if args.xl_version:
#         #     coreml_decoder.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
#         coreml_decoder.license = "OpenRAIL (https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE)"
#         coreml_decoder.version = args.model_version
#         coreml_decoder.short_description = \
#             "Stable Diffusion generates images conditioned on text or other images as input through the diffusion process. " \
#             "Please refer to https://openreview.net/forum?id=gU58d5QeGv for details."

#         # Set the input descriptions
#         coreml_decoder.input_description["sample"] = \
#             "The low resolution latent feature maps being denoised through reverse diffusion"
#         coreml_decoder.input_description["timestep"] = \
#             "A value emitted by the associated scheduler object to condition the model on a given noise schedule"
#         coreml_decoder.input_description["encoder_hidden_states"] = \
#             "Output embeddings from the associated text_encoder model to condition to generated image on text. " \
#             "A maximum of 77 tokens (~40 words) are allowed. Longer text is truncated. " \
#             "Shorter text does not reduce computation."
#         if args.xl_version:
#             coreml_decoder.input_description["time_ids"] = \
#                 "Additional embeddings that if specified are added to the embeddings that are passed along to the UNet blocks."
#             coreml_decoder.input_description["text_embeds"] = \
#                 "Additional embeddings from text_encoder_2 that if specified are added to the embeddings that are passed along to the UNet blocks."

#         # Set the output descriptions
#         coreml_decoder.output_description["noise_pred"] = \
#             "Same shape and dtype as the `sample` input. " \
#             "The predicted noise to facilitate the reverse diffusion (denoising) process"

#         # Set package version metadata
#         from python_coreml_stable_cascade._version import __version__
#         coreml_decoder.user_defined_metadata["com.github.apple.ml-stable-diffusion.version"] = __version__

#         coreml_decoder.save(out_path)
#         logger.info(f"Saved decoder into {out_path}")

#         # Parity check PyTorch vs CoreML
#         if args.check_output_correctness:
#             coreml_out = list(
#                 coreml_decoder.predict(coreml_sample_decoder_inputs).values())[0]
#             report_correctness(baseline_out, coreml_out,
#                                "unet baseline PyTorch to reference CoreML")

#         del coreml_decoder
#         gc.collect()
#     else:
#         del pipe.decoder
#         gc.collect()
#         logger.info(
#             f"`decoder` already exists at {out_path}, skipping conversion.")

#     if args.chunk_decoder and not decoder_chunks_exist:
#         logger.info(f"Chunking {model_name} in two approximately equal MLModels")
#         args.mlpackage_path = out_path
#         args.remove_original = False
#         args.merge_chunks_in_pipeline_model = False
#         chunk_mlprogram.main(args)

def _get_controlnet_base_model(controlnet_model_version):
    from huggingface_hub import model_info
    info = model_info(controlnet_model_version)
    return info.cardData.get("base_model", None)

def get_pipeline(args):
    model_version = args.model_version

    logger.info(f"Initializing StableCascadeCombinedPipeline with {model_version}..")
    # model_version is "stabilityai/stable-cascade" (for CLI)
    pipe = StableCascadeCombinedPipeline.from_pretrained(model_version,
                                        torch_dtype=torch.float16,
                                        variant="bf16",
                                        use_safetensors=True,
                                        )

    logger.info(f"Done. Pipeline in effect: {pipe.__class__.__name__}")

    return pipe


def main(args):
    os.makedirs(args.o, exist_ok=True)

    # Instantiate diffusers pipe as reference
    pipe = get_pipeline(args)

    # Register the selected attention implementation globally
    unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet.AttentionImplementations[
        args.attention_implementation]
    logger.info(
        f"Attention implementation in effect: {unet.ATTENTION_IMPLEMENTATION_IN_EFFECT}"
    )

    # Convert models
    if args.convert_vqgan_decoder:
        logger.info("Converting vqgan_decoder")
        convert_vqgan_decoder(pipe, args)
        logger.info("Converted vqgan_decoder")

    if args.convert_controlnet:
        logger.info("Converting controlnet")
        convert_controlnet(pipe, args)
        logger.info("Converted controlnet")

    # if args.convert_decoder:
    #     logger.info("Converting decoder")
    #     convert_decoder(pipe, args)
    #     logger.info("Converted decoder")
    
    # if args.convert_prior:
    #     logger.info("Converting prior")
    #     convert_decoder(pipe, args)
    #     logger.info("Converted prior")

    if args.convert_text_encoder and hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        logger.info("Converting text_encoder")
        convert_text_encoder(pipe.text_encoder, pipe.tokenizer, "text_encoder", args)
        del pipe.text_encoder
        logger.info("Converted text_encoder")

    if args.quantize_nbits is not None:
        logger.info(f"Quantizing weights to {args.quantize_nbits}-bit precision")
        quantize_weights(args)
        logger.info(f"Quantized weights to {args.quantize_nbits}-bit precision")

    if args.bundle_resources_for_swift_cli:
        logger.info("Bundling resources for the Swift CLI")
        bundle_resources_for_swift_cli(args)
        logger.info("Bundled resources for the Swift CLI")


def parser_spec():
    parser = argparse.ArgumentParser()

    # Select which models to export (All are needed for text-to-image pipeline to function)
    parser.add_argument("--convert-text-encoder", action="store_true")
    parser.add_argument("--convert-vqgan-decoder", action="store_true")
    # parser.add_argument("--convert-decoder", action="store_true")
    # parser.add_argument("--convert-prior", action="store_true")
    parser.add_argument(
        "--convert-controlnet", 
        nargs="*",
        type=str,
        help=
        "Converts a ControlNet model hosted on HuggingFace to coreML format. " \
        "To convert multiple models, provide their names separated by spaces.",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="The pre-trained model checkpoint and configuration to restore. "
    )
    parser.add_argument("--compute-unit",
                        choices=tuple(cu for cu in ct.ComputeUnit._member_names_),
                        default="ALL")
    parser.add_argument(
        "--latent-h",
        type=int,
        default=None,
        help=
        "The spatial resolution (number of rows) of the latent space. `Defaults to pipe.unet.config.sample_size`",
    )
    parser.add_argument(
        "--latent-w",
        type=int,
        default=None,
        help=
        "The spatial resolution (number of cols) of the latent space. `Defaults to pipe.unet.config.sample_size`",
    )
    parser.add_argument(
        "--text-token-sequence-length",
        type=int,
        default=None,
        help=
        "The token sequence length for the text encoder. `Defaults to pipe.text_encoder.config.max_position_embeddings`",
    )
    parser.add_argument(
        "--text-encoder-hidden-size",
        type=int,
        default=None,
        help=
        "The hidden size for the text encoder. `Defaults to pipe.text_encoder.config.hidden_size`",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=tuple(ai for ai in unet.AttentionImplementations._member_names_),
        default=unet.ATTENTION_IMPLEMENTATION_IN_EFFECT.name,
        help="The enumerated implementations trade off between ANE and GPU performance",
    )
    parser.add_argument(
        "-o",
        default=os.getcwd(),
        help="The resulting mlpackages will be saved into this directory")
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
        help=
        "If specified, compares the outputs of original PyTorch and final CoreML models and reports PSNR in dB. "
        "Enabling this feature uses more memory. Disable it if your machine runs out of memory."
        )
    parser.add_argument(
        "--chunk-decoder",
        action="store_true",
        help=
        "If specified, generates two mlpackages out of the unet model which approximately equal weights sizes. "
        "This is required for ANE deployment on iOS and iPadOS. Not required for macOS."
        )
    parser.add_argument(
        "--quantize-nbits",
        default=None,
        choices=(1, 2, 4, 6, 8),
        type=int,
        help="If specified, quantized each model to nbits precision"
    )
    return parser


if __name__ == "__main__":
    parser = parser_spec()
    args = parser.parse_args()

    main(args)

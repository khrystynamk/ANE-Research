import torch
import coremltools as ct
import numpy as np
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

def convert_stablecascade():
    """
    Convert the PyTorch ML model to CoreML
    """
    # Define example inputs for tracing

    example_prompt = "A rainy day in New York city"
    negative_prompt = ""
    num_images_per_prompt = 1

    # Load pretrained models
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior",
                                                      torch_dtype=torch.bfloat16)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",
                                                          torch_dtype=torch.float16)

    # Trace the models
    prior_traced_model = torch.jit.trace(prior, example_prompt)
    decoder_traced_model = torch.jit.trace(decoder, example_prompt)

    # Convert the traced models to CoreML format
    prior_model = ct.convert(
        prior_traced_model,
        inputs=[example_prompt],  # Adjust input shape as needed
    )

    decoder_model = ct.convert(
        decoder_traced_model,
        inputs=[example_prompt],  # Adjust input shape as needed
    )

    # Save the converted models
    prior_model.save("stablecascade_prior.mlmodel")
    decoder_model.save("stablecascade_decoder.mlmodel")
    print("Models converted and saved as stablecascade_prior.mlmodel and stablecascade_decoder.mlmodel")

if __name__ == "__main__":
    convert_stablecascade()

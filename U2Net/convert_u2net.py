from model.u2net_refactor import U2NET
import torch
import coremltools as ct
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def convert_u2net():
    net_model = U2NET(3,1)
    net_model.eval()
    example_input = torch.rand(1, 3, 320, 320)
    traced_model = torch.jit.trace(net_model, example_input)

    scale = 1 / (0.226*255.0)
    bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

    image_input = ct.ImageType(name="input_1",
                            shape=example_input.shape,
                            scale=scale, bias=bias)

    model = ct.convert(
        traced_model,
        inputs=[image_input],
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    model.short_description = "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"
    model.license = "Apache 2.0"
    model.author = "Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin"

    model.save("u2net.mlpackage")
    print("Model converted and saved as u2net.mlpackage")
    return model

def example_run(model: ct.models.MLModel, img_path: str):
    """
    Run the CoreML model with some provided example.
    """

if __name__ == "__main__":
    mlmodel = convert_u2net()
    example_run(mlmodel, "daisy.jpg")

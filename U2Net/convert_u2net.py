from model.u2net import U2NET
import torch
import coremltools as ct
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# def add_activation_layer(model: ct.models.MLModel):
#     spec = model.get_spec()
#     nn = getattr(spec, spec.WhichOneof('Type'))
#     spec_layers = nn.layers
#     output_layers = []
#     for layer in spec_layers:
#         if layer.name[:2] == "25":
#             print("name: %s input: %s output: %s" % (layer.name, layer.input, layer.output))
#             output_layers.append(layer)
#     new_layers = []
#     layernum = 0
#     for layer in output_layers:
#         new_layer = spec_layers.add()
#         new_layer.name = 'out_p'+str(layernum)
#         new_layers.append('out_p'+str(layernum))
#         new_layer.activation.linear.alpha=255
#         new_layer.activation.linear.beta=0
#         new_layer.input.append('var_'+layer.name)
#         new_layer.output.append('out_p'+str(layernum))
#         output_description = next(x for x in spec.description.output if x.name==output_layers[layernum].output[0])
#         output_description.name = new_layer.name
#         layernum = layernum + 1
#     # Make output GrayScale image.
    
#     for output in spec.description.output:
#         if output.name not in new_layers:
#             continue
#         if output.type.WhichOneof('Type') != 'multiArrayType':
#             raise ValueError("%s is not a multiarray type" % output.name)
#         # output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
#         output.type.imageType.width = 320
#         output.type.imageType.height = 320
#     # save updated model.
#     updated_model = ct.models.MLModel(spec)
#     updated_model.save("u2net.mlmodel")

def convert_u2net():
    net_model = U2NET(3, 1)
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

    img = Image.open(img_path)
    img = img.resize([320, 320])
    coreml_out_dict = model.predict({"input_1" : img})
    print(coreml_out_dict) # There is an array of pixels, need to convert to an image and show it

if __name__ == "__main__":
    mlmodel = convert_u2net()
    example_run(mlmodel, "/Users/khrystynamysak/Documents/UCU APPS/APPS_ACS/ANE-Research/daisy.jpg")

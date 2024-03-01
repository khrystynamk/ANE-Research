import torch
import urllib
import coremltools as ct
from PIL import Image
import numpy as np
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def convert_mobilenetV2():
    """
    Convert the PyTorch ML model to CoreML
    """
    torch_model = torchvision.models.mobilenet_v2(pretrained=True) # Load a pre-trained version of MobileNetV2 model.

    # To ensure that operations such as dropout are disabled, it’s 
    # important to set the model to evaluation mode (not training mode)
    # before tracing. This setting also results in a more optimized
    # version of the model for conversion.
    torch_model.eval()

    # The rank and shape of the tensor depends on your model’s use case.
    # If your model expects a fixed-size input, use that size for the example image.
    # In all cases, the rank of the tensor must be fixed.
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(torch_model, example_input)

    # Download class labels in ImageNetLabel.txt.
    label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()
    class_labels = class_labels[1:] # remove the first class, which is background
    assert len(class_labels) == 1000

    # Set the image scale and bias for input image preprocessing.
    scale = 1 / (0.226*255.0)
    bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

    # By default, the Core ML Tools converter generates a Core ML model
    # with inputs of type MLMultiArray. By providing an additional inputs argument,
    # as shown in the next section, you can use either TensorType or ImageType.
    image_input = ct.ImageType(name="input_1",
                            shape=example_input.shape,
                            scale=scale, bias=bias)

    # Convert to Core ML using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        inputs=[image_input],
        classifier_config = ct.ClassifierConfig(class_labels),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    # Save the converted model.
    model.save("mobilenet.mlpackage")
    print("Model converted and saved as mobilenet.mlpackage")
    return model

def example_run(model: ct.models.MLModel, img_path: str):
    """
    Run the CoreML model with some provided example.
    """
    # Load the test image and resize to 224, 224.
    img = Image.open(img_path)
    img = img.resize([224, 224])

    # Get the protobuf spec of the model.
    spec = model.get_spec()
    for out in spec.description.output:
        if out.type.WhichOneof('Type') == "dictionaryType":
            coreml_dict_name = out.name
            break

    # Make a prediction with the Core ML version of the model.
    coreml_out_dict = model.predict({"input_1" : img})
    print("coreml predictions: ")
    print("top class label: ", coreml_out_dict["classLabel"])

    coreml_prob_dict = coreml_out_dict[coreml_dict_name]

    values_vector = np.array(list(coreml_prob_dict.values()))
    keys_vector = list(coreml_prob_dict.keys())
    top_3_indices_coreml = np.argsort(-values_vector)[:3]
    for i in range(3):
        idx = top_3_indices_coreml[i]
        score_value = values_vector[idx]
        class_id = keys_vector[idx]
        print("class name: {}, raw score value: {}".format(class_id, score_value))

if __name__ == "__main__":
    mlmodel = convert_mobilenetV2()
    example_run(mlmodel, "daisy.jpg")

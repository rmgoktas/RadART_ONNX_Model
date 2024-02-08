from flask import Flask, request, jsonify
from PIL import Image
import base64
import os
import io
import json
import onnxruntime as rt
import numpy as np
import re

app = Flask(__name__)

class ONNXModel:
    def __init__(self, dir_path) -> None:
        
        model_dir = os.path.dirname(dir_path)
        signature_dir = "RadART_ONNX_Model-main FOLDER_PATH" 
        #example: /Users/rmgoktas/Desktop/RadART_ONNX_Model-main
        

        with open(os.path.join(signature_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)

        
        input_shape = self.signature.get("inputs").get("Image").get("shape")
        if len(input_shape) != 4:
            raise ValueError(f"Invalid input shape. Expected 4 dimensions, got {len(input_shape)} dimensions.")

        self.model_file = os.path.join(signature_dir, "model.onnx")
        self.labels_file = os.path.join(signature_dir, "labels.txt")

        if not os.path.isfile(self.model_file) or not os.path.isfile(self.labels_file):
            raise FileNotFoundError("Model file or labels file does not exist.")

        self.session = None

        if "Image" not in self.signature.get("inputs"):
            raise ValueError("ONNX model doesn't have 'Image' input! Check signature.json.")

        version = self.signature.get("export_model_version")
        if version is None or version != 1:
            print(
                f"There has been a change to the model format. Please use a model with a signature 'export_model_version' that matches {1}."
            )

        with open(self.labels_file, "r") as labels_file:
            self.labels = [label.strip() for label in labels_file.readlines()]

        
        self.signature_inputs = self.signature.get("inputs")
        self.signature_outputs = self.signature.get("outputs")


    def load(self) -> None:
        """Load the model from path to model file"""
        
        self.session = rt.InferenceSession(path_or_bytes=self.model_file)

    def predict(self, image: Image.Image) -> dict:
        """
        Predict
        """
        # process image to be compatible with the model
        img = self.process_image(image, self.signature_inputs.get("Image").get("shape"))
        # run the model
        fetches = [(key, value.get("name")) for key, value in self.signature_outputs.items()]
        # make the image a batch of 1
        feed = {self.signature_inputs.get("Image").get("name"): [img]}
        outputs = self.session.run(output_names=[name for (_, name) in fetches], input_feed=feed)
        return self.process_output(fetches, outputs)

    def process_image(self, image: Image.Image, input_shape: list) -> np.ndarray:
        """
        Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
        """
        width, height = image.size
        # ensure image type is compatible with model and convert if not
        if image.mode != "RGB":
            image = image.convert("RGB")
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # format input as model expects
        return image.astype(np.float32)


    def process_output(self, fetches: dict, outputs: dict) -> dict:
        # un-batch since we ran an image with batch size of 1,
        # convert to normal python types with tolist(), and convert any byte strings to normal strings with .decode()
        out_keys = ["label", "confidence"]
        results = {}
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output

# create onnx model
dir_path = os.getcwd()
model = ONNXModel(dir_path=dir_path)
model.load()

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        req = request.get_json(force=True)
        image_data = req.get("image")
        image_data = re.sub(r"^data:image/.+;base64,", "", image_data)
        image_base64 = bytearray(image_data, "utf8")
        
        
        image = Image.open(io.BytesIO(base64.decodebytes(image_base64)))

        
        img = model.process_image(image, model.signature.get("inputs").get("Image").get("shape"))

        
        result = model.predict(Image.fromarray((img[0] * 255).astype(np.uint8)))

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

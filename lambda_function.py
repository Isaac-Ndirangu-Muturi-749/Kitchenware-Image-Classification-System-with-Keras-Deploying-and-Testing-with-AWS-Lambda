import os
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import json

MODEL_NAME = os.getenv('MODEL_NAME', 'kitchenware_model.tflite')

# Load the TF-Lite model
interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

target_size = (224, 224)
classes = np.array(['cup', 'fork', 'glass', 'knife', 'plate', 'spoon'])

def download_image(url):
    with request.urlopen(url) as resp:
        img_data = resp.read()
    return Image.open(BytesIO(img_data))

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    img = download_image(url)
    preprocessed_img = prepare_image(img, target_size)

    # Convert the PIL image to a NumPy array
    image_array = np.array(preprocessed_img)

    # Preprocess the image array (assuming normalization is required)
    normalized_image = (image_array / 255.0).astype(np.float32)

    # Reshape the image
    desired_input_shape = (1, 224, 224, 3)
    reshaped_image = np.reshape(normalized_image, desired_input_shape)

    # Set the input tensor of the TF-Lite model
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.tensor(input_tensor_index)()[0] = reshaped_image

    # Run the interpreter
    interpreter.invoke()

    # Get the output tensor of the TF-Lite model
    output_tensor_index = interpreter.get_output_details()[0]['index']
    model_output = interpreter.tensor(output_tensor_index)()

    # Get probabilities for all classes
    probabilities = [float(score) for score in model_output[0]]

    # Get the class label with the highest confidence score
    predicted_class_index = np.argmax(model_output)
    confidence_score = float(model_output[0, predicted_class_index])
    predicted_class = classes[predicted_class_index]

    # Return the result
    return {
        "Predicted_class": predicted_class,
        "Confidence_score": confidence_score,
        "Probabilities": {class_label: prob for class_label, prob in zip(classes, probabilities)}
    }

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    print(result)

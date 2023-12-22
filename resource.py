from PIL import Image
import io
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
from google.cloud import storage
import base64

storage_client = storage.Client()
bucket = storage_client.get_bucket('run-model-machinelearning')
blob = bucket.blob('Xception_model.tflite')
blob.download_to_filename('/tmp/Xception_model.tflite')

# Create preprocessor for the model
preprocessor = create_preprocessor('xception', target_size=(224, 224))

# Create Interpreter that load the model
interpreter = tflite.Interpreter(model_path='/tmp/Xception_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
   "No-Rust",
    "Rusty"
]


# Predict Function
def predict(request):
    data = request.get_json()
    url = data['url']
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))
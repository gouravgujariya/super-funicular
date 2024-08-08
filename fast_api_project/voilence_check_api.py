
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
from PIL import Image
import numpy as np
import onnxruntime as ort
import logging
import json
import base64
import cv2
import os  # Import os for file operations
from datetime import datetime  # Import datetime for unique filenames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the ONNX model
model_path = "C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\model\\yolo_voilence.onnx"
try:
    session = ort.InferenceSession(model_path)
    logger.info("ONNX model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    raise RuntimeError(f"Failed to load ONNX model: {e}")

# Get input and output details
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define the correct image size based on the model's expected input
IMAGE_SIZE = (128, 128)  # Update to the correct size for your model

# Initialize the FastAPI app
app = FastAPI()

# Create an output directory if it doesn't exist
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

class Prediction(BaseModel):
    class_name: str
    confidence: float

def process_frame(frame):
    try:
        logger.info("Processing frame...")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Ensure the image has the right shape (batch_size, channels, height, width)
        if image_array.shape[-1] == 3:  # Convert to (channels, height, width) if needed
            image_array = np.transpose(image_array, (2, 0, 1))

        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Run inference
        outputs = session.run([output_name], {input_name: image_array})
        output_data = outputs[0][0]  # Assuming batch size of 1

        confidence_scores = np.exp(output_data) / np.sum(np.exp(output_data))
        class_names = ['non_violence', 'violence']  # Replace with your actual class names
        results = [Prediction(class_name=class_names[idx], confidence=float(confidence_scores[idx])) for idx in
                   range(len(class_names))]

        # Determine the class with the highest confidence
        best_result = max(results, key=lambda x: x.confidence)
        class_name = best_result.class_name
        confidence = best_result.confidence

        # # Save the frame with class and confidence in the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Unique filename with timestamp
        output_path = os.path.join(output_dir, f"frame_{timestamp}_{class_name}_{confidence:.2f}.jpg")
        cv2.imwrite(output_path, frame)
        logger.info(f"Frame saved to {output_path}")

        return results
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"error": f"Error processing frame: {e}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                try:
                    # Decode base64 to bytes
                    frame_bytes = base64.b64decode(data)

                    # Convert bytes to numpy array
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is None:
                        await websocket.send_text(json.dumps({"error": "Failed to decode frame."}))
                        continue

                    results = process_frame(frame)
                    if isinstance(results, dict) and "error" in results:
                        await websocket.send_text(json.dumps(results))
                    else:
                        await websocket.send_text(json.dumps([result.dict() for result in results]))

                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    await websocket.send_text(json.dumps({"error": f"Error processing frame: {e}"}))
    except WebSocketDisconnect:
        logger.info(f"Client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket.send_text(json.dumps({"error": f"Unexpected error: {e}"}))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

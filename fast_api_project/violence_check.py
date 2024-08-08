import cv2
import asyncio
import websockets
import json
import base64

async def send_video_frames(video_path, websocket_url):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    async with websockets.connect(websocket_url) as websocket:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Encode bytes to base64
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

            # Send frame to the server
            await websocket.send(frame_base64)

            # Receive response from the server
            response = await websocket.recv()
            print(json.loads(response))

    cap.release()

if __name__ == "__main__":
    video_path = "C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\data\\samplefight1 - Made with Clipchamp.mp4"
    websocket_url = "ws://127.0.0.1:8000/ws"
    asyncio.run(send_video_frames(video_path, websocket_url))

#img
# import cv2
# import asyncio
# import websockets
# import json
# import base64
#
# async def send_image(image_path, websocket_url):
#     # Read the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error opening image file")
#         return
#
#     async with websockets.connect(websocket_url) as websocket:
#         # Encode the image as JPEG
#         _, buffer = cv2.imencode('.jpg', image)
#         image_bytes = buffer.tobytes()
#
#         # Encode bytes to base64
#         image_base64 = base64.b64encode(image_bytes).decode('utf-8')
#
#         # Send image to the server
#         await websocket.send(image_base64)
#
#         # Receive response from the server
#         response = await websocket.recv()
#         print(json.loads(response))
#
# if __name__ == "__main__":
#     image_path = "C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\data\\blood.jpeg"
#     websocket_url = "ws://127.0.0.1:8000/ws"
#     asyncio.run(send_image(image_path, websocket_url))

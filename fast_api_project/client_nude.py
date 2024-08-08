
# this is for video

import asyncio
import websockets
import cv2

async def send_video(uri):
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture("C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\data\\normal1.mp4")  # Use your webcam or a video file
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _, buffer = cv2.imencode('.jpg', frame)
                await websocket.send(buffer.tobytes())
                response = await websocket.recv()
                print(response)
        except websockets.ConnectionClosedError as e:
            print(f"Connection closed with error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            cap.release()

async def main():
    uri = 'ws://localhost:8000/ws'
    while True:
        try:
            await send_video(uri)
        except Exception as e:
            print(f"Error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

asyncio.run(main())

#
# # this is for the images
#
# import asyncio
# import websockets
# import json
#
#
# async def send_image(image_path):
#     uri = "ws://localhost:8000/ws"
#     async with websockets.connect(uri) as websocket:
#         with open(image_path, "rb") as image_file:
#             image_data = image_file.read()
#
#         # Send the image data
#         await websocket.send(image_data)
#
#         # Receive the prediction results
#         response = await websocket.recv()
#         print("Prediction results:", response)
#
#
# # Run the client
# image_path = "C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\data\\chess.png"  # Update with the path to your image file
# asyncio.run(send_image(image_path))

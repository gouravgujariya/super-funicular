from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from nudenet import NudeClassifier
import cv2
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

app = FastAPI()

classifier = NudeClassifier()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


def process_frame(frame_bytes):
    try:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None  # or raise an exception, depending on your needs
        classification = classifier.classify(frame)

        # Ensure the classification dictionary contains 'labels'
        if 'labels' in classification:
            return {'classification': 'unsafe', 'labels': classification['labels']}
        else:
            return {'classification': classification}
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, process_frame, data)
            if result:
                await manager.send_message(str(result), websocket)
            else:
                await manager.send_message("no nudity detected", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
        await manager.send_message("Unexpected error", websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
from openai import OpenAI
import os
import matplotlib.pyplot as plt
import numpy as np
import time

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

video = cv2.VideoCapture("new_video/Complex_Video.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

# Can edit prompt here
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to upload. How many red vehicles can be seen?.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::5]),
        ],
    },
]
params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 200,
}

start = time.perf_counter()
result = client.chat.completions.create(**params)
end = time.perf_counter()
latency = end - start
print(result.choices[0].message.content)
print(f"Latency: {latency}")

# Assuming base64Frames is your list of base64 encoded frames
for b64_frame in base64Frames[0::5]:
    # Decode the base64 frame
    img_bytes = base64.b64decode(b64_frame)
    # Convert bytes to numpy array
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    # Decode numpy array into image
    img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
    # Convert color style from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use plt to display the image
    plt.imshow(img)
    plt.show()
import cv2
from moviepy.editor import VideoFileClip
import os
import base64
from openai import OpenAI
import time

# We'll be using the OpenAI DevDay Keynote Recap video. You can review the video here: https://www.youtube.com/watch?v=h02ti0Bl6zk
VIDEO_PATH = "new_video/Complex_Video.mp4"
MODEL="gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
video = cv2.VideoCapture(VIDEO_PATH)

# def process_video(video_path, seconds_per_frame=2):
#     base64Frames = []
#     base_video_path, _ = os.path.splitext(video_path)

#     video = cv2.VideoCapture(video_path)
#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = video.get(cv2.CAP_PROP_FPS)
#     frames_to_skip = int(fps * seconds_per_frame)
#     curr_frame=0

#     # Loop through the video and extract frames at specified sampling rate
#     while curr_frame < total_frames - 1:
#         video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
#         success, frame = video.read()
#         if not success:
#             break
#         _, buffer = cv2.imencode(".jpg", frame)
#         base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
#         curr_frame += frames_to_skip
#     video.release()

#     return base64Frames

# # Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
# base64Frames = process_video(VIDEO_PATH, seconds_per_frame=24)
# print(f"Extracted Frames: {len(base64Frames)}")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

start = time.perf_counter()
response = client.chat.completions.create(
    model=MODEL,
    messages=[
    {"role": "system", "content": "You are looking at a cctv video. What is happening in the video? How many red cars can be seen?."},
    {"role": "user", "content": [
        "These are the frames from the video.",
        *map(lambda x: {"type": "image_url", 
                        "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames[0::5])
        ],
    }
    ],
    temperature=0,
)
end = time.perf_counter()
latency = end - start
print(response.choices[0].message.content)
print(f"Latency for API Call: {latency}")


# import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
# import base64
# from openai import OpenAI
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# video = cv2.VideoCapture("new_video/Complex_Video.mp4")

# base64Frames = []
# while video.isOpened():
#     success, frame = video.read()
#     if not success:
#         break
#     _, buffer = cv2.imencode(".jpg", frame)
#     base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

# video.release()
# print(len(base64Frames), "frames read.")

# # Can edit prompt here
# PROMPT_MESSAGES = [
#     {
#         "role": "user",
#         "content": [
#             "You are looking at a cctv video. What is happening in the video? How many red cars can be seen?.",
#             *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::5]),
#         ],
#     },
# ]
# params = {
#     "model": "gpt-4o",
#     "messages": PROMPT_MESSAGES,
#     "max_tokens": 200,
# }

# start = time.perf_counter()
# result = client.chat.completions.create(**params)
# end = time.perf_counter()
# latency = end - start
# print(result.choices[0].message.content)
# print(f"Latency: {latency}")


# # for b64_frame in base64Frames[0::5]:
# #     # Decode the base64 frame
# #     img_bytes = base64.b64decode(b64_frame)
# #     # Convert bytes to numpy array
# #     img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
# #     # Decode numpy array into image
# #     img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
# #     # Convert color style from BGR to RGB
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     # Use plt to display the image
# #     plt.imshow(img)
# #     plt.show()
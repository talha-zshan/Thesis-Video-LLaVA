import os
import cv2
import glob
import autogen
import asyncio
import time

# Autogen Imports
from autogen import UserProxyAgent
from typing_extensions import Annotated
from autogen.cache import Cache

# Kafka Imports
from Kafka import KafkaManager
from Gpt_API import call_gpt_model
from Extractor import extract_frames

# Setup Configs
llm_config = {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}
kafka_config = {'bootstrap.servers': 'localhost:9092'}
kafka_manager = KafkaManager(kafka_config)


### ======================================== AUTOGEN AGENTS =========================================================================

reflector = autogen.ConversableAgent(
    name="Decider",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=100,
    code_execution_config=False,
    llm_config=llm_config,
    system_message="""You act on behalf of the Admin and work with Engineer-1 and Engineer-2 to complete the task. You can converse with the Admin agent. You always respond with TERMINATE""",
)

engineer_1 = autogen.AssistantAgent(
    name="Engineer-1",
    llm_config=llm_config,
    system_message="""
    I'm Engineer. I'm expert in python programming. I'm executing code tasks required by the Decider.
    """,
)

engineer_2 = autogen.AssistantAgent(
    name="Engineer-2",
    llm_config=llm_config,
    system_message="""
    I'm Engineer. I'm expert in python programming. I'm executing code tasks required by the Decider.
    """,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A proxy for the user for executing code.",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=100,
    code_execution_config=False,
)

### ======================================== VIDEO FRAME FUNCTIONS =========================================================================

# Kafka Consumer Function
@reflector.register_for_execution()
@engineer_1.register_for_llm(description="Get the path to the video directory by consuming a message from a kafka broker. The topic is inferred from the user prompt")
def kafka_consume(topic: Annotated[str, "The path to the directory where multiple videos are stored. Use default value as None"]):
    msg = kafka_manager.consume_messages(topic)
    print(msg)
    return msg

# Frame Extraction Function
@reflector.register_for_execution()
@engineer_1.register_for_llm(description="Extract frames from a video whose title is provided by the user. The video path is taken from the output of kafka_consume function")
def extract_video_frames(
    video_path: Annotated[str, "The path to the directory where the video is stored. The default directory is demo_videos/. Add the title of the video by the user to the default directory to get the full path."], 
    output_root: Annotated[str, "The path to the directory where the output is stored. The default directory to use is extracted_frames/"], 
    ):

    # Timer Start
    func_start = time.perf_counter()
    output_dir = extract_frames(video_path, output_root)
    # Timer end 
    func_end = time.perf_counter()
    func_latency = func_end - func_start
    print(f"Latency For extract_video_frames: {func_latency:.6f} seconds")
    return output_dir

# Video Creation Function 
@reflector.register_for_execution()
@engineer_1.register_for_llm(description="Create a video using the frames stored in the directory path returned. Save the video with .mp4 extension in the new_video directory. Dont prompt for user input after the function is executed.")
def create_video_from_frames(
    frames_path: Annotated[str, "The path to the directory where the frames are stored. Use thhe path returned by the extract_video_frames function"], 
    output_video_path: Annotated[str, "Add the title of the video by the user with extension set to .mp4 to get the full path"], 
    frame_rate=24.0
    ):

    # Start Timer
    func_start = time.perf_counter()


    # Get all the frame filenames in the directory
    frame_files = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))

    # Read the first frame to determine the width and height
    frame = cv2.imread(frame_files[0])
    # height, width, layers = frame.shape
    width=640
    height=360
    

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Loop through all the frames and add them to the video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
            
        if frame is not None:
            try:
                resized_frame = cv2.resize(frame, (width, height))
                video.write(resized_frame)
            except Exception as e:
                print(e)
                return "TERMINATE"

    # Release the video writer
    video.release()

    # End Timer
    func_end = time.perf_counter()
    func_latency = func_end - func_start
    print(f"Latency For create_video_from_frames: {func_latency:.6f} seconds")

    return f"Success! Video Created & Stored in {output_video_path}"


@reflector.register_for_execution()
@engineer_2.register_for_llm(description="Use the video path of the newly created video sent by Engineer-1 and the original prompt sent by the Admin")
def call_model(
        video_path: Annotated[str, ""],
        user_message: Annotated[str, ""]
):
    response = call_gpt_model(video_path, user_message)

    return response


## ======================== SPEAKER SELECTION ==================================
def custom_speaker_selection_func(last_speaker: autogen.Agent, groupchat: autogen.GroupChat):
    """Define a customized speaker selection function based on the FSM diagram.

    Returns:
        Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
    """
    messages = groupchat.messages

    if len(messages) <= 1:
        return user_proxy

    if last_speaker is user_proxy:
        if "RECHECK" in messages[-1]["content"]:
            # If the last message contains the keyword "RECHECK", select Engineer-1 Agent
            return engineer_1
        elif "QUERY" in messages[-1]["content"]:
            # If the last message contains the keyword "QUERY", select Engineer-2 Agent
            return engineer_2
        else:
            # No special keyword, continue with User Proxy Agent
            return user_proxy

    elif last_speaker is engineer_1:
        # After Engineer-1 Agent, always follow with Engineer-2 Agent
        return engineer_2

    elif last_speaker is engineer_2:
        # After Engineer-2 Agent, return control to the Reflection Agent for further decisions
        return reflector

    elif last_speaker is reflector:
        if "RECHECK" in messages[-1]["content"]:
            # If the last message contains the keyword "RECHECK", select Engineer-1 Agent
            return engineer_1
        elif "QUERY" in messages[-1]["content"]:
            # If the last message contains the keyword "QUERY", select Engineer-2 Agent
            return engineer_2
        else:
            # No special keyword, continue with User Proxy Agent
            return user_proxy


### ======================================== AUTOGEN GROUP-CHAT INITIALIZATION =========================================================================
groupchat = autogen.GroupChat(
    agents=[engineer_1, engineer_2, reflector,user_proxy],
    messages=[],
    max_round=500,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


async def main():
    with Cache.disk() as cache:
        await user_proxy.a_initiate_chat(  # noqa: F704
            manager,
            message="""RECHECK, What is happening in the video in camera-1""",
            cache=cache,
        )


if __name__ == "__main__":
    # create_video_from_frames("extracted_frames/Complex_Video_frames", "new_video")
   start_time = time.perf_counter()
   asyncio.run(main())
   end_time = time.perf_counter()

   latency = end_time - start_time

   print(f"Latency For Main: {latency:.6f} seconds")
    
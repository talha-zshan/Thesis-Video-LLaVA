import os
import math
import json
import cv2
import glob
import autogen
import asyncio
import torch

# Autogen Imports
from autogen import UserProxyAgent, AssistantAgent
from pathlib import Path
from autogen import UserProxyAgent, AssistantAgent
from typing_extensions import Annotated
from autogen.cache import Cache

# Video LL-AVA imports
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# Kafka Imports
from Kafka import KafkaManager


# Setup Configs
llm_config = {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}
kafka_config = {'bootstrap.servers': 'localhost:9092'}
kafka_manager = KafkaManager(kafka_config)
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)


### ======================================== VIDEO FRAME FUNCTIONS =========================================================================

coder_agent = AssistantAgent(
    name="coderbot",
    system_message="For coding tasks, only use the functions you have been provided with. The functions are written in python. You have a frame_extractor and a video_creator, these tools should be used sequentially with frame_extractor used first. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    system_message="""
    I'm Engineer. I'm expert in python programming. I'm executing code tasks required by Admin.
    """,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A proxy for the user for executing code.",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=15,
    code_execution_config=False,
)

### ======================================== VIDEO FRAME FUNCTIONS =========================================================================

# Kafka Consumer Function
@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Get the path to the video directory by consuming a message from a kafka broker. The topic is inferred from the user prompt")
def kafka_consume(topic: Annotated[str, "The path to the directory where multiple videos are stored. Use default value as None"]):
    msg = kafka_manager.consume_messages(topic)
    print(msg)
    return msg

# Frame Extraction Function
@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Extract frames from a video whose title is provided by the user. The video path is taken from the output of kafka_consume function")
def extract_video_frames(
    video_path: Annotated[str, "The path to the directory where the video is stored. The default directory is demo_videos/. Add the title of the video by the user to the default directory to get the full path."], 
    dir_path: Annotated[str, "The path to the directory where multiple videos are stored. Use default value as None"], 
    sampling: Annotated[int, "The sampling rate to be used. Use default value as 2"], 
    output_root: Annotated[str, "The path to the directory where the output is stored. The default directory to use is extracted_frames/"], 
    workers=Annotated[int, "The number of workers to be used. Use default value as None"]
    ):
    import os.path as osp
    from multiprocessing import Pool
    
    # Supported video and frame extensions
    supported_video_ext = ('.avi', '.mp4')
    supported_frame_ext = ('.jpg', '.png')
    
    class FrameExtractor:
        def __init__(self, video_file, output_dir, frame_ext='.jpg', sampling=-1):
            if not osp.exists(video_file):
                raise FileExistsError('Video file {} does not exist.'.format(video_file))
            self.video_file = video_file
            self.output_dir = output_dir
            if not osp.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if frame_ext not in supported_frame_ext:
                raise ValueError("Not supported frame file format: {}".format(frame_ext))
            self.frame_ext = frame_ext
            self.sampling = sampling
            self.video = cv2.VideoCapture(self.video_file)
            self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
            self.video_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.sampling != -1:
                self.video_length = self.video_length // self.sampling

        def extract(self):
            success, frame = self.video.read()
            frame_cnt = 0
            while success:
                curr_frame_filename = osp.join(self.output_dir, "{:08d}{}".format(frame_cnt, self.frame_ext))
                cv2.imwrite(curr_frame_filename, frame)
                success, frame = self.video.read()
                if self.sampling != -1:
                    frame_cnt += math.ceil(self.sampling * self.video_fps)
                    self.video.set(1, frame_cnt)
                else:
                    frame_cnt += 1

    def process_video_file(video_file, output_dir, sampling):
        if os.stat(video_file).st_size > 0:
            if not osp.isdir(output_dir):
                extractor = FrameExtractor(video_file=video_file, output_dir=output_dir, sampling=sampling)
                extractor.extract()
        else:
            os.remove(video_file)

    if video_path:
        video_basename = osp.basename(video_path).split('.')[0]
        video_ext = osp.splitext(video_path)[-1]
        if video_ext not in supported_video_ext:
            raise ValueError("Not supported video file format: {}".format(video_ext))
        output_dir = osp.join(output_root, '{}_frames'.format(video_basename))
        process_video_file(video_path, output_dir, sampling)
        return output_dir

    if dir_path:
        video_list = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_ext = osp.splitext(file)[-1]
                if file_ext in supported_video_ext:
                    video_file_path = osp.join(root, file)
                    output_dir = osp.join(output_root, osp.relpath(root, dir_path), '{}_frames'.format(osp.splitext(file)[0]))
                    video_list.append((video_file_path, output_dir, sampling))

        with Pool(processes=workers) as pool:
            pool.starmap(process_video_file, video_list)

# Video Creation Function 
@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Create a video using the frames stored in the directory path returned. Save the video with .mp4 extension in the new_video directory. Dont prompt for user input after the function is executed.")
def create_video_from_frames(
    frames_path: Annotated[str, "The path to the directory where the frames are stored. Use thhe path returned by the extract_video_frames function"], 
    output_video_path: Annotated[str, "Add the title of the video by the user with extension set to .mp4 to get the full path"], 
    frame_rate=24.0
    ):
    # Get all the frame filenames in the directory
    frame_files = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))

    # Read the first frame to determine the width and height
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Loop through all the frames and add them to the video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video.write(frame)

    # Release the video writer
    video.release()

    return f"Success! Video Created & Stored in {output_video_path}"

# Video-LLava Call Function
@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Send the video path of the newly created video and the original prompt sent by the user. Loop on this function and wait for more prompts from the user. Update the prompt parameter according to the user. If user enters exit, then terminate the function.")
def call_video_llava(
    video_path: Annotated[str, "The path to the directory where the video is stored. Use the path returned by the create_video_from_frames function"], 
    user_input: Annotated[str, "The prompt sent by the user"]
    ):
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    while True:
        video = video_path
        inp = input("Enter your prompt (type 'exit' to quit): ")
        if inp.lower() == 'exit':
            break
        
        video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
        if isinstance(video_tensor, list):
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)
        
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(outputs)
    return outputs


groupchat = autogen.GroupChat(
    agents=[engineer, user_proxy],
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
            message="""What is happening in the video in camera-1""",
            cache=cache,
        )


if __name__ == "__main__":
   asyncio.run(main())
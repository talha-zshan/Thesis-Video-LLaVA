import os
import cv2
import time
import glob
import threading
import sys
import anyio
import gradio as gr

# Autogen Imports
import autogen
from autogen import UserProxyAgent
from pathlib import Path
from autogen.code_utils import extract_code

# Gradio
from gradio import ChatInterface, Request
from gradio.helpers import special_args

# Util Imports
from Kafka import KafkaManager
from Extractor import extract_video_frames
from typing_extensions import Annotated

# Setup Configs
LOG_LEVEL = "INFO"
TIMEOUT = 60
llm_config = {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}
kafka_config = {'bootstrap.servers': 'localhost:9092'}
kafka_manager = KafkaManager(kafka_config)
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)


### ======================================== AUTOGEN AGENTS =========================================================================


import os
import sys
import threading
from itertools import chain

import anyio
import autogen
import gradio as gr
from autogen import Agent, AssistantAgent, OpenAIWrapper, UserProxyAgent
from autogen.code_utils import extract_code
from gradio import ChatInterface, Request
from gradio.helpers import special_args

LOG_LEVEL = "INFO"
TIMEOUT = 60


class myChatInterface(ChatInterface):
    async def _submit_fn(
        self,
        message: str,
        history_with_input: list[list[str | None]],
        request: Request,
        *args,
    ) -> tuple[list[list[str | None]], list[list[str | None]]]:
        history = history_with_input[:-1]
        inputs, _, _ = special_args(self.fn, inputs=[message, history, *args], request=request)

        if self.is_async:
            await self.fn(*inputs)
        else:
            await anyio.to_thread.run_sync(self.fn, *inputs, limiter=self.limiter)

        # history.append([message, response])
        return history, history


with gr.Blocks() as demo:

    def flatten_chain(list_of_lists):
        return list(chain.from_iterable(list_of_lists))

    class thread_with_trace(threading.Thread):
        # https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
        # https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
        def __init__(self, *args, **keywords):
            threading.Thread.__init__(self, *args, **keywords)
            self.killed = False
            self._return = None

        def start(self):
            self.__run_backup = self.run
            self.run = self.__run
            threading.Thread.start(self)

        def __run(self):
            sys.settrace(self.globaltrace)
            self.__run_backup()
            self.run = self.__run_backup

        def run(self):
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)

        def globaltrace(self, frame, event, arg):
            if event == "call":
                return self.localtrace
            else:
                return None

        def localtrace(self, frame, event, arg):
            if self.killed:
                if event == "line":
                    raise SystemExit()
            return self.localtrace

        def kill(self):
            self.killed = True

        def join(self, timeout=0):
            threading.Thread.join(self, timeout)
            return self._return

    def update_agent_history(recipient, messages, sender, config):
        if config is None:
            config = recipient
        if messages is None:
            messages = recipient._oai_messages[sender]
        message = messages[-1]
        message.get("content", "")
        # config.append(msg) if msg is not None else None  # config can be agent_history
        return False, None  # required to ensure the agent communication flow continues

    def _is_termination_msg(message):
        """Check if a message is a termination message.
        Terminate when no code block is detected. Currently only detect python code blocks.
        """
        if isinstance(message, dict):
            message = message.get("content")
            if message is None:
                return False
        cb = extract_code(message)
        contain_code = False
        for c in cb:
            # todo: support more languages
            if c[0] == "python":
                contain_code = True
                break
        return not contain_code

    def initialize_agents(config_list):
        assistant = AssistantAgent(
            name="assistant",
            max_consecutive_auto_reply=10,
            system_message="""
            I'm Engineer. I'm expert in python programming. I'm executing code tasks required by the userproxy.
            """,
            llm_config={
                # "seed": 42,
                "timeout": TIMEOUT,
                "config_list": config_list,
            },
        )

        userproxy = UserProxyAgent(
            name="userproxy",
            system_message="A proxy for the user for executing code.",
            human_input_mode="NEVER",
            is_termination_msg=_is_termination_msg,
            max_consecutive_auto_reply=5,
            code_execution_config=False,
        )

        # assistant.register_reply([Agent, None], update_agent_history)
        # userproxy.register_reply([Agent, None], update_agent_history)

        return assistant, userproxy

    def chat_to_oai_message(chat_history):
        """Convert chat history to OpenAI message format."""
        messages = []
        if LOG_LEVEL == "DEBUG":
            print(f"chat_to_oai_message: {chat_history}")
        for msg in chat_history:
            messages.append(
                {
                    "content": msg[0].split()[0] if msg[0].startswith("exitcode") else msg[0],
                    "role": "user",
                }
            )
            messages.append({"content": msg[1], "role": "assistant"})
        return messages

    def oai_message_to_chat(oai_messages, sender):
        """Convert OpenAI message format to chat history."""
        chat_history = []
        messages = oai_messages[sender]
        if LOG_LEVEL == "DEBUG":
            print(f"oai_message_to_chat: {messages}")
        for i in range(0, len(messages), 2):
            chat_history.append(
                [
                    messages[i]["content"],
                    messages[i + 1]["content"] if i + 1 < len(messages) else "",
                ]
            )
        return chat_history

    def agent_history_to_chat(agent_history):
        """Convert agent history to chat history."""
        chat_history = []
        for i in range(0, len(agent_history), 2):
            chat_history.append(
                [
                    agent_history[i],
                    agent_history[i + 1] if i + 1 < len(agent_history) else None,
                ]
            )
        return chat_history

    def initiate_chat(config_list, user_message, chat_history):
        if LOG_LEVEL == "DEBUG":
            print(f"chat_history_init: {chat_history}")
        # agent_history = flatten_chain(chat_history)
        if len(config_list[0].get("api_key", "")) < 2:
            chat_history.append(
                [
                    user_message,
                    "Hi, nice to meet you! Please enter your API keys in below text boxs.",
                ]
            )
            return chat_history
        else:
            llm_config = {
                # "seed": 42,
                "timeout": TIMEOUT,
                "config_list": config_list,
            }
            assistant.llm_config.update(llm_config)
            assistant.client = OpenAIWrapper(**assistant.llm_config)

        if user_message.strip().lower().startswith("show file:"):
            filename = user_message.strip().lower().replace("show file:", "").strip()
            filepath = os.path.join("coding", filename)
            if os.path.exists(filepath):
                chat_history.append([user_message, (filepath,)])
            else:
                chat_history.append([user_message, f"File {filename} not found."])
            return chat_history

        assistant.reset()
        oai_messages = chat_to_oai_message(chat_history)
        assistant._oai_system_message_origin = assistant._oai_system_message.copy()
        assistant._oai_system_message += oai_messages

        try:
            userproxy.initiate_chat(assistant, message=user_message)
            messages = userproxy.chat_messages
            chat_history += oai_message_to_chat(messages, assistant)
            # agent_history = flatten_chain(chat_history)
        except Exception as e:
            # agent_history += [user_message, str(e)]
            # chat_history[:] = agent_history_to_chat(agent_history)
            chat_history.append([user_message, str(e)])

        assistant._oai_system_message = assistant._oai_system_message_origin.copy()
        if LOG_LEVEL == "DEBUG":
            print(f"chat_history: {chat_history}")
            # print(f"agent_history: {agent_history}")
        return chat_history

    def chatbot_reply_thread(input_text, chat_history, config_list):
        """Chat with the agent through terminal."""
        thread = thread_with_trace(target=initiate_chat, args=(config_list, input_text, chat_history))
        thread.start()
        try:
            messages = thread.join(timeout=TIMEOUT)
            if thread.is_alive():
                thread.kill()
                thread.join()
                messages = [
                    input_text,
                    "Timeout Error: Please check your API keys and try again later.",
                ]
        except Exception as e:
            messages = [
                [
                    input_text,
                    str(e) if len(str(e)) > 0 else "Invalid Request to OpenAI, please check your API keys.",
                ]
            ]
        return messages

    def chatbot_reply_plain(input_text, chat_history, config_list):
        """Chat with the agent through terminal."""
        try:
            messages = initiate_chat(config_list, input_text, chat_history)
        except Exception as e:
            messages = [
                [
                    input_text,
                    str(e) if len(str(e)) > 0 else "Invalid Request to OpenAI, please check your API keys.",
                ]
            ]
        return messages

    def chatbot_reply(input_text, chat_history, config_list):
        """Chat with the agent through terminal."""
        return chatbot_reply_thread(input_text, chat_history, config_list)

    def get_description_text():
        return """
        # Microsoft AutoGen: Multi-Round Human Interaction Chatbot Demo

        This demo shows how to build a chatbot which can handle multi-round conversations with human interactions.

        #### [AutoGen](https://github.com/microsoft/autogen)  [Discord](https://discord.gg/pAbnFJrkgZ)  [Paper](https://arxiv.org/abs/2308.08155)  [SourceCode](https://github.com/thinkall/autogen-demos)
        """

    def update_config():
        config_list = autogen.config_list_from_models(
            model_list=[os.environ.get("MODEL", "gpt-4o")],
        )
        if not config_list:
            config_list = [
                {
                    "api_key": "",
                    "base_url": "",
                    "api_type": "azure",
                    "api_version": "2023-07-01-preview",
                    "model": "gpt-3.5-turbo",
                }
            ]

        return config_list

    def set_params(model, oai_key, aoai_key, aoai_base):
        os.environ["MODEL"] = model
        os.environ["OPENAI_API_KEY"] = oai_key
        os.environ["AZURE_OPENAI_API_KEY"] = aoai_key
        os.environ["AZURE_OPENAI_API_BASE"] = aoai_base

    def respond(message, chat_history, model, oai_key, aoai_key, aoai_base):
        set_params(model, oai_key, aoai_key, aoai_base)
        config_list = update_config()
        chat_history[:] = chatbot_reply(message, chat_history, config_list)
        if LOG_LEVEL == "DEBUG":
            print(f"return chat_history: {chat_history}")
        return ""

    config_list, assistant, userproxy = (
        [
            {
                "api_key": "",
                "base_url": "",
                "api_type": "azure",
                "api_version": "2023-07-01-preview",
                "model": "gpt-4o",
            }
        ],
        None,
        None,
    )
    
    assistant, userproxy = initialize_agents(config_list)

    @userproxy.register_for_execution()
    @assistant.register_for_llm(description="Extract frames from a video whose title is provided by the user. The video path is taken from the output of kafka_consume function")
    def extract_frames(
        video_path: Annotated[str, "The path to the directory where the video is stored. The default directory is demo_videos/. Add the title of the video by the user to the default directory to get the full path."], 
        sampling: Annotated[int, "The sampling rate to be used. Use default value as 1"], 
        output_root: Annotated[str, "The path to the directory where the output is stored. The default directory to use is extracted_frames/"], 
        ):

        # Timer Start
        extract_video_frames(video_path, output_root, sampling)
        return f"The frames have been extracted in the directory {output_root}"

    # Video Creation Function 
    @userproxy.register_for_execution()
    @assistant.register_for_llm(description="Create a video using the frames stored in the directory path returned. Save the video with .mp4 extension in the new_video directory. Dont prompt for user input after the function is executed.")
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

    description = gr.Markdown(get_description_text())

    with gr.Row() as params:
        txt_model = gr.Dropdown(
            label="Model",
            choices=[
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ],
            allow_custom_value=True,
            value="gpt-4o",
            container=True,
        )
        txt_oai_key = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter OpenAI API Key",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )
        txt_aoai_key = gr.Textbox(
            label="Azure OpenAI API Key",
            placeholder="Enter Azure OpenAI API Key",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )
        txt_aoai_base_url = gr.Textbox(
            label="Azure OpenAI API Base",
            placeholder="Enter Azure OpenAI Base Url",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(
            "human.png",
            (os.path.join(os.path.dirname(__file__), "autogen.png")),
        ),
        render=False,
        height=600,
    )

    txt_input = gr.Textbox(
        scale=4,
        show_label=False,
        placeholder="Enter text and press enter",
        container=False,
        render=False,
        autofocus=True,
    )

    chatiface = myChatInterface(
        respond,
        chatbot=chatbot,
        textbox=txt_input,
        additional_inputs=[
            txt_model,
            txt_oai_key,
            txt_aoai_key,
            txt_aoai_base_url,
        ],
        examples=[
            ["write a python function to count the sum of two numbers?"],
            ["what if the production of two numbers?"],
            [
                "Plot a chart of the last year's stock prices of Microsoft, Google and Apple and save to stock_price.png."
            ],
            ["show file: stock_price.png"],
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
    
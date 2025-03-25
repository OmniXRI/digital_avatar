import openvino as ov
import openvino_genai as ov_genai
from uuid import uuid4
from threading import Event, Thread
import queue
import sys
import re
from genai_helper import ChunkStreamer

max_new_tokens = 1024
core = ov.Core()

chinese_examples = [
 ["向 5 歲的孩子解釋重力。"],
 ["給我講一個關於微積分的笑話。"],
 ["編寫程式碼時需要避免哪些常見錯誤？"],
 ["撰寫一篇關於「人工智慧和 OpenVINO 的優勢」的 100 字部落格文章"],
 ["解方程式：2x + 5 = 15"],
 ["說出養貓的 5 個優點"],
 ["簡化 (-k + 4) + (-2 + 3k)"],
 ["求半徑為20的圓的面積"],
 ["對未來5年AI趨勢進行預測"],
]

english_examples = [
    ["Explain gravity to a 5-year-old."],
    ["Tell me a joke about calculus."],
    ["What are some common mistakes to avoid when writing code?"],
    ["Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"],
    ["Solve the equation: 2x + 5 = 15."],
    ["Name 5 advantages to be a cat"],
    ["Simplify (-k + 4) + (-2 + 3k)"],
    ["Find the area of ​​a circle with radius 20"],
    ["Make a forecast about AI trends for next 5 years"],
]


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_SYSTEM_PROMPT_CHINESE = """\
你是個樂於助人、尊重他人、誠實可靠的助手。在安全的情況下，始終盡可能有幫助地回答。 您的回答不應包含任何有害、不道德、種族主義、性別歧視、有毒、危險或非法的內容。請確保您的回答在社會上是公正的和積極的。
如果一個問題沒有任何意義或與事實不符，請解釋原因，而不是回答錯誤的問題。如果您不知道問題的答案，請不要分享虛假資訊。另外，答案請使用繁體中文。 \
"""


def get_system_prompt(model_language, system_prompt=None):
    if system_prompt is not None:
        return system_prompt
    return DEFAULT_SYSTEM_PROMPT_CHINESE if (model_language == "Chinese") else DEFAULT_SYSTEM_PROMPT


def make_demo(pipe, model_configuration, model_id, model_language, disable_advanced=False):
    import gradio as gr

    start_message = get_system_prompt(model_language, model_configuration.get("system_prompt"))
    if "genai_chat_template" in model_configuration:
        pipe.get_tokenizer().set_chat_template(model_configuration["genai_chat_template"])

    def get_uuid():
        """
        universal unique identifier for thread
        """
        return str(uuid4())

    def default_partial_text_processor(partial_text: str, new_text: str):
        """
        helper for updating partially generated answer, used by default

        Params:
        partial_text: text buffer for storing previosly generated text
        new_text: text update for the current step
        Returns:
        updated text string

        """
        new_text = re.sub(r"^<think>", "<em><small>I am thinking...", new_text)
        new_text = re.sub("</think>", "I think I know the answer</small></em>", new_text)
        partial_text += new_text
        return partial_text

    text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)

    def bot(message, history, temperature, top_p, top_k, repetition_penalty, max_tokens):
        """
        callback function for running chatbot on submit button click

        Params:
        message: new message from user
        history: conversation history
        temperature:  parameter for control the level of creativity in AI-generated text.
                        By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
        top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
        top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
        repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
        active_chat: chat state, if true then chat is running, if false then we should start it here.
        Returns:
        message: reset message and make it ""
        history: updated history with message and answer from chatbot
        active_chat: if we are here, the chat is running or will be started, so return True
        """
        streamer = ChunkStreamer(pipe.get_tokenizer())
        if not disable_advanced:
            config = pipe.get_generation_config()
            config.temperature = temperature
            config.top_p = top_p
            config.top_k = top_k
            config.do_sample = temperature > 0.0
            config.max_new_tokens = max_tokens
            config.repetition_penalty = repetition_penalty
            if "stop_strings" in model_configuration:
                config.stop_strings = set(model_configuration["stop_strings"])
        else:
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
        history = history or []
        if not history:
            pipe.start_chat(system_message=start_message)

        history.append([message, ""])
        new_prompt = message

        stream_complete = Event()

        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            streamer.reset()
            pipe.generate(new_prompt, config, streamer)
            stream_complete.set()
            streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            yield "", history, streamer

    def stop_chat(streamer):
        if streamer is not None:
            streamer.end()
        return None

    def stop_chat_and_clear_history(streamer):
        if streamer is not None:
            streamer.end()
        pipe.finish_chat()
        streamer.reset()
        return None, None

    examples = chinese_examples if (model_language == "Chinese") else english_examples

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        streamer = gr.State(None)
        conversation_id = gr.State(get_uuid)
        gr.Markdown(f"""<h1><center>OpenVINO {model_id} Chatbot</center></h1>""")
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")
        with gr.Row(visible=not disable_advanced):
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=50,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
                    with gr.Column():
                        with gr.Row():
                            max_tokens = gr.Slider(
                                label="Max new tokens",
                                value=256,
                                minimum=128,
                                maximum=1024,
                                step=32,
                                interactive=True,
                                info=("Maximum new tokens added to answer. Higher value can work for long response, but require more time to complete"),
                            )
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")

        msg.submit(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty, max_tokens],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        submit.click(
            fn=bot,
            inputs=[msg, chatbot, temperature, top_p, top_k, repetition_penalty, max_tokens],
            outputs=[msg, chatbot, streamer],
            queue=True,
        )
        clear.click(fn=stop_chat_and_clear_history, inputs=streamer, outputs=[chatbot, streamer], queue=False)

        return demo

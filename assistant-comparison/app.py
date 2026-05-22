import os

import gradio as gr
from assistant import FrontierAssistant, OpenSourceAssistant, summary_message

open_source_assistant = OpenSourceAssistant()
frontier_assistant = None
if os.getenv("OPENAI_API_KEY"):
    try:
        frontier_assistant = FrontierAssistant()
    except Exception:
        frontier_assistant = None


def add_message(history, user_message, assistant_message):
    history = history + [(user_message, assistant_message)]
    return history, history


def respond_oss(user_message, history):
    if not user_message or user_message.strip() == "":
        return history, history
    response = open_source_assistant.generate(history, user_message)
    return add_message(history, user_message, response)


def respond_frontier(user_message, history):
    if not user_message or user_message.strip() == "":
        return history, history
    if frontier_assistant is None:
        error = "Frontier assistant is not configured. Set OPENAI_API_KEY to enable it."
        return history + [(user_message, error)], history + [(user_message, error)]
    response = frontier_assistant.generate(history, user_message)
    return add_message(history, user_message, response)


def clear_state():
    return []

with gr.Blocks(title="Assistant Comparison", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Assistant Comparison Demo")
    gr.Markdown(summary_message())
    with gr.Tabs():
        with gr.TabItem("Open Source Assistant"):
            oss_chatbot = gr.Chatbot(label="OSS Conversation")
            oss_input = gr.Textbox(label="Ask the open-source assistant", placeholder="Type your message...")
            oss_history = gr.State([])
            with gr.Row():
                oss_submit = gr.Button("Send")
                oss_clear = gr.Button("Reset")
            oss_submit.click(respond_oss, [oss_input, oss_history], [oss_chatbot, oss_history])
            oss_submit.click(lambda: "", None, oss_input)
            oss_clear.click(clear_state, None, [oss_chatbot, oss_history])
        with gr.TabItem("Frontier Assistant"):
            frontier_chatbot = gr.Chatbot(label="Frontier Conversation")
            frontier_input = gr.Textbox(label="Ask the hosted assistant", placeholder="Type your message...")
            frontier_history = gr.State([])
            with gr.Row():
                frontier_submit = gr.Button("Send")
                frontier_clear = gr.Button("Reset")
            frontier_submit.click(respond_frontier, [frontier_input, frontier_history], [frontier_chatbot, frontier_history])
            frontier_submit.click(lambda: "", None, frontier_input)
            frontier_clear.click(clear_state, None, [frontier_chatbot, frontier_history])
    gr.Markdown("---")
    gr.Markdown("## Notes\n- The open-source backend loads a Hugging Face model locally.\n- The frontier backend uses a hosted API when `OPENAI_API_KEY` is configured.\n- Safety patterns are applied before generation.\n- Use multiple turns to observe short-term memory.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)

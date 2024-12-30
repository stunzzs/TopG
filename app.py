import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random

# Load a pre-trained conversational AI model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load or initialize conversation history
try:
    with open("history.json", "r") as f:
        conversation_history = json.load(f)
except FileNotFoundError:
    conversation_history = {}

def save_history():
    with open("history.json", "w") as f:
        json.dump(conversation_history, f)

# Chat function
def tate_bot(input_text, user_id):
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    if user_id in conversation_history and len(conversation_history[user_id]) > 1:
    response += "\n\nAlso, I remember you told me about this earlier. Stay focused and keep hustling."

    # Tokenize input and add to conversation history
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    conversation_history[user_id].append(new_user_input_ids)

    # Generate bot response
    bot_input_ids = torch.cat(conversation_history[user_id], dim=-1)
    response_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Add motivational snippets
    motivational_quotes = [
        "STOP being weak. Take ACTION. Hustle harder or stay broke forever.",
        "You want success? Take risks. Nothing great comes from staying comfortable.",
        "ACTION fixes everything. The longer you wait, the longer you stay broke."
    ]
    response += f"\n\n{random.choice(motivational_quotes)}"

    # Save history
    save_history()
    return response

# Gradio interface
def chat(input_text, user_id):
    return tate_bot(input_text, user_id)

interface = gr.Interface(
    fn=chat,
    inputs=["text", "text"],  # User message and unique user ID
    outputs="text",
    live=True,
    description="Andrew Tate AI Mentor: Motivation, Business, and Hustling Advice"
)

# Launch the app
interface.launch()
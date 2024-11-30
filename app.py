import streamlit as st
import os
from unsloth import FastLanguageModel
from transformers import TextStreamer



working_dir = os.getcwd()

TEMPERATURE = 0.3
TOP_P = 1
MAX_TOKENS_TO_GENERATE = 1024
TOP_K = 500



st.set_page_config(page_title="🚀One-liner to story!", layout='wide')
st.markdown("<h1 style='text-align: left; color:#FFEE8C'>Generate stories for your kids here!</h1>", unsafe_allow_html=True)
#st.markdown("<h5 style='text-align: left;'>Hi! I'm LaRa, here to boost your job prospects. Let's customize your resume together to impress recruiters.</h5>", unsafe_allow_html=True)
# st.markdown("<h6 style='text-align: left;'>For questions, email Akshitha Kumbam at ak11071@nyu.edu.</h5>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

    
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your one-liner!")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "akshitha-k/oneliner-to-stories", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    
if user_input:
    
    inputs = tokenizer([user_input], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    generated_ids = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512)

    # Decode generated IDs and skip special tokens
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        st.markdown(generated_text)
        st.session_state.chat_history.append({"role": "assistant", "content": generated_text})
    










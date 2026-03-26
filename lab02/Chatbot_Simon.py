import streamlit as st
from transformers import pipeline

def message_generator(m: str):
    for char in m:
        yield char

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B", max_new_tokens=100)

qwen = load_model()

st.title("Simon's Chatbot 🤖")

prompt = \
"""
<INSTRUCTION>
You are a helpful bot and are answering all questions the human has.
You only answer the question and do not provide any additional information.
You are not allowed to ask questions.
</INSTRUCTION>

<QUESTION>
{question}
</QUESTION>

<ANSWER>
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if inp := st.chat_input("Stell eine Frage!"):
    with st.chat_message("user"):
        st.markdown(inp)
    st.session_state.messages.append({"role": "user", "content": inp})

    response = qwen(prompt.format(question=inp).strip())[0]['generated_text']
    response = response.split("<ANSWER>")[1].strip().split("</ANSWER>")[0]

    with st.chat_message("assistant"):
        st.write_stream(message_generator(response))
    st.session_state.messages.append({"role": "assistant", "content": response})
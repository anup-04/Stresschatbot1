import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Welcome! Let's Talk About Managing Stress")

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Function to get response from the API considering the conversation history
def get_response(conversation_history):
    completion = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=conversation_history,
        temperature=0.0,
        top_p=1,
        max_tokens=512,
        stream=True
    )

    # Collect the chunks of response text
    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content

    return response_text

# Initialize session state for storing chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Input box for the user's query
user_input = st.text_input("You: ", key="input", on_change=lambda: st.session_state.update({"input_value": st.session_state.input}))

# Automatically send the message when input is detected (continuous chat)
if user_input:
    # Append the user's message to the conversation history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    # Get the response from the API considering the entire conversation history
    response = get_response(st.session_state["chat_history"])

    # Append the bot's response to the conversation history
    st.session_state["chat_history"].append({"role": "assistant", "content": response})

    # Clear the input box for the next query
    st.session_state["input_value"] = ""

# Display the conversation history
st.subheader("Chat History")
for chat in st.session_state["chat_history"]:
    role = "You" if chat["role"] == "user" else "Bot"
    st.write(f"{role}: {chat['content']}")
    st.write("---")

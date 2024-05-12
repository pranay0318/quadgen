import streamlit as st
from groq import Groq
import os
import requests
from dotenv import load_dotenv
import json

# Ensure environment variables are loaded
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama3-8b-8192"
api_key = os.getenv("OPENWEATHERMAP_API_KEY")

def get_current_weather(location, unit="metric"):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units={unit}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print (f"this is the current weather\n {data['main']}")
            return json.dumps(data['main']['temp'])  # Ensure this is what you want to return
        else:
            return json.dumps({"error": f"HTTP Error: {response.status_code} - {response.reason}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Define the tools variable outside of any function, at the top level
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g., San Francisco, CA"},
                "unit": {"type": "string", "enum": ["metric", "imperial"]}
            },
            "required": ["location"]
        },
    },
}]

def complete_chat(message, messages, tools):
    messages.append({"role": "user", "content": message})
    chat_completion = client.chat.completions.create(
        model=MODEL, messages=messages, tool_choice="auto", tools=tools, max_tokens=2048
    )
    
    tool_calls = chat_completion.choices[0].message.tool_calls
    if tool_calls:
        process_tool_calls(tool_calls, messages)
    
    # Append the latest assistant response from the model
    messages.append({"role": "assistant", "content": chat_completion.choices[0].message.content})
    return chat_completion.choices[0].message.content




def process_tool_calls(tool_calls, messages):
    for tool in tool_calls:
        function_name = tool.function.name
        if function_name == "get_current_weather":
            function_params = json.loads(tool.function.arguments)
            location = function_params.get("location")
            unit = function_params.get("unit", "metric")  # Default to 'metric' if not specified
            tool_response_content = get_current_weather(location, unit)
            
            # Ensure tool_call_id is correctly passed back
            tool_response_message = {
                "role": "tool",
                "name": function_name,
                "content": tool_response_content,
                "tool_call_id": tool.id
            }
            messages.append(tool_response_message)
    return messages




# Initialize Streamlit UI components
st.title("💬 Weather Chatbot")
st.caption("🚀 A streamlit chatbot powered by Llama LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    response = complete_chat(prompt, st.session_state.messages, tools)
    st.session_state.messages.append({"role": "assistant", "content": response})
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

import streamlit as st
from groq import Groq
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = "llama3-8b-8192"
api_key = os.getenv("OPENWEATHERMAP_API_KEY")

def get_current_weather(location, unit):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units={unit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print (f"this is the current weather\n {data['main']}")
        return json.dumps(data['main']['temp'])
    else:
        return (f"Error: {response.status_code} - {response.reason}\n {response.content}")


tools = [
        {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location everytime",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string", 
                        "enum": ["metric", "imperial"]
                    },
                },
                "required": ["location"],
            },
        },   
    }
]


st.title("💬 Weather Chatbot")
st.caption("🚀 A streamlit chatbot powered by Llama LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": "You are a function calling LLM that uses a function to get the current weather (in metric) in a city in json and answers user's questions, everytime the user asks a Location, you call the API and Use the response from the API. Use only the Information recieved from the Json and do not hallucinate.",
        }, 
        {
            "role": "assistant", 
            "content": "How can I help you?"
            }
        ]

for msg in st.session_state.messages:
    if not msg["role"]=="system":
        st.chat_message(msg["role"]).write(msg["content"])

def complete_chat(message, messages, tools):

    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        tool_choice="auto",
        tools=tools,
        max_tokens=2048,
    )
    
    tool_calls = chat_completion.choices[0].message.tool_calls

    if tool_calls:
        process_tool_calls(tool_calls, messages)

    messages.append({"role": "assistant", "content": chat_completion.choices[0].message.content})

    return chat_completion.choices[0].message.content

def process_tool_calls(tool_calls, messages):
    available_functions = {
        "get_current_weather": get_current_weather     
    }
    for tool in tool_calls:
            function_name = tool.function.name
            function_to_call = available_functions[function_name]
            function_params = json.loads(tool.function.arguments)
            function_response = function_to_call(
                location = function_params.get("location"),
                unit =  "metric",
            )
            messages.append(
                {
                    "tool_call_id": tool.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            ) 

            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            messages.append(
                {
                "role": "assistant",
                "content": second_response.choices[0].message.content,
                }
            )
            return second_response.choices[0].message.content
    else:
        messages.append(
            {
                "role": "assistant",
                "content": chat_completion.choices[0].message.content,
            }
        )
        return chat_completion.choices[0].message.content
    


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    msg = complete_chat(prompt, st.session_state.messages, tools)
    st.chat_message("assistant").write(msg)
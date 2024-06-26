import streamlit as st
import requests

st.title('Atomic Habits Guru')

query_url = 'http://127.0.0.1:8000/query_program'
question = st.text_input('Enter your question:', '')

if st.button('Submit'):
    if question:
        response = requests.post(query_url, json={"question": question})
        if response.status_code == 200:
            answer = response.json()['response']
            st.write('Response:', answer)
        else:
            st.error('Failed to retrieve response')
    else:
        st.error('Please enter a question to submit.')
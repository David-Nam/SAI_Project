import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set page configuration
st.set_page_config(
    page_title="우리 SAI 챗봇",
    page_icon="💬"
)

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("우리 SAI 챗봇")
st.subheader("OpenAI 모델과 대화하세요")

# Sidebar for model selection
with st.sidebar:
    st.header("설정")
    model = st.selectbox(
        "OpenAI 모델 선택:",
        ["gpt-3.5-turbo", "gpt-4"]
    )
    
    # Display API key status
    if os.getenv("OPENAI_API_KEY"):
        st.success("API 키가 설정되었습니다!")
    else:
        st.error("API 키를 찾을 수 없습니다. .env 파일을 확인하세요.")
        st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input for user message
if prompt := st.chat_input("메시지 보내기"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response from OpenAI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Call OpenAI API with new client approach
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True
            )
            
            # Process streaming response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            full_response = f"죄송합니다, 오류가 발생했습니다: {str(e)}"
            message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.button("대화 내용 지우기"):
    st.session_state.messages = []
    st.rerun()
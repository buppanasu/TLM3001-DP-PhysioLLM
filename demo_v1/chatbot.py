import streamlit as st

def chatbot_page():
    st.header("Chat with PhysioBot!")

    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize user input in session state if it doesn't exist
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""

    # Custom CSS to style chat messages and adjust width
    st.markdown(
        """
        <style>
        .chat-message {
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 100%;
            font-size: 16px;
        }
        .user-message {
            background-color: #2B2B2B;  /* Dark grey for user messages */
            color: white;
            text-align: left;

        }
        .bot-message {
            background-color: #3A3B3C;  /* Slightly lighter grey for bot messages */
            color: white;
            text-align: left;

        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Function to handle sending message
    def send_message():
        user_message = st.session_state.user_input
        if user_message:
            # Append user input to chat history
            st.session_state.chat_history.append({'role': 'user', 'message': user_message})

            # Append bot response to chat history
            bot_response = f"Hi, human! You said: '{user_message}'."
            st.session_state.chat_history.append({'role': 'bot', 'message': bot_response})

            # Clear user input after sending
            st.session_state.user_input = ""  # Correct way to reset user input

        # Display chat history
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"<div class='chat-message user-message'><b>🧑‍💻 You:</b> {chat['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'><b>🤖 PhysioBot:</b> {chat['message']}</div>", unsafe_allow_html=True)

    # Input box for user with 'on_change' callback to send message
    st.text_input("What do you need help with?", key="user_input", placeholder="Type your message here...", on_change=send_message)


import streamlit as st
# from prompts import instructions_data
from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import Clarifai
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
import streamlit.components.v1 as components
import requests



def query(payload, model = "Security"): 
  if model == "Security":
    API_URL = "https://oa6kdk8gxzmfy79k.us-east-1.aws.endpoints.huggingface.cloud"

    headers = {
                "Accept" : "application/json", "Content-Type": "application/json" 
                }
    
    response = requests.post(API_URL, headers=headers, json=payload)

    return response.json()
  else:
      API_URL = "https://s6izgwncr4ncciig.us-east-1.aws.endpoints.huggingface.cloud"
      headers = {
      "Accept" : "application/json", "Authorization": "Bearer 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",
      "Content-Type": "application/json" }

      response = requests.post(API_URL, headers=headers, json=payload)

      return response.json()


# st.set_page_config(layout="wide")



def get_default_models():
  # if 'DEFAULT_MODELS' not in st.secrets:
  #   st.error("You need to set the default models in the secrets.")
  #   st.stop()

  # models_list = [x.strip() for x in st.secrets.DEFAULT_MODELS.split(",")]
  models_list =  ["codegen2"]
  authers = ["codegen2"]
  apps = ["something"]
  models_map = {}
  select_map = {}
  for i in range(len(models_list)):
    m = models_list[i]
    # id, rem = m.split(':')
    # author, app = rem.split(';')
    id = i
    models_map[id] = {}
    models_map[id]['author'] = authers[i]
    models_map[id]['app'] = apps[i]
    select_map[str(id)+' : '+ authers[i]] = i
  return models_map, select_map


def show_previous_chats():
  # Display previous chat messages and store them into memory
  chat_list = []
  for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
      if message["role"] == 'user':
        msg = HumanMessage(content=message["content"])
      else:
        msg = AIMessage(content=message["content"])
      chat_list.append(msg)
      if message["role"] == "user":
         st.write(message["content"])
      elif message["role"] == "assistant" and "function Greeting(props)" in message["content"]:
         st.code(message["content"])
      else:
         st.write(message["content"])

  # conversation.memory.chat_memory = ChatMessageHistory(messages=chat_list)
      




def chatbot():
  input_message = """function Greeting(props) {"""
  if message := st.chat_input(placeholder=input_message, key="input"):
    st.chat_message("user").write(message)
    st.session_state['chat_history'].append({"role": "user", "content": message})
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        # print(message)
        output = query({ "inputs": message, "config": {"max_length":1024}}, "Security")
        try:
            response = output[0]["generated_text"]
            st.code(response, line_numbers=True)
        except:
            response = "Server is starting...please try again in five minute"
            st.text(response)

        # print(output)
        # response = output[0]["generated_text"]
        # st.code(response, line_numbers=True)

        # st.write(response)
        # st.write(f":blue[{response}]")
        # st.write("This is :blue[test]")
        message = {"role": "assistant", "content": response}
        st.session_state['chat_history'].append(message)
    st.write("\n***\n")


def chat():
    prompt_list = ["React"]
    models_map, select_map = get_default_models()
    default_llm = "codegen react finetuned"
    llms_map = {'Select an LLM':None}
    llms_map.update(select_map)

    chosen_instruction_key = st.selectbox(
        'Select a prompt',
        options=prompt_list,
        index=(prompt_list.index(st.session_state['chosen_instruction_key']) if 'chosen_instruction_key' in st.session_state else 0)
    )

    # Save the chosen option into the session state
    st.session_state['chosen_instruction_key'] = chosen_instruction_key

    if st.session_state['chosen_instruction_key'] != "Select a prompt":

        with open('./styles.css') as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

        if 'chosen_llm' not in st.session_state.keys():
            chosen_llm = st.selectbox(label="Select an LLM", options=llms_map.keys())
            if chosen_llm and llms_map[chosen_llm] is not None:
                if 'chosen_llm' in st.session_state.keys():
                    st.session_state['chosen_llm'] = None
                st.session_state['chosen_llm'] = llms_map[chosen_llm]


    # Access instruction by key
    # instruction = instructions_data[st.session_state['chosen_instruction_key']]['instruction']

    template = f"""{"Write first few lines of a function and I will try to finish it for you!"} + {{chat_history}}
    Human: {{input}}
    AI Assistant:"""

    prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])

    template = f"""Write first few lines of a function and I will try to finish it for you! + {{chat_history}}
    Human: {{input}}
    AI Assistant:"""



    # Initialize the bot's first message only after LLM was chosen
    if "chosen_llm" in st.session_state.keys() and "chat_history" not in st.session_state.keys():
        with st.spinner("Chatbot is initializing..."):
            # initial_message = conversation.predict(input='', chat_history=[])
            init_user_input = """import React from 'react';

function Greeting(props) {"""
            init_assistant_response = """import React from'react';

function Greeting(props) {
  return (
    <div>
      <h1>Hello {props.name}</h1>
    </div>
  );
}

export default Greeting;
#"""
            initial_message = "Write the first few lines of a function and I will try to finish it for you!"
            welcome_message = """Hello My name is Cody, Start your code and I will finish it, 
            below is an example of how I work"""
            st.session_state['chat_history'] = [{"role": "assistant", "content": welcome_message}, {"role": "user", "content": init_user_input}, {"role":"assistant", "content":init_assistant_response}, {"role": "assistant", "content": initial_message}]
            # st.session_state['chat_history'] = [{"role": "user", "content": "hello"}]

    if "chosen_llm" in st.session_state.keys():
        show_previous_chats()
        chatbot()

    st.markdown(
        """
    <style>
    .streamlit-chat.message-container .content p {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    .output {
        white-space: pre-wrap !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

chat()

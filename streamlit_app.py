###################################################
# Import Packages
##################################################

from __future__ import annotations

from dotenv import load_dotenv

# streamlit 
import streamlit as st
from streamlit_chat import message

# os and Image
import os
import io
from PIL import Image

# Langchain
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


########################################################
# Page config
#######################################################

st.set_page_config(
    page_title="RAG : User Manual Chatbot ",
)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

openai.api_key = openai_api_key    

#########################################################
#     Main Functionality
#########################################################

st.markdown("<h1 style='color: #ffffff;'>User Manual Chatbot</h1>", unsafe_allow_html=True)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

chat = ChatOpenAI(openai_api_key=openai_api_key)               

vector_store = FAISS.load_local("manual_faiss_index", embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"score_threshold": .6, 
    "k": 5})
 
###############################################
# UDF - initialize_chat()
################################################
@st.cache_resource
def initialize_chat():
          
    prompt_template = """
        The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context.
        If the AI does not know the answer to a question, it responds by saying 'Sorry! Couldn't find that info.'
        {context}
        Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
        if not present in the document. 
        Solution:"""
    
    PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    condense_qa_template = """
        Given the following conversation and a follow up question in English language, rephrase the question and pass it as a standalone question.
        Provide the answer only in English language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    
    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    qa = ConversationalRetrievalChain.from_llm(
                llm=chat, 
                retriever=retriever, 
                condense_question_prompt=standalone_question_prompt, 
                return_source_documents=True, 
                combine_docs_chain_kwargs={"prompt":PROMPT})
    return qa
        
####################################################################################
    # UDF ends
####################################################################################

qa_chat =  initialize_chat()


# session variables    
if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if 'answer' not in st.session_state:
    st.session_state['answer'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Chat UI
qa_chat_container = st.container()

with qa_chat_container:
    message("Hey, Please ask your questions related to user manual here.", 
            avatar_style="bottts",
            seed="b6e3f4")

qa_input_container = st.container()
with qa_input_container:
    with st.form("chat_qa_input", clear_on_submit=True):
        col1,col2 = st.columns([12,2])
        input = col1.text_area('Question',
                                placeholder = "Enter your question", 
                                key="chat_qa_input",
                                label_visibility="collapsed")
            
        submitted = col2.form_submit_button("Submit")

        

        if submitted:
            st.session_state.questions.append(input)
            chat_history = st.session_state["chat_history"]
            if len(chat_history) == 3:
                chat_history = chat_history[:-1]

                #st.write(chat_history)  
            #chat_history =[]
            result = qa_chat({"question": input, "chat_history": chat_history})
            answer = result['answer']
                #st.write(answer)

            if answer == "Sorry! Couldn't find that info.":
                result["source_documents"] = []
                #st.write(result)
            st.session_state.answer.append(answer)
            chat_history.append((input, answer))


if st.session_state['questions']:
    with qa_chat_container:
        for i in range(0,len(st.session_state['questions'])):
                
            message(st.session_state['questions'][i], 
                    is_user=True, 
                    key=str(i) + '_user', 
                    avatar_style="thumbs",
                    seed=0)
                
            message(st.session_state["answer"][i], 
                    key=str(i),
                    avatar_style = "bottts", 
                    seed="b6e3f4")

            ###############################
            # source documents
            ##############################
        
           
        if 'source_documents' in result:
            
            with st.expander("Source documents"):
                i = 0
                for d in result['source_documents']:
                    i = i+1
                       
                    st.write("Document ", str(i),": ",d.metadata['source'], "; page:",str(d.metadata['page']))
                    st.write('Supporting content:')
                    st.caption(d.page_content)
                    st.divider()
            
         
    



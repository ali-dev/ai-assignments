import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAI
import os
import cmd
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"




def search_pinecone(index_name):
    """this functions hansles the search in the pinecone index"""
    # we could pass the model as a paramter but no need for now
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    document_vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = document_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.7,  "k":10})
    messages = st.session_state.messages
    context = retriever.get_relevant_documents(",".join(str(element) for element in messages))
    content = ''
    for doc in context:
        content+=doc.page_content+'\n'
        content+=doc.metadata['source']+'\n'
    return content

# Tools to be used by the agent
@tool
def get_meal_recommendation():
    """This tool shall be called by the agent to get meal recommendations based on the conversation."""
    print("Meal recommender is invoked")
    response = search_pinecone("cookunity-meals")        
    return response

@tool
def get_help_desk():
    """This tool shall be invoked when the customer asks a general questions about cookunity and how to use the service. This includes general questions about how the subscription works, canceling account, skipping order, rescheduling , account settings, packaging sustainability, """
    print("Help desk is invoked")
    response = search_pinecone("cookunity-help-center")        
    return response


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a Cookunity assistant and your name is Zest.
                You are the chatbot of the CookUnity web site and your mission is to answer the customer's inquiries. 
                These inquiries include general questions about how the subscription works, canceling account, skipping order, rescheduling , account settings, packaging sustainability, etc. 
                The user may also ask for meal recommendations based on their preferences.
                The menu has the following diets categories: Paleo, Vegetarian, Keto, Vegan, Low Carbs, Low Calories, Pescatarian, Mediterranean.
                The menu has the following cuisines categories: American, Italian, Mexican, Indian, Mediterranean, Asian, Latin American, African, Caribbean, French, European.
                You are having a conversation with a CookUnity user, be gentle and friendly
                there are 2 tools to choose from to help you answer the user's questions: get_meal_recommendation, and get_help_desk.
                In the future, we will add more tools like delkivery_status_tool, billing_tool, etc..
                You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!
                
                """
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

st.title("Cookunity Bot")

with st.chat_message("assistant"):
    st.write("Hello, This chatbot can help with general questions about Cookunity and can recommend meals")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []




prompt = st.chat_input("Ask me anything about Cookunity or ask for meal recommendations")

if prompt:
    st.chat_message("user").markdown(f"{prompt}")    
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = f"Echo: {prompt}"

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    tools = [get_meal_recommendation, get_help_desk]
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt_template, verbose=False)

    result = agent_executor.invoke({"messages": st.session_state.messages})
    with st.chat_message("assistant"):
        st.write(f"{result['output']}")

    st.session_state.messages.append({"role": "ai", "content": result['output']})   



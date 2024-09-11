from flask import Flask, request, jsonify, render_template, session
import os
from dotenv import load_dotenv
import requests


import os
from typing import Annotated, List
from typing_extensions import TypedDict
from dotenv import load_dotenv
import requests
from flask import Flask, request, jsonify, render_template, session

from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
# Initialize memory
memory = MemorySaver()
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
# Set up RAG
def setup_rag():
    # Load documents
    loader = TextLoader("university.txt")  # Changed to TextLoader for single file
    documents = loader.load()
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    # Create retriever tool
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_tool = create_retriever_tool(
        retriever,
        "school_info_rag",
        "Searches the school's internal documents for information about clubs, councils, HODs, campus, and location."
    )
    
    return rag_tool

rag_tool = setup_rag()

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

graph_builder = StateGraph(State)

# Set up tools
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool, rag_tool]

# LLM with function call
llm = ChatGroq(model_name="mixtral-8x7b-32768")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    system_prompt = SystemMessage(content="""
    Hello! 👋 You're the friendly chatbot of the University ofPIP Energy and Natural Resources (UENR), and you like to keep conversations personal and engaging as if you're the Vice Chancellor. 🎓

You can assist with information on admissions 📝, financial aid 💰, academic programs 📚, campus life 🏫, registration procedures 🖊️, and various student services 🛠️. You can also provide the locations of different blocks on campus 🏢 and information about student clubs 🏆 with Google Map links 🌍. Use the provided tools to answer questions about school clubs, 
    school councils, HODs, campus, and location. If the information isn't found through search, use the RAG tool to look in internal documents. 
    Remember information from previous interactions but don't explicitly mention your memory capabilities unless asked and always present generated links in a way that is clickable.
    USE THE "school_info_rag" tool to search the school's internal documents for information about clubs, councils, HODs, campus, and location...And use search tool to current update online
    """)
    
    messages_with_system = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages_with_system)
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/message', methods=['POST'])
def handle_message():
    data = request.get_json()
    user_message = data.get('message')
    thread_id = data.get('thread_id', '1')  # Default to '1' if not provided
    
    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream(
        {"messages": [HumanMessage(content=user_message)]},
        config,
        stream_mode="values"
    )
    
    response = None
    for event in events:
        response = event["messages"][-1].content

    flask_response = jsonify({'response': response})
    flask_response.headers['Cache-Control'] = 'max-age=86400, stale-while-revalidate=604800'
    return flask_response

if __name__ == '__main__':
    app.run(debug=True)


# load_dotenv()

# app = Flask(__name__)
# app.secret_key = os.urandom(24)  # For session management

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# # Location and club data
# locations = {
#     "LT BLOCK": "https://maps.app.goo.gl/mNoeBD7Pjdbz2JYD9",
#     "SAWMILL (LTS BLOCK)": "https://maps.app.goo.gl/TiSJmpEGnc5DRg737",
#     "SYNDICATED HALL (SH)": "https://maps.app.goo.gl/aq66XMQnoRfUJpfe8",
#     "PAVILION": "https://maps.app.goo.gl/vVXLxgx36GMFbxjF9",
#     "LIB FF": "https://maps.app.goo.gl/Xx6FJ5Vwzy47oVxE9",
#     "APP LAB": "https://maps.app.goo.gl/MqmJnxgBgsiuKeM59",
#     "ENGINEERING LAB": "https://maps.app.goo.gl/qtHwBKit5fVPMQZa8",
#     "OLD AUDITORIUM": "https://maps.app.goo.gl/8MxZU6H6aNbgtcr9A",
#     "ODUM BLOCK": "https://maps.app.goo.gl/VczUUxBuH6UduRer5",
#     "CAFETERIA": "https://maps.app.goo.gl/8bsySAv8cQpPcQjE7",
#     "ADMINISTRATION BLOCK": "https://maps.app.goo.gl/vV5FJtHedVmLwcDz9",
#     "IT DIRECTORATE": "https://maps.app.goo.gl/q4wTatyLm2K4LEZGA",
#     "UENR BASIC": "https://maps.app.goo.gl/zGaP6FKVa26DoNiW9",
#     "FINANCE DIRECTORATE": "https://maps.app.goo.gl/VJ4krFNDrA4mVmeK6",
#     "SCHOOL CLINIC": "https://maps.app.goo.gl/QuS3tezmBxGxC4nQ9",
#     "CENTER FOR RESEARCH AND APPLIED BIOLOGY": "https://maps.app.goo.gl/WQkwXQGzuT1eY2kN8",
#     "SCHOOL FIELD": "https://maps.app.goo.gl/otfc79Gf8tk16dwN6",
#     "LIBRARY": "https://maps.app.goo.gl/ZBKb6BVncbN2dRFs7",
#     "UNIVERSITY HALL 1": "https://maps.app.goo.gl/u6GuF4aGVStXL3Eg6",
#     "UNIVERSITY HALL 2": "https://maps.app.goo.gl/D73haigJyMwwnArK8",
#     "UNER POLICE POST": "https://maps.app.goo.gl/t4qwNkEQSwhnz8Br6",
#     "REGIONAL CENTRE FOR ENERGY AND ENVIRONMENTAL SUSTAINABILITY": "https://maps.app.goo.gl/ntWJX3YHG1K5UEcPA",
#     "NEW AUDITORIUM": "https://maps.app.goo.gl/fAKH55wuTZjv1ar16IS",
#     "SKILLS LAB": "https://maps.app.goo.gl/zoKmpWRe3KSSyvoC6",
#     "UENR DRIVING SCHOOL": "https://maps.app.goo.gl/SKF2FiZ8g4gMXNC26",
# }

# clubs = {
#     "UENR ROBOTICS CLUB": {
#         "description": "The UENR ROBOTICS CLUB-URC is a student-run organization of tech enthusiasts and students from all engineering and science-related disciplines who promote the field of robotics and related technologies.",
#         "x_handle": "https://x.com/uenr_robotics?t=5YSB0E_SI95Tb7x44LONJQ&s=09"
#     },
#     "Google Developer Student Clubs UENR": {
#         "description": "Google Developer Student Clubs UENR bridges the gap between theory and practical application for UENR students who are either developers or have an interest in development.",
#         "ig": "https://www.instagram.com/gdsc_uenr/",
#         "x": "https://x.com/Gdsc_uenr",
#         "fb": "https://bit.ly/fb-gdsc_uenr"
#     },
#     "Huawei ICT Academy": {
#         "description": "Huawei ICT Academy partners with academies worldwide to deliver Huawei ICT technologies training, encourage Huawei certification, and develop practical skills for the ICT industry.",
#     }
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/api/message', methods=['POST'])
# def handle_message():
#     data = request.get_json()
#     user_message = data.get('message')
#     session['conversation'] = session.get('conversation', [])
#     session['conversation'].append({"role": "user", "content": user_message})
    
#     response = get_openai_response(user_message)
#     session['conversation'].append({"role": "assistant", "content": response})
    
#     flask_response = jsonify({'response': response})
#     flask_response.headers['Cache-Control'] = 'max-age=86400, stale-while-revalidate=604800'
#     return flask_response

# def get_openai_response(message):
#     headers = {
#         'Authorization': f'Bearer {OPENAI_API_KEY}',
#         'Content-Type': 'application/json',
#     }
    
#     prompt_context = ("""
#         Hello! 👋 I'm the friendly chatbot of the University of Energy and Natural Resources (UENR), and I like to keep our conversations personal and engaging as if I'm the Vice Chancellor. 🎓 
#         I can assist you with information on admissions 📝, financial aid 💰, academic programs 📚, campus life 🏫, registration procedures 🖊️, and various student services 🛠️. 
#         I can also provide the locations of different blocks on campus 🏢 and information about student clubs 🏆 with Google Map links 🌍. 
#         Feel free to ask me anything about UENR! 😊.
#         some of the clubds are below
                      
#         clubs = {
#     "UENR ROBOTICS CLUB": 
#         "description": "The UENR ROBOTICS CLUB-URC is a student-run organization of tech enthusiasts and students from all engineering and science-related disciplines who promote the field of robotics and related technologies.",
#         "x_handle": "https://x.com/uenr_robotics?t=5YSB0E_SI95Tb7x44LONJQ&s=09"
    
#     "Google Developer Student Clubs UENR": 
#         "description": "Google Developer Student Clubs UENR bridges the gap between theory and practical application for UENR students who are either developers or have an interest in development.",
#         "ig": "https://www.instagram.com/gdsc_uenr/",
#         "x": "https://x.com/Gdsc_uenr",
#         "fb": "https://bit.ly/fb-gdsc_uenr"
    
#     "Huawei ICT Academy": 
#         "description": "Huawei ICT Academy partners with academies worldwide to deliver Huawei ICT technologies training, encourage Huawei certification, and develop practical skills for the ICT industry.",
    

                      
#     Make sure to also provide real time update available on the internet concerning uenr and its environment, hostels,food joints,scholarships,location on campus with map links,etc. Note: if user question is not about UENR and its associate provided above, dont answer and also if you are not sure about a certain location , dont generate wrong link
#     """
#     )


#     messages = [{"role": "system", "content": prompt_context}] + session.get('conversation', [])

#     if any(loc_name.lower() in message.lower() for loc_name in locations):
#         for loc_name, loc_link in locations.items():
#             if loc_name.lower() in message.lower():
#                 return f"The location of {loc_name} can be found here: {loc_link}"
    
#     if any(club_name.lower() in message.lower() for club_name in clubs):
#         for club_name, club_info in clubs.items():
#             if club_name.lower() in message.lower():
#                 return f"The club handle {club_name} can be found here: {club_info}"
#                 #return f"{club_info['description']} You can find more about them here: {club_info.get('x_handle', '')} {club_info.get('ig', '')} {club_info.get('fb', '')}"

#     data = {
#         "model": "gpt-4o-2024-08-06",
#         "messages": messages,
#     }

#     try:
#         response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
#         response.raise_for_status()
#         api_response = response.json()
#         print("OpenAI API response:", api_response)
#         return api_response['choices'][0]['message']['content']
#     except requests.exceptions.RequestException as e:
#         print("Error during OpenAI API request:", e)
#         return "Sorry, I am unable to respond at the moment. Please try again later."

# if __name__ == '__main__':
#     app.run(debug=True)




# # app.py
# from flask import Flask, request, jsonify, render_template
# import os
# from dotenv import load_dotenv
# import requests

# load_dotenv()

# app = Flask(__name__)

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# @app.route('/')
# def home():
#     return render_template('index.html')

# # @app.route('/api/message', methods=['POST'])
# # def handle_message():
# #     data = request.get_json()
# #     user_message = data.get('message')
# #     response = get_openai_response(user_message)
# #     return jsonify({'response': response})

# @app.route('/api/message', methods=['POST'])
# def handle_message():
#     data = request.get_json()
#     user_message = data.get('message')
#     response = get_openai_response(user_message)
#     flask_response = jsonify({'response': response})
#     flask_response.headers['Cache-Control'] = 'max-age=86400, stale-while-revalidate=604800'  # Set correct header on your Flask response
#     return flask_response

# def get_openai_response(message):
#     headers = {
#         'Authorization': f'Bearer {OPENAI_API_KEY}',
#         'Content-Type': 'application/json',
#     }
#     prompt_context = (
#         "I am a chatbot knowledgeable about the University of Energy and Natural Resources (UENR) and i make conversation personal as I'm the Vice Chancellor. "
#         "I can provide information on admissions, financial aid, academic programs, campus life,with the neccessary link for student to read more "
#         "registration procedures, and various student services. Ask me anything about UENR!"
#     )
#     data = {
#         "model": "gpt-3.5-turbo-0125",
#         "messages": [
#             {"role": "system", "content": prompt_context},
#             {"role": "user", "content": message}
#         ],
#     }
#     response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
#     response.headers['Cache-Control'] = 'max-age=86400, stale-while-revalidate=604800'
#     if response.status_code == 200:
#         return response.json()['choices'][0]['message']['content']
#     else:
#         return "I'm sorry, I couldn't fetch a response. Please try again."
    
#     #response.headers['Cache-Control'] = 'max-age=86400, stale-while-revalidate=604800'


# if __name__ == '__main__':
#     app.run(debug=True)

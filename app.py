import streamlit as st
import os
import openai
from llama_index import ServiceContext, StorageContext, VectorStoreIndex, SummaryIndex, load_index_from_storage
from llama_index.llms import OpenAI

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine

import nest_asyncio
nest_asyncio.apply()

#load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"] #os.environ['OPENAI_API_KEY'] 

#define LLM
llm = OpenAI(temperature=0.2, model_name="gpt-4")


service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

# # For clipboard functionality
# @st.cache(allow_output_mutation=True)
# def get_clipboard_func():
#     return components.declare_component("clipboard", url="http://localhost:3001")

# # Load LlamaIndex
# @st.cache(allow_output_mutation=True)
# def load_index():
#     return LlamaIndex()

# # Define Streamlit App
# def main():

# Title
st.title("Med Research App")

def get_subdirectories(path):
    """Return a list of directory names (not full paths) just below the directory 'path'"""
    # List all entities in the directory pointed by path
    all_entities = os.listdir(path)
    
    # Filter only directories and exclude files
    return [name for name in all_entities if os.path.isdir(os.path.join(path, name))]

parent_folder = "./storage"

# Sidebar
with st.sidebar:
    st.header("Select Papers")
    # Assuming there's a folder named 'data' with papers
    all_files = get_subdirectories(parent_folder)
    selected_files = st.multiselect("Select files", all_files)

    load_button = st.button("Load papers")
    clear_button = st.button("Clear selection")

# Manage LlamaIndex state
session_state = st.session_state

# Initializing session state for the first time
if 'loaded_files' not in session_state:
    session_state.loaded_files = []

if 'query_response' not in session_state:
    session_state.query_response = ""

# if 'index' not in session_state:
#     session_state.index = load_index()

if load_button:
    # Assuming each file in 'data' directory can be loaded to LlamaIndex
    #or file in selected_files:
        #session_state.index.load(os.path.join('data', file))
    session_state.loaded_files = selected_files

if clear_button:
    session_state.loaded_files = []
    # Assuming there's a reset method in LlamaIndex to clear the data
    #session_state.index.reset()

def query_engine_from_papers(file_list, num_sources=4):
    # can update sources to return for top k later
    #dynamicly load selected indexes and create query engine tools
    index_set = {}
    for paper in file_list:
        st.write('loading:', paper)
        storage_context = StorageContext.from_defaults(persist_dir=f"./storage/{paper}")
        cur_index = load_index_from_storage(
            storage_context, service_context=service_context
        )
        index_set[paper] = cur_index
    #create query engine tools from indexes
    individual_query_engine_tools = [
        QueryEngineTool(
            query_engine=index_set[paper].as_query_engine(similarity_top_k= num_sources),
            metadata=ToolMetadata(
                name=f"vector_index_{paper}",
                description=f"Good for answering questions related to {paper}",
            ),
        )
        for paper in file_list
        ]
    #create subquestion q engine
    s_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        service_context=service_context,
        use_async=True,
        )
    return s_engine
    
# Main App Content
st.write("Select papers on the left to analyze, and select 'Load papers'. After your papers are loaded, enter your query and select 'Run query'")

query = st.text_input("Enter your query")
run_query_button = st.button("Run query")

if run_query_button:
    st.write('running query')
    # Assuming the LlamaIndex has a 'query' method
    engine = query_engine_from_papers(session_state.loaded_files)
    response = engine.query(query)
    st.write(str(response))
    st.write("***Sources used to generate response: ****")
    with st.expander(label="sources"):
        for i in range(len(response.source_nodes)):
            st.write(response.source_nodes[i].get_content(metadata_mode="all"))


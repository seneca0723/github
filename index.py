import os
import openai

from llmsherpa.readers import LayoutPDFReader
from llama_index.readers.schema.base import Document
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.tools import ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.llms import OpenAI
from dotenv import load_dotenv


import nest_asyncio
nest_asyncio.apply()

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY'] 
#define LLM
llm = OpenAI(temperature=0.1, model_name="gpt-4")

# pdf_url = "/Users/chandlermccann/projects/med_research_app/med_app/grand_rounds_articles/vanDijk_2016_Return_to_sports_and_clinical_outcomes_in_patients_treated_for_peroneal_tendon_dislocation_a_systematic_review.pdf"
# pdf_reader = LayoutPDFReader(llmsherpa_api_url)
# doc = pdf_reader.read_pdf(pdf_url)



# #define LLM
# llm = OpenAI(temperature=0.2, model_name="gpt-4")


def get_all_filenames(directory_path):
    """Return a list of filenames in the given directory."""
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

#get filenames
folder_path =  "/Users/chandlermccann/projects/med_research_app/github/articles/grand_rounds_articles"
titles = get_all_filenames(folder_path)
print(titles)

#read each file and store in a documents dict
# documents = {}
# for title in titles:
#     documents[title] = SimpleDirectoryReader(input_files=[f"{folder_path}/{title}"]).load_data()
# print(f"loaded documents with {len(documents)} documents")

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)
documents = {}
simple_docs = []
sherpa_docs = []
for title in titles:
    try:
        pdf_path = f"{folder_path}/{title}"  
        print(pdf_path)
        documents[title] = pdf_reader.read_pdf(pdf_path)
        sherpa_docs.append(title)
    except:
        documents[title] = SimpleDirectoryReader(input_files=[f"{folder_path}/{title}"]).load_data()
        simple_docs.append(title)
print(f"loaded documents with {len(documents)} documents of {len(titles)} articles")




#prepare index dict for all documents
index_set = {}
service_context = ServiceContext.from_defaults(llm=llm)#, chunk_size=512)

#index each doc and save to disk
for title in titles:
    storage_context = StorageContext.from_defaults()
    print(f"creating embeddings for {title}")
    if title in sherpa_docs:
        cur_index_vec = VectorStoreIndex([],
            service_context=service_context,
            storage_context=storage_context,
        )
        #index = VectorStoreIndex([])
        cur_doc = documents[title]
        for chunk in cur_doc.chunks():
            cur_index_vec.insert(Document(text=chunk.to_context_text(), extra_info={}))
    else:
        cur_index_vec = VectorStoreIndex.from_documents(
            documents[title],
            service_context=service_context,
            storage_context=storage_context,
        )
    index_set[title] = cur_index_vec
    storage_context.persist(persist_dir=f"/Users/chandlermccann/projects/med_research_app/github/storage/{title}")
    print(f"index complete")
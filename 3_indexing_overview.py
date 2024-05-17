# Below is a real world example of indexing and retrieving documents using the Langchain library.
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '<your_langchain_api_key>'
os.environ['OPENAI_API_KEY'] = '<your_openai_api_key>'

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#### INDEXING ####

# Here we load our document from the web simply by providing the URL to a webpage.
# The SoupStrainer here filters the html doc on the class markdown-body and entry-content. 
# This way, we can filter out a lot of the noise from the webpage for our vectorstore
loader = WebBaseLoader(
    web_paths=("https://github.com/ReeceGibbsGit/RAG_Breakdown/blob/master/Example%20Docs/example.md",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer("p")
    ),
)

docs = loader.load()

# We then split the document into smaller chunks to be embedded.
# We do this because the OpenAI Embedding model has a limit on the number of tokens it can process at once.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(docs)

# We then pass the splits, a list of documents, to the chroma vectorstore.
# The vectorestore is a data structure that allows us to store and retrieve embeddings with a reference to the original document.
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

# We can then fetch a retriever from the vectorstore for later use.
retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Here we pull a template prompt from LangChain hub. It looks like this:
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:
prompt = hub.pull("rlm/rag-prompt")

# We define the LLM we want to use
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# The function below is called after the retriever has been invoked and has identified the relevant documents within the vector to query.
# We can see this by printing the relevant documents and comparing them to the original document.
# Look how it has cut out the fluff and managed to keep the relevant information.
def format_docs(docs):
    relevant_doc_retrieval = "\n\n".join(doc.page_content for doc in docs)
    # print(relevant_doc_retrieval)

    return relevant_doc_retrieval

# Here we defined the chain of operations we want to perform on the input.
# The RunnablePassThrough function is a placeholder for the question we ask later. We can fill this gap by leveraging the invoke method on the chain.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
print(rag_chain.invoke("Who smiled warmly in the story?"))
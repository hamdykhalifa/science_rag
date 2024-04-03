"""class for augmenting retrieval with context from papers"""
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

class CustomRAG:
    def __init__(self):
        # Initialize all the necessary components here
        
        # Embedding model
        self.embedding_model = AzureOpenAIEmbeddings(azure_deployment="text-embedding")
        
        # Language model
        self.llm = AzureChatOpenAI(azure_deployment="gpt-35-turbo-1106", temperature=0)
        
        # Vector store
        self.vectorstore = Chroma(persist_directory="db/recursive_splits", embedding_function=self.embedding_model)
        
        # Retriever
        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20})
        
        # Prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a helpful math research assistant. 
                Always mention the sections or pages you based your answer on."""),
                ("human", """Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer or the answer is not in the context, just say that you don't know.
                Question: {question}
                Context: {context}
                Answer:""")
            ]
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, question):
        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

"""class for augmenting retrieval with context from papers"""
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
        
        # Contextualization Prompt
        self.contextualize_q_system_prompt = """Given a chat history and the latest user question
        which might reference context in the chat history, formulate a standalone question
        which can be understood without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is."""
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", self.contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # QA System Prompt
        self.qa_system_prompt = """You are a helpful math research assistant. Always mention the sections or pages you base your answer on.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.

        {context}"""
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # History-Aware Retriever
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt)
        
        # Question-Answer Chain
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        
        # Retrieval and QA Chain
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain)

    def invoke(self, question, chat_history):
        input_dict = {
            "input": question,
            "chat_history": chat_history
        }
        # Pass the structured input dictionary to the invoke method
        response = self.rag_chain.invoke(input_dict)
        return response

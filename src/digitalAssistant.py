import os
from groq import Groq
from dotenv import load_dotenv
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from .utils.mongo_client import MongoDB
from typing import List, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import json
import asyncio
# from utils.helper import log_time
from operator import add
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL")


def fallback_function(value, new_value):
    return new_value

class State(TypedDict):
    question: str
    rephrased_question: Annotated[str, fallback_function]
    relevance: str
    chat_history: List
    rag_docs: List = []
    session_id: str
    model_response: str
    user_question_from_chat_history: List
    ai_answer_from_chat_history: List  

class DigitalAssistant():
    def __init__(self):
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.QDRANT_HOST = os.getenv("QDRANT_HOST")
        self.QDRANT_PORT = os.getenv("QDRANT_PORT")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
        self.client = QdrantClient(url = self.QDRANT_URL, api_key = self.QDRANT_API_KEY)
        self.vectorstore = QdrantVectorStore(client=self.client, collection_name=self.QDRANT_COLLECTION_NAME, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k": 5})
        self.mongo = MongoDB()
        self.llm = Groq(api_key=self.GROQ_API_KEY)
        self.model = MODEL

        self.check_for_irrelevancy_template = """ You are a query classifier who is going to classify the given query to either chitchat or if it is relevant to Kiran, his career, work, educational background. If the query is related to Kiran, then you are supposed to reply with a True, if it is not, then you are supposed to reply with a False. You are supposed to evaluate the chat history and the current user question and then make the decision of either replying with a True or False. Stick to saying only True or False, you dont have to provide any explanation. All sorts of greetings or other topics should be classified as chithcat. Only questions specific to Kiran, his career or his personal work and habits should be classified as True. Here are a few examples:
        Question: How are you doing today?
        Classification: Fale
        Question: What is Kiran's favorite food?
        Classification: True
        Question: What is the weather like today?
        Classification: False
        Question: Tell me about Kiran's work
        Classification: True
        Question: Tell me about his educational background
        Classification: True
        Question: What are some programming languages Kiran knows?
        Classification: True
        Question: What is the time now?
        Classification: False
        Question: What is the capital of India?
        Classification: False
        Question: What is his tech stack?
        Classification: True
        Question: What are some projects he worked on?
        Classification: True
        Question: Any blogs or articles he has written?
        Classification: True
        Question: What is the best way to reach out to him?
        Classification: True
        Question: give me his contact details
        Classification: True
        Question: What is his emailId?
        Classification: True
        Question: Who is his girlfriend?
        Classification: True
        """

        self.grounding_template = """You are Ash, a digital assistant of Kiran. All the questions which are not related to the Kiran or his work, or anything about him are routed to you. You are responsible for answering these questions. You decline to answer any questions which are not related to Kiran. If it is a greeting, you just greet the user and ask the user about he is feeling today and have some chit-chat conversation. But ground yourself only to talking about Kiran only and encourage the user to ask more questions related to him. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
           """

        self.rephrase_question_template = """Combine the chat history and follow up question into a standalone question.
            Provide only the follow up question and no premable or explanation. If you dont see the relation in chat history to the current question, just reply the current question as it is and do not modify the question or do any changes to the current question. You are supposed to only respond with the question and not any sort of explanation.Give a lot of weightage to the current question asked by the user. But do not alter or add any context other than what is needed to make the question standalone."""

        self.rag_template = """You are "Ash", Kiran's digital assistant. You are supposed to answer the user question based on the context given to you. You are supposed to evaluate the chat history and the current user question and reply to the user. Be very polite and keep the answers short and sweet and a little humorous and smart. If you dont know the answer, just say that you dont know. Use three sentences maximum, be polite and humble and complete your answers."""

        self.workflow = StateGraph(State)
        
        self._setup_workflow()

    def _setup_workflow(self):
        #add more nodes here
        self.workflow.add_node("retrieve_chat_history", self.retrieve_chat_history)
        self.workflow.add_node("check_for_irrelevancy", self.check_for_irrelevancy)
        self.workflow.add_node("rephrase_question", self.rephrase_question)
        self.workflow.add_node("grounding_model", self.grounding_model)
        self.workflow.add_node("retrieve_vector_store", self.retrieve_vector_store)
        self.workflow.add_node("aggregate_answer", self.aggregate_answer)
        self.workflow.add_node("save_chat_history", self.save_chat_history)

        #add edges
        self.workflow.add_edge(START, "retrieve_chat_history")
        self.workflow.add_edge("retrieve_chat_history", "check_for_irrelevancy")
        self.workflow.add_conditional_edges("check_for_irrelevancy",
                                           lambda state: "rephrase_question" if state["relevance"] == True else "grounding_model",
                                        {
                                            "rephrase_question": "rephrase_question",
                                            "grounding_model": "grounding_model"
                                        })
        self.workflow.add_edge("rephrase_question", "retrieve_vector_store")
        self.workflow.add_edge("retrieve_vector_store", "aggregate_answer")
        self.workflow.add_edge("aggregate_answer", "save_chat_history")
        self.workflow.add_edge("grounding_model", "save_chat_history")
        self.workflow.add_edge("save_chat_history", END)
        self.graph = self.workflow.compile()


    def retrieve_chat_history(self, state):
        print("---RETRIEVE CHAT HISTORY NODE---")
        session_id = state["session_id"]
        chat_history = self.mongo.read_documents({"session_id": session_id}, limit=6, sort_order=-1, sort_field="createdAt")
        reversed_chat_history = chat_history[::-1]
        user_question_from_chat_history = []
        ai_answer_from_chat_history = []
        chat_history_reformatted = []
        # print count of messages if not show empty
        if len(chat_history) == 0:
            print("No chat history found")
        else:
            print(f"Found {len(chat_history)} messages")
            for msg in reversed_chat_history:
                if msg["type"] == "AI":
                    ai_answer_from_chat_history.append(msg["content"])
                    chat_history_reformatted.append(msg["content"])
                else:
                    user_question_from_chat_history.append(msg["content"])
                    chat_history_reformatted.append(msg["content"])

        return {"chat_history": chat_history_reformatted, "user_question_from_chat_history": user_question_from_chat_history, "ai_answer_from_chat_history": ai_answer_from_chat_history}

    def check_for_irrelevancy(self, state):
        print("---CHECK FOR IRRELEVANCY NODE---")
        user_input = state["question"]
        chat_history = state["chat_history"]
        response = self.llm.chat.completions.create(
            messages = [
                {
                    "role": "user",
                    "content": self.check_for_irrelevancy_template + f" This is the chat history: {chat_history}. This is the user question: {user_input}",
                }
                        ],
            model=self.model,)
        answer = response.choices[0].message.content
        print("response from check_for_irrelevancy node", answer)
        if "True" in answer or "true" in answer:
            return {"relevance": True}
        else:
            return {"relevance": False}
    
    def rephrase_question(self, state):
        print("---REPHRASE QUESTION NODE---")
        user_question_from_chat_history = state["user_question_from_chat_history"]
        user_input = state["question"]
        # chat_history = state["chat_history"]
        response = self.llm.chat.completions.create(
            messages = [
                {
                    "role": "user",
                    "content": self.rephrase_question_template + f" These are the previous user questions : {user_question_from_chat_history}. This is the current user question: {user_input}",
                }
                        ],
            model=self.model,)
        answer = response.choices[0].message.content
        print("response from rephrase_question node :: ", answer)
        print("--------------------------------------")
        return {"rephrased_question": answer}
    
    def grounding_model(self, state):
        print("---GROUNDING MODEL NODE---")
        user_input = state["question"]
        chat_history = state["chat_history"]
        response = self.llm.chat.completions.create(
            messages = [
                {
                    "role": "user",
                    "content": self.grounding_template + f" These are the previous conversations : {chat_history}. This is the current user question: {user_input}",
                }
                        ],
            model=self.model,)
        answer = response.choices[0].message.content
        print("response from grounding_model node :: ", answer)
        return {"model_response": answer, "rag_docs": []}
    
    def retrieve_vector_store(self, state):
        print("---RETRIEVE VECTOR STORE NODE---")
        rephrased_question = state["rephrased_question"]
        # documents = self.retriever.invoke(question)
        documents = self.retriever.get_relevant_documents(rephrased_question)
        print("these are the retrieved docs", documents)
        print("--------------------------------------")
        return {"rag_docs": documents}
    
    def aggregate_answer(self, state):
        print("---AGGREGATE ANSWER NODE---")
        chat_history = state["chat_history"]
        user_input = state["rephrased_question"]
        rag_docs = state["rag_docs"]
        response = self.llm.chat.completions.create(
        messages = [
                {
                    "role": "user",
                    "content": self.rag_template + f" This is the context you have access to. Context: {str(rag_docs)}.  These are the previous conversations : {chat_history}. This is the current user question: {user_input}",
                }
                        ],
            model=self.model,)
        answer = response.choices[0].message.content
        print("response from aggregate_answer node :: ", answer)
        return {"model_response": answer}
    
    def save_chat_history(self, state):
        print("---SAVE CHAT HISTORY NODE---")
        session_id = state["session_id"]
        question = state["question"]
        model_response = state["model_response"]
        rag_docs = state["rag_docs"]

        self.mongo.insert_document(session_id=session_id, doc_type="USER", content_text = question)
        self.mongo.insert_document(session_id=session_id, doc_type="AI", content_text = {"model_response":model_response, "rag_docs": str(rag_docs)})
        return state
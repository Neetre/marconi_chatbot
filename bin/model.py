from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import asyncio


class Model:
    def __init__(self, groq_api_key: str):
        self.chat = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
        
    def chat(self, msg):
        if not msg:
            raise "Please provide a message to chat with the bot."
        
        system = "You are a helpful assistant in a Discord server. Be concise, friendly, and engaging."
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        chain = prompt | self.chat
        response = asyncio.to_thread(chain.invoke, {"text": msg})
        text_response = response.content
        
        return text_response


def main():
    

if __name__ == "__main__":
    main()
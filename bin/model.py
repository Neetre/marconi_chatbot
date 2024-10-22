from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class Model:
    def __init__(self, groq_api_key: str, language: str = "english", school_info: str = "", additional_instructions: str = ""):
        self.llm = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
        self.language = language
        self.school_info = school_info
        self.additional_instructions = additional_instructions
        
    async def chat(self, msg):
        if not msg:
            raise ValueError("Please provide a message to chat with the bot.")
        
        system_content = f"""You are a helpful assistant on a school community home page. Respond in {self.language}. 
Be concise, friendly, and engaging.

School Information:
{self.school_info}

Additional Instructions:
{self.additional_instructions}

Always maintain a positive and supportive tone, and prioritize the well-being and education of students."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=msg)
        ]

        response = await self.llm.ainvoke(messages)
        return response.content


async def inference():
    school_info = """
    Name: ITI G. Marconi
    Location: Verona, Italy
    Motto: "Learning Today, Leading Tomorrow"
    Grades: 1-5 superiori
    """
    '''
    additional_instructions = """
    - Provide information about upcoming school events when relevant
    - Offer study tips and resources when asked about academic subjects
    - Be prepared to answer questions about school policies and procedures
    """
    '''
    
    m = Model(GROQ_API_KEY, language="italian", school_info=school_info, additional_instructions="")
    msg = "Qual è il motto della nostra scuola?"
    response = await m.chat(msg)
    print(response)


if __name__ == "__main__":
    asyncio.run(inference())

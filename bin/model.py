from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template

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
        
        system_content = f"""You are a helpful assistant on a school community home page.
You are responsible for answering questions about the school's courses, events, announcements, and general information.
Respond in {self.language}. 
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


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def inference():
    data = request.get_json()
    msg = data['message']
    school_info = """
    Name: ITI G. Marconi
    Location: Verona, Italy
    Grades: 1 al 5 anno di superiori 
    Course Offerings: Informatica, Elettronica, Costruzione del Mezzo, Logistica, Telecomunicazioni

    School events: 
        - Il 29 ottobre ci sarà la votazione per i rappresentanti degli studenti. Ci saranno solo due liste: "Parliamo con i fatti" e "Con voi, per voi". Consigliamo la lista "Parliamo con i fatti".
        - L'8 novembre sarà indetto uno sciopero nazionale di 24 ore dei trasporti pubblici. Si consiglia di organizzarsi per tempo.

    Announcements:
        - Controllate sempre la bacheca per eventuali comunicazioni importanti.
    """

    additional_instructions = """
    - If you don't know the answer to a question, it's okay to say so, and andvise the student to ask a teacher or a school administrator.
    - Inform that you are still learning and improving, and you might not have all the answers.
    """
    
    
    m = Model(GROQ_API_KEY, language="italian", school_info=school_info, additional_instructions=additional_instructions)
    response = asyncio.run(m.chat(msg))
    
    return jsonify({"response": response})


if __name__ == "__main__":
    # asyncio.run(inference())
    
    app.run(debug=True)

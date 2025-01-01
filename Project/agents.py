from crewai import Agent
import os
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tools import tool

load_dotenv()
#os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Load a Google LLM Model
#genai=ChatGoogleGenerativeAI(model="gemini-pro",
#        temperature=0.3)

# Load a Groq model
groq_model=ChatGroq(model="llama-3.1-8b-instant",verbose=True,temperature=0.2)


course_plan=Agent(
    role="course plan generator",
    goal="generate a course plan",
    llm=groq_model,
    backstory=(),
    tools=[tool],
    allow_delegation=False
)

assessment=Agent(
    role="assessment generator",
    goal="generate assessments",
    llm=groq_model,
    backstory=(),
    tools=[tool], #verbose=True
    allow_delegation=False # True - If we want a separate tool that gives the final output
)

manager=Agent(
    role="plan generator",
    goal="Genrate a complete course plan",
    llm=groq_model,
    backstory=(),
    memory=True,
    verbose=True,
    allow_delegation=True
)


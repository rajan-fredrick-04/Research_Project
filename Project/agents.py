from crewai import Agent
import os
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tools import tool
from langchain_openai import ChatOpenAI
from litellm import completion

load_dotenv()
#os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
#os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
#os.environ["OPENAI_API_KEY"]="sk-proj-1111"
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# #Load a Google LLM Model
# genai=ChatGoogleGenerativeAI(model="gemini/gemini-pro",verbose=True,
#         temperature=0.0)

genai = completion(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini/gemini-pro"
)

# Load a Groq model
#groq_model=ChatGroq(model="groq/gemma2-9b-it",verbose=True,temperature=0.2,max_tokens=2048)

# Load a Ollama Model
# ollama_model=ChatOpenAI(
#     model="ollama/llama3.2",
#     base_url="http://127.0.0.1:11434 "
# )

course_plan=Agent(
    role="course plan generator",
    goal="Create a structured, comprehensive course plan with evenly distributed 1-hour sessions tailored to the provided unit data.",
    llm=genai,
    backstory=(
        "The agent specializes in generating educational plans for instructors, ensuring logical flow and manageable pacing. "
        "The focus is on breaking down complex unit content into well-structured 1-hour sessions that align with the total teaching hours. "
        "By prioritizing core concepts and maintaining student engagement, the agent supports effective learning outcomes."
    ),
    tools=[tool],
    allow_delegation=True
)

assessment=Agent(
    role="assessment generator",
    goal="Create tailored and comprehensive assessments aligned with course units, ensuring logical categorization, relevance, and cognitive depth.",
    backstory=(
        "The agent specializes in designing educational assessments that align closely with unit content and learning outcomes. "
        "It categorizes vague assessments into clear types such as quizzes, written assignments, hands-on tasks, and visual assessments. "
        "The agent ensures that the generated assessments are ranked for suitability, effectively measure student understanding, and cover all critical concepts in a structured manner."
    ),
    llm=genai,
    tools=[tool], #verbose=True
    allow_delegation=True # True - If we want a separate tool that gives the final output
)

manager = Agent(
    role="plan generator",
    goal="Combine course plans and assessments to generate a comprehensive final report that integrates structured sessions with aligned evaluations.",
    backstory=(
        "The agent acts as a manager that synthesizes outputs from the course plan generator and assessment generator. "
        "Its primary objective is to produce a unified and detailed report that ensures a cohesive learning experience. "
        "By integrating structured session plans with relevant assessments, the agent guarantees alignment between teaching strategies and evaluation methods, offering a complete educational framework."
    ),
    llm=genai,
    memory=True,
    verbose=True,
    allow_delegation=False
)


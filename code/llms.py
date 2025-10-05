import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_perplexity import ChatPerplexity


from dotenv import load_dotenv

load_dotenv()


available_models = [
    "gpt-4o-mini",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "llama3-8b-8192",
    "sonar",
]


def get_llm(model: str):
    if model not in available_models:
        raise ValueError(f"Invalid model. Available models: {available_models.keys()}")

    if model == "gpt-4o-mini":
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    elif model == "llama3-8b-8192":
        return ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        
    elif model == "sonar":
        return ChatPerplexity(
            model ="sonar",
            temperature=0.0,
            api_key=os.getenv("PPLX_API_KEY"),
        )    

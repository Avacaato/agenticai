import os
import platform
import subprocess
from utils import save_text_to_file
# from pathlib import Path
from paths import OUTPUTS_DIR
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_perplexity import ChatPerplexity
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def save_text_to_file(text, path, header=None):
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + "\n\n")
        f.write(text)

def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", str(path)])
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", str(path)])

# Define output path
# output_dir = OUTPUTS_DIR  # Replace with your actual output directory path
# Path(output_dir).mkdir(parents=True, exist_ok=True)

output_file_path = os.path.join(OUTPUTS_DIR, "generated_questions_answers.md")

# Save only the result content


# First chain generates questions
question_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate 3 questions about {topic}:"
)

# Second chain generates answers based on questions
answer_prompt = PromptTemplate(
    input_variables=["questions"],
    template="Answer the following questions:\n{questions}\n You response should contain the question and the answer to it."
)

# Create the model
llm = ChatPerplexity(temperature=0.0)

# Output parser to convert model output to string
output_parser = StrOutputParser()

# Build the question generation chain
question_chain = question_prompt | llm | output_parser

# Build the answer generation chain
answer_chain = answer_prompt | llm | output_parser

# Define a function to create the combined input for the answer chain
def create_answer_input(output):
    return {"questions": output}

# Chain everything together
qa_chain = question_chain | create_answer_input | answer_chain

# Run the chain
result = qa_chain.invoke({"topic": "artificial intelligence"})
# print(result)

save_text_to_file(result, output_file_path)

# Check if file exists, then open
if os.path.exists(output_file_path):
    print(f"Output saved to: {output_file_path}")
    open_file(output_file_path)
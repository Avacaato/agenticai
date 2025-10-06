import os
import platform
import subprocess
from paths import OUTPUTS_DIR, APP_CONFIG_FPATH
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llms import get_llm
from utils import load_publication, save_text_to_file
from langchain.output_parsers.pydantic import PydanticOutputParser
from utils import load_yaml_config

load_dotenv()


class Entity(BaseModel):
    type: str = Field(description="The type of the entity. Either 'model' or 'task'")
    name: str = Field(description="The name of the entity")


class Entities(BaseModel):
    entities: list[Entity] = Field(
        description="The entities mentioned in the publication"
    )

def open_file(path: str):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", path])
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", path])

def no_structured_output(model: str = "sonar"):
    """
    This function demonstrates how to use the LLM without a structured output.
    """
    publication_content = load_publication()

    prompt = """
    Provide a list of entities mentioned in the publication. An entity is either a model or a task.

    <publication>
    {publication_content}
    </publication>
    """.format(
        publication_content=publication_content
    )

    llm = get_llm(model)

    response = llm.invoke(prompt)

    saved_text = f""" # Prompt: {prompt}

# Response:
{response.content}
    """
    file_path = os.path.join(OUTPUTS_DIR, f"no_structured_output_llm_response.md")
    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR, f"no_structured_output_llm_response.md"),
        header=f"LLM Response Without Structured Output",
    )

    if os.path.exists(file_path):
        print(f"LLM response saved to: {file_path}")
        open_file(file_path)
    else:
        print("LLM response was empty or failed.")

def with_prompting_to_structure_output(model: str = "sonar"):
    """
    This function demonstrates how to use the LLM with prompting to structure the output.
    """
    publication_content = load_publication()

    prompt = """
    Provide a list of entities mentioned in the publication. An entity is either a model or a task.

    <publication>
    {publication_content}
    </publication>

    Return a JSON object with a single field "entities" which is a list of dictionaries. Each dictionary should have two fields: "type" and "name".

    Example:
    {{
        "entities": [
            {{
                "type": "model",
                "name": "GPT-4"
            }},
            {{
                "type": "task",
                "name": "Text Classification"
            }}
        ]
    }}
    """.format(
        publication_content=publication_content
    )

    llm = get_llm(model)

    response = llm.invoke(prompt)

    saved_text = f""" # Prompt: {prompt}

# Response:
{response.content}
    """
    file_path = os.path.join(OUTPUTS_DIR, f"with_prompting_to_structure_output_llm_response.md")
    save_text_to_file(
        saved_text,
        os.path.join(
            OUTPUTS_DIR, f"with_prompting_to_structure_output_llm_response.md"
        ),
        header=f"LLM Response With Prompting to Structure Output",
    )

    if os.path.exists(file_path):
        print(f"LLM response saved to: {file_path}")
        open_file(file_path)
    else:
        print("LLM response was empty or failed.")    


def with_output_parser(model: str = "sonar"):
    """
    This function demonstrates how to use the LLM with prompting to structure the output.
    """
    publication_content = load_publication()

    prompt = """
    Provide a list of entities mentioned in the publication. An entity is either a model or a task.

    <publication>
    {publication_content}
    </publication>

    {format_instructions}
    """

    llm = get_llm(model)

    output_parser = PydanticOutputParser(pydantic_object=Entities)

    format_instructions = output_parser.get_format_instructions()

    prompt = prompt.format(
        publication_content=publication_content,
        format_instructions=format_instructions,
    )

    response = llm.invoke(prompt)

    parsed_response = output_parser.parse(response.content)

    saved_text = f""" # Prompt: {prompt}

    # Before Parsing:
    {response.content}

    # After Parsing:
    {parsed_response}
    """
    file_path = os.path.join(OUTPUTS_DIR, f"with_output_parser_llm_response.md")
    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR, f"with_output_parser_llm_response.md"),
        header=f"With Output Parser",
    )

    if os.path.exists(file_path):
        print(f"LLM response saved to: {file_path}")
        open_file(file_path)
    else:
        print("LLM response was empty or failed.")


def model_native_structured_output(model: str = "sonar"):
    publication_content = load_publication()

    prompt = """
    Provide a list of entities mentioned in the publication. An entity is either a model or a task.

    <publication>
    {publication_content}
    </publication>
    """.format(
        publication_content=publication_content
    )

    llm = get_llm(model).with_structured_output(Entities)

    response = llm.invoke(prompt)

    saved_text = f""" # Prompt: {prompt}
    # Response:
    {str(response.model_dump())}
    """
    file_path = os.path.join(OUTPUTS_DIR, f"model_native_structured_output_llm_response.md")
    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR, f"model_native_structured_output_llm_response.md"),
        header=f"LLM Response With Model Native Structured Output",
    )

    if os.path.exists(file_path):
        print(f"LLM response saved to: {file_path}")
        open_file(file_path)
    else:
        print("LLM response was empty or failed.")


if __name__ == "__main__":

    config = load_yaml_config(APP_CONFIG_FPATH)
    model = config.get("llm", "gpt-4o-mini")

    # no_structured_output(model)
    # with_prompting_to_structure_output(model)
    # with_output_parser(model)
    model_native_structured_output(model)

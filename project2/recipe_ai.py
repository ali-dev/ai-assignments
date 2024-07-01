# Importing necessary modules and classes
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import functools
import operator

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable
import json
import requests
load_dotenv()
# Defining tools
tavily_tool = TavilySearchResults(max_results=5)


# System prompt for the supervisor agent
members = ["RecipeResearcher", "UniqueRecipeCreator", "RecipeFileWriter"] #"Recipe Documentor"  # Fixed the names to match the pattern
system_prompt = (
    "You are a head chef tasked with managing information between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH. Assess the results and status from each"
    " worker. Make sure to complete the task in a timely manner."
)

# Options for the supervisor to choose from
options = ["FINISH"] + members

# Function definition for OpenAI function calling
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

# Prompt for the supervisor agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the results and conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# Initializing the language model
llm = ChatOpenAI(model="gpt-4o")


# Creating the supervisor chain
supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)


# Defining a typed dictionary for agent state
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# Function to create an agent
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Function to create an agent node
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

@tool
@traceable
def create_recipe(suggestions: List[str]) -> str:
    """Create a new recipe based on the suggestions that were gathered from the researcher"""
    print("CREATE RECIPE TOOL INVOKED")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    response = llm.invoke(f"Create a unique recipe based on the suggestions: {suggestions} ")    
    return response

@tool
def generate_recipe_photo(recipe_name: str, ingredients: str, instructions: str) -> str:
    """This function generates a photo for the recipe that is created by the agent. 
    """
    print('tool invokation: generate photo for the recipe')
    client = OpenAI()
    prompt = "Generate a photo for a meal. The photo should be be the finished product so do take into account the cooking process in the instructions. the photo should be high quality and made to be displayed on the cookunity website with the intention to intice customers to add the meal to their cart. Here are the details - meal name: "+recipe_name +" ingredients: " +ingredients + " instructions: " + instructions

    print(prompt)
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024",
        model="dall-e-3",
        quality="standard",
        # response_format="b64_json",
    )
    new_file_name = f"recipe.json"
    # masked = np.array(response)
 
    if response.data:
        image_data = response.data[0].url  # Modify this line based on actual structure
        print("Image URL:", image_data)

    
    new_file_name = "recipe.json"
    with open(new_file_name, "w") as file:
        json.dump(response.data[0].url, file)  # Adjust the data extraction based on the actual response structure


    img_data = requests.get(response.data[0].url).content
    with open('recipe_photo.jpg', 'wb') as handler:
        handler.write(img_data)
    # # print(response["data"][0]["b64_json"])
    # with open("./" + new_file_name, mode="w", encoding="utf-8") as file:
    #     json.dump(response, file)
    
    return new_file_name


@tool
def save_recipe(recipe_name: str, ingredients_list: str, instructions: str , photo: str) -> None:
    """This function saves the recipe to a markdown file. The file should include the recipe photo, and a section that has a scaled version of the recipe (100 servings). 
        the recipe format should adhere to the standard here https://github.com/cnstoll/Grocery-Recipe-Format/blob/master/README.md
    """
    print('TOOL INVOKATION: save recipe to pdf')
    print(recipe_name)
    print(ingredients_list)
    print(instructions)
    print(photo)
    with open("recipe.md", "w") as f:
        f.write("Recipe Name: " + recipe_name + "\n")
        f.write("Ingredients list: " + ingredients_list + "\n")
        f.write("Instructions: " + instructions + "\n")
        f.write("Recipe Photo: " + photo + "\n")
    pass


# Creating agents and their corresponding nodes
recipe_researcher_agent = create_agent(llm, [tavily_tool], "You are a web researcher. You search the internet for recipes based on given ingredients.")
research_node = functools.partial(agent_node, agent=recipe_researcher_agent, name="RecipeResearcher")

recipe_creator = create_agent(llm, [create_recipe, generate_recipe_photo], "You are an expert recipe creator. Create a new unique recipe based on the research provided from the recipe researcher.")
creator_node = functools.partial(agent_node, agent=recipe_creator, name="UniqueRecipeCreator")

recipe_writer = create_agent(llm, [save_recipe], "Write the recipe to a file. the file should be writen using a standard format for writing recipes. The file should include the recipe photo, and a section that has a scaled version of the recipe (100 servings) ")
writer_node = functools.partial(agent_node, agent=recipe_writer, name="RecipeFileWriter")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
# test_agent = create_agent(
#     llm,
#     [python_repl_tool],
#     "You may generate safe python code to test functions and classes using unittest or pytest. You run the code. Return only if the code is working or not.",
# )
# test_node = functools.partial(agent_node, agent=test_agent, name="QATester")  # Fixed the name


# Defining the workflow using StateGraph
workflow = StateGraph(AgentState)
workflow.add_node("RecipeResearcher", research_node)
workflow.add_node("UniqueRecipeCreator", creator_node)
workflow.add_node("RecipeFileWriter", writer_node)
workflow.add_node("supervisor", supervisor_chain)

# Adding edges to the workflow
for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Finally, add entry point
workflow.set_entry_point("supervisor")

# Compile the workflow into a graph
graph = workflow.compile()
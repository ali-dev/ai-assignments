## About this project
This is a first draft of a recipe generator application. In this application, I use a graph with multiple agents (RecipeResearcher, UniqueRecipeCreator, RecipeFileWriter) and a supervisor (router) 

## Running the appliation
- make sure to copy `.env-example` to `.env` 
- start the environment `python -m venv venv` and do `pip install -r requirements.txt`
- run python server.py - this will serve the application using langserve with port 8001


## Output
Go to `AI-generated recipes.pdf` to check out receipes that the agent generated

## About this project
This is a chatbot that is built around an AI agent that is able to provide Cookunity customers with general support (help desk), as well as meal recommendations. In the future, this agent will have other tools, that could integrate with other sources of data, ex: a `delivery_status` tool.


## The Tech
I used langchain, open ai, embeddings, and pinecone datastore. 
For the UI, I used streamlit 
For the vector store, I decided to use 2 different indexes even though it is probably not necessary. Just wanted to demonstrate the idea of getting data from different sources. In realtu these sources could be relational dbs, spreadsheets, etc.. 


## Setting up the project
Setup the environment:
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `cp .env.dist .env` and add your pinecone and openIA keys
- in your pinecone, create the following indexes: `cookunity-help-center` and `cookunity-meals` - using `text-embedding-3-small`

To populate the data, go to leaders folders and run the following:
- `python load_help_center_docs.py`
- `python load_meal_pages.py`
Disclaimer: this part of the project could use some more attention, like creating a `base_loader`class, but I didn't want to focus too much on this part.

To run the chatbot:
- go back to the root of the project and run `streamlit run chatbot.py`


<video src="chatbot.mp4" width="300" />



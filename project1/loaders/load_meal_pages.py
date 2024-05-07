from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


load_dotenv()

loader = AsyncChromiumLoader([
    "https://www.cookunity.com/meals/haitian-legume-epis-marinated-beef-vegetable-stew",
    "https://www.cookunity.com/meals/chana-masala",
    "https://www.cookunity.com/meals/butternut-squash-eggplant-red-curry",
    "https://www.cookunity.com/meals/spicy-braised-beef-with-glass-noodles",
    "https://www.cookunity.com/meals/pesto-shrimp-quinoa-bowl",
    "https://www.cookunity.com/meals/lemon-baked-tilapia-with-cajun-cream-sauce",
    "https://www.cookunity.com/meals/ammas-chicken-thighs",
    "https://www.cookunity.com/meals/crispy-tilapia-over-greek-salad",
    "https://www.cookunity.com/meals/sesame-soba-noodle-salad",
    "https://www.cookunity.com/meals/super-fresh-veggie-quinoa-bowl",
    "https://www.cookunity.com/meals/roasted-salmon-with-basil-sauce",
    "https://www.cookunity.com/meals/grilled-hanger-steak-with-asparagus",
    "https://www.cookunity.com/meals/oven-baked-chicken-parmesan",
    "https://www.cookunity.com/meals/parmesan-crusted-chicken",
    "https://www.cookunity.com/meals/grilled-hanger-steak-with-roasted",
    "https://www.cookunity.com/meals/grass-fed-mini-burgers",
    "https://www.cookunity.com/meals/sun-dried-tomato-pesto-rigatoni-with-grilled-asparagus",
    "https://www.cookunity.com/meals/grilled-herb-marinated-hanger-steak",
    "https://www.cookunity.com/meals/gingery-salmon-cakes",
    "https://www.cookunity.com/meals/chicken-avocado-casserole",
    "https://www.cookunity.com/meals/beef-meatballs-with-rosa-sauce",
    "https://www.cookunity.com/meals/montreal-spice-rubbed-flank-steak",
    "https://www.cookunity.com/meals/grilled-shrimp-with-sauteed-bok-choy",
    "https://www.cookunity.com/meals/keto-southwest-chicken-bowl",
    "https://www.cookunity.com/meals/creamy-basil-and-tomato-chicken",
    "https://www.cookunity.com/meals/steak-diane-with-creamy-mushroom-pan-sauce",
    "https://www.cookunity.com/meals/keto-shepherds-pie-with-beef",
    "https://www.cookunity.com/meals/truffled-fettuccine-alfredo-with-shrimp",
    "https://www.cookunity.com/meals/chicken-in-mole-poblano",
    "https://www.cookunity.com/meals/italian-sausage-zoodle-lasagna",
    "https://www.cookunity.com/meals/vegan-chorizo-chili-quesadilla",
    "https://www.cookunity.com/meals/vegan-chorizo-and-black-bean-chili",
    "https://www.cookunity.com/meals/shrimp-moilee-coconut-curry",
    "https://www.cookunity.com/meals/jgs-adobo-chicken-mac-n-cheese",
    "https://www.cookunity.com/meals/jgs-short-rib-mac-n-cheese",
    "https://www.cookunity.com/meals/grilled-skirt-steak-with-herbed-frites",
    "https://www.cookunity.com/meals/vegan-holiday-feast",
    "https://www.cookunity.com/meals/turkey-meatballs-with-sweet-and-spicy-tomato-sauce",
    "https://www.cookunity.com/meals/thanksgiving-glazed-ham",
    "https://www.cookunity.com/meals/turkey-meatloaf-mushroom-gravy",
    "https://www.cookunity.com/meals/slow-roasted-turkey",
    "https://www.cookunity.com/meals/roasted-turkey-sausage-dinner",
    "https://www.cookunity.com/meals/butternut-squash-ravioli-with-candied-bacon",
    "https://www.cookunity.com/meals/herb-roasted-turkey-breast-dinner",
    "https://www.cookunity.com/meals/pesto-pasta-with-turkey-meatballs",
    ])
html = loader.load()
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(html)
print(docs_transformed)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs_transformed)
print(f"Adding {len(documents)} docs to Pinecone")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name="cookunity-meals")


from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

"""
These are the pages that need to be loaded into the vector store for the CookUnity help center.
Unfortunetely, the way these pages are constructed didn't allow for a simple web craweler tto crawl the pages. 
TODO: Add a way to properly crawl these pages.
"""
loader = AsyncChromiumLoader([
    "https://support.cookunity.com/en_us/can-i-cancel-an-order-after-it's-been-charged-rJdaPrZAI",
    "https://support.cookunity.com/en_us/how-do-i-place-a-hold-on-my-account-BknhPSZRL"
    "https://support.cookunity.com/en_us/can-i-reschedule-my-delivery-BkNgOr0L",
    "https://support.cookunity.com/en_us/how-do-i-cancel-my-subscription-H1A2DBbR8"
    "https://support.cookunity.com/en_us/can-i-make-permanent-changes-on-my-weekly-plan-S1ivSZ0I",
    "https://support.cookunity.com/en_us/how-do-i-contact-you-HJAcvHZCU",
    "https://support.cookunity.com/en_us/how-does-my-subscription-work-BJmsvSWRL",
    "https://support.cookunity.com/en_us/do-i-have-to-order-every-week-B1DhwBbAI",
    "https://support.cookunity.com/en_us/what-do-i-need-to-know-before-i-sign-up-for-my-weekly-plan-r1KiPSWCL",
    "https://support.cookunity.com/en_us/how-often-am-i-charged-for-my-meals-SJHhPBbAL",
    "https://support.cookunity.com/en_us/when-is-my-cut-off-time-B14tMjpr5",
    "https://support.cookunity.com/en_us/where-do-i-find-tracking-information-HJrgdHWC8",
    "https://support.cookunity.com/en_us/drop-off-options-rk01urZ0U",
    "https://support.cookunity.com/en_us/will-my-meals-stay-fresh-ryleOrWCU",
    "https://support.cookunity.com/en_us/service-areas-BkT6VU97K",
    "https://support.cookunity.com/en_us/my-order-is-late-ry7A6_pr9",
    "https://support.cookunity.com/en_us/sustainability-recycling-S1S9QHiQF",
    "https://support.cookunity.com/en_us/do-i-get-to-choose-my-meals-S1ojwSWAI",
    "https://support.cookunity.com/en_us/can-i-set-my-dietary-preferences-SJ2fUiLUc",
    "https://support.cookunity.com/en_us/how-do-i-prepare-my-meals-rJAv7Fpt2",
    "https://support.cookunity.com/en_us/can-i-freeze-my-meals-HJivbH6a",
    "https://support.cookunity.com/en_us/password-reset-ryfSCLcQt",
    "https://support.cookunity.com/en_us/how-do-i-change-my-delivery-day-or-time-ByJy4YaHc",
    "https://support.cookunity.com/en_us/i-have-an-issue-with-billing-S17s3zP9",
    "https://support.cookunity.com/en_us/refer-your-friends-and-family-S1OLaViQY",
    "https://support.cookunity.com/en_us/my-promo-code-didn-t-apply-to-my-order-HkKJV3Mvq",
    "https://support.cookunity.com/en_us/where-can-i-find-my-receipts-Bklh1hzvq",
    "https://support.cookunity.com/en_us/how-do-i-apply-a-promo-code-to-my-order-H1q4KsGP9"
    ])
html = loader.load()
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(html)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs_transformed)
print(f"Adding {len(documents)} docs to Pinecone")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name="cookunity-help-center")



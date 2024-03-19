import os
from langchain_openai import AzureOpenAIEmbeddings

os.environ["AZURE_OPENAI_API_KEY"] = "e971867bb3a34fca9edad6ab96c35ca1"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://apac-openai-test.openai.azure.com/"


embedding = AzureOpenAIEmbeddings(
# embedding = OpenAIEmbeddings(
    model="apac-text-embedding-3-large",
    openai_api_version="2022-12-01",
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],

)
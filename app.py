from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Load env
load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone
index_name = "medibot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.4
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

# Helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG function (SAFE)
def get_rag_response(query):
    try:
        docs = retriever.invoke(query)
        context = format_docs(docs)

        final_prompt = prompt.invoke({
            "context": context,
            "input": query
        })

        response = llm.invoke(final_prompt)

        return response.content

    except Exception as e:
        print("ERROR:", e)
        return "⚠️ Error processing request"

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")

    print("User:", msg)

    response = get_rag_response(msg)

    print("Bot:", response)

    return str(response)

# Run
if __name__ == "__main__":
    app.run(debug=True)
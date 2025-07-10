# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

app = FastAPI()

# ✅ load env vars from .env
load_dotenv()

# ✅ get Google API key from env var
google_api_key = os.environ["GOOGLE_API_KEY"]

class QueryRequest(BaseModel):
    video_id: str
    question: str

@app.post("/ask")
def ask_video_question(req: QueryRequest):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(req.video_id, languages=['en'])
        transcript = " ".join([chunk['text'] for chunk in transcript_list])
    except TranscriptsDisabled:
        return {"answer": "No captions available for this video."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    Model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.1
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant.
        Answer ONLY from the context provided transcript context.
        If the context is not sufficient to answer the question, say "I don't know".
        {context}
        Question: {question}""",
        input_variables=["context", "question"]
    )

    def format_docs(retrieved_docs):
        return "\n".join([doc.page_content for doc in retrieved_docs])

    parallel_chain = RunnableParallel({
        'question': RunnablePassthrough(),
        'context': retriever | RunnableLambda(format_docs)
    })

    parser = StrOutputParser()
    final_prompt = parallel_chain | prompt | Model | parser

    response = final_prompt.invoke(req.question)

    return {"answer": response}

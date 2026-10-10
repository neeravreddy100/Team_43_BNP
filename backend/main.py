import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

app = FastAPI(title="IMPART RegTech Intelligence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Local DB & Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = Ollama(model="llama3.2", temperature=0)

class ChangeRequest(BaseModel):
    cr_text: str

IMPART_PROMPT = """You are a Regulatory Compliance AI for a global bank. Evaluate the Change Request (CR) against the provided regulatory context.

GUARDRAILS:
1. NO HALLUCINATIONS: If the context does not mention the regulation, strictly state "No relevant regulation found".
2. TRACEABILITY: You MUST cite the exact Clause or Article from the context in your reason.

Step 1: Analyze the CR intent.
Step 2: Check for conflicts in the context.
Step 3: Format the final output EXACTLY as the following JSON array:
[
  {{
    "Regulation": "ASIC 2024 Rules",
    "Compliant": "Yes" or "No",
    "Reason": "Detailed explanation citing the specific clause and why it passes or fails."
  }}
]

Context: {context}

Change Request: {question}

JSON Output:"""

QA_PROMPT = PromptTemplate(template=IMPART_PROMPT, input_variables=["context", "question"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Modern LCEL Routing Pipeline
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | QA_PROMPT
    | llm
    | StrOutputParser()
)

@app.post("/api/evaluate")
def evaluate_cr(request: ChangeRequest):
    result = qa_chain.invoke(request.cr_text)
    
    return {
        "status": "success", 
        "cr_processed": request.cr_text, 
        "impact_assessment": result
    }
import openai
import os
import shutil
import argparse
import tiktoken
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from docx import Document
import camelot

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

DATA_PATH = "data"
CHROMA_PATH = "chroma"

SUMMARIZER_PROMPT_TEMPLATE = """
Based on the following context, generate a concise 2-page summary and a 1-page summary. Use the delimiter '###' to separate the 2-page summary from the 1-page summary.

2-page Summary:
1. Business Overview: Include information such as the company's formation/incorporation date, headquarters location, business description, employee count, latest revenues, stock exchange listing and market capitalization, number of offices and locations, and details on their clients/customers.

2. Business Segment Overview: Extract the revenue percentage of each component (verticals, products, segments, and sections) as a part of the total revenue. Evaluate the performance of each component by comparing the current year's sales/revenue and market share with the previous year's numbers. Explain the causes of the increase or decrease in the performance of each component.

3. Geographical Segment Overview: Breakdown of sales and revenue by geography, specifying the percentage contribution of each region to the total sales. Summarize geographical data, such as workforce, clients, and offices, and outline the company's regional plans for expansion or reduction. Analyze and explain regional sales fluctuations, including a geographical sales breakdown to identify sales trends.

4. Year-over-Year Analysis: Summarize year-over-year sales increase or decline and reasons for the change.

5. Summary of Rationale & Considerations: Provide an analysis of the risks and mitigating factors.

6. SWOT Analysis: Summarize the company's strengths, weaknesses, opportunities, and threats.

7. Credit Rating Information: Provide information about credit rating/credit rating changes/changes in the rating outlook.

### 1-page Summary:
The 1-page summary should be a condensed version of the 2-pager, including key numbers and important points from the business segment overview and geographical segment overview.

Context:
{context}

---

Generate the summaries based on the above context.
"""

def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear the database.")
    args = parser.parse_args()
    if args.clear:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    
    # Initialize the embedding model once
    embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
    save_to_chroma(chunks, embed_model)

    two_page_summary, one_page_summary, total_tokens = summarize_documents()
    print(f"Total tokens used: {total_tokens}")

    # Save summaries to .docx files
    save_summary_to_docx(two_page_summary, "2_page_summary.docx")
    save_summary_to_docx(one_page_summary, "1_page_summary.docx")

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    
    # Extract table data from PDFs
    for doc in documents:
        pdf_path = os.path.join(DATA_PATH, doc.metadata["source"])
        tables = extract_tables_from_pdf(pdf_path)
        doc.page_content += "\n\n".join(tables)
    
    return documents

def extract_tables_from_pdf(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages="all", flavor='stream')
    table_data = []
    for table in tables:
        table_data.append(table.df.to_string())
    return table_data

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def save_to_chroma(chunks, embed_model):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, embedding=embed_model, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def summarize_documents():
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Combine all chunks into one context text for summarization
    results = db.similarity_search_with_score(SUMMARIZER_PROMPT_TEMPLATE, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text)

    # Count tokens for the prompt
    prompt_tokens = count_tokens(prompt)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    response_text = response.choices[0].message['content']

    # Split the response into 2-page and 1-page summaries
    two_page_summary, one_page_summary = response_text.split("###", 1)
    one_page_summary = one_page_summary.strip()

    # Count tokens for the response
    response_tokens = count_tokens(response_text)

    total_tokens = prompt_tokens + response_tokens

    print(f"2-page Summary: {two_page_summary}")
    print(f"1-page Summary: {one_page_summary}")

    return two_page_summary.strip(), one_page_summary.strip(), total_tokens

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def save_summary_to_docx(summary, filename):
    doc = Document()
    doc.add_paragraph(summary)
    doc.save(filename)
    print(f"Saved summary to {filename}")

if __name__ == "__main__":
    main()

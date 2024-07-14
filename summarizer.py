import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from docx import Document

# Set your OpenAI API key here
openai.api_key = 'your-api-key-here'

def pdf_to_text(file_path):
    """Convert PDF to text."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def get_embeddings(text, model_name='all-MiniLM-L6-v2'):
    """Convert text into embeddings using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    sentences = text.split('\n')
    embeddings = model.encode(sentences)
    return sentences, embeddings

def perform_similarity_search(query, sentences, embeddings, top_k=5):
    """Perform similarity search to find relevant chunks."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(np.array([query_embedding]), top_k)
    return [sentences[i] for i in I[0]]

def summarize_section(text, section_title, max_tokens=1500):
    """Summarize a specific section using OpenAI API."""
    prompt = f"Extract and summarize the following section from the financial report: {section_title}. Here is the text:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    summary = response.choices[0].message['content']
    return summary

def generate_summaries(financial_text):
    """Generate summaries for each section and combine them."""
    sections = [
        "Business Overview",
        "Business Segment Overview",
        "Performance",
        "Sales Increase/Decrease explanation",
        "Breakdown of sales and revenue by geography",
        "Geographical data summary",
        "Regional sales fluctuations",
        "Year-over-year sales changes",
        "Rationale & considerations",
        "SWOT Analysis",
        "Credit rating information"
    ]
    
    summaries = {}
    sentences, embeddings = get_embeddings(financial_text)
    for section in sections:
        relevant_chunks = perform_similarity_search(section, sentences, embeddings)
        relevant_text = '\n'.join(relevant_chunks)
        summaries[section] = summarize_section(relevant_text, section, max_tokens=3000 if section == "Business Overview" else 1500)
    
    # Generate combined summaries
    two_page_summary = "\n\n".join([f"{section}\n{summaries[section]}" for section in summaries])
    one_page_summary = summarize_section(two_page_summary, "1-page summary", max_tokens=1500)
    
    return two_page_summary, one_page_summary

def generate_summary_doc(summary, output_path):
    """Generate a .docx file from the summary text."""
    doc = Document()
    doc.add_heading('Financial Summary', level=1)
    for section in summary.split('\n\n'):
        doc.add_paragraph(section)
    doc.save(output_path)

def main():
    """Main function to convert PDF to text and generate summaries."""
    # Convert PDF to text
    financial_text = pdf_to_text('financial_report.pdf')
    
    # Generate and save summaries
    two_page_summary, one_page_summary = generate_summaries(financial_text)
    
    generate_summary_doc(two_page_summary, 'two_page_summary.docx')
    generate_summary_doc(one_page_summary, 'one_page_summary.docx')

if __name__ == '__main__':
    main()
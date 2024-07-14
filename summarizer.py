import PyPDF2
import openai
from docx import Document

openai.api_key = 'open ai api key will come here'

def pdf_to_text(file_path):
    """Convert PDF to text."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text 

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
    for section in sections:
        summaries[section] = summarize_section(financial_text, section, max_tokens=3000 if section == "Business Overview" else 1500)
    
    # Summaries
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

    # Extract text from pdf
    financial_text = pdf_to_text('financial_report.pdf')
    
    # Generate and save summaries
    two_page_summary, one_page_summary = generate_summaries(financial_text)
    
    generate_summary_doc(two_page_summary, 'two_page_summary.docx')
    generate_summary_doc(one_page_summary, 'one_page_summary.docx')
    file = open("myfile.txt","w")

    file.write(financial_text)
    file.close()


if __name__ == '__main__':
    main()
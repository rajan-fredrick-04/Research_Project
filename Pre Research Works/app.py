import streamlit as st
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
import tempfile

def process_csv(file):
    # Step 1: Load CSV and extract structured data
    loader = CSVLoader(file_path=file.name, csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['', 'Unit', 'Topic', 'Contents', 'Teaching Hours', 'Course Outcomes', 'Similarity Score', 'Verbs', 'Assessments']
    })
    documents = loader.load()

    data = []
    for doc in documents:
        content = doc.page_content.strip()
        content_lines = content.split('\n')
        if len(content_lines) >= 9:
            entry = {
                "Unit": content_lines[0].strip(),
                "Topic": content_lines[2].strip(),
                "Contents": content_lines[3].strip(),
                "Assessments": set(content_lines[8].strip().split(';'))
            }
            entry["Assessments"] = ', '.join(entry["Assessments"])
            data.append(entry)

    # Step 2: Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["unit", "contents", "assessments"],
        template=(
            """
You are an AI specializing in educational assessments, designed to work through tasks step by step to generate targeted and comprehensive assessments. Below is information about a unit:
Unit: {unit}
Contents: {contents}
Vague Assessments: {assessments}

Task:

Identify Assessment Types: Analyze the provided vague assessments and identify a broad range of specific assessment types, such as 'essay,' 'MCQ,' 'fill in the blank,' 'case study,' 'diagram labeling,' 'problem-solving,' 'coding exercises,' or others.
Link to Content: Extract the key topics and concepts from the unit's content that align with the identified assessment types.
Generate Tailored Assessments: For each assessment type, create detailed and actionable assessments based on the unit's content. Examples include:
Essay: If the content includes 'word embeddings,' the assessment could be: "Write a detailed essay explaining the concept of word embeddings and their role in natural language processing."
Fill in the Blank: For the topic 'vector representations,' the assessment could be: "_____ is the process of representing words as vectors in a high-dimensional space."
Case Study: If the content covers 'applications of word embeddings,' the assessment could be: "Analyze a case study where word embeddings were used to improve search engine results."
Diagram Labeling: If the content involves 'neural network structures,' the assessment could be: "Label the components of the diagram showing the architecture of a word embedding model."
Problem-Solving: For a topic on 'cosine similarity,' the assessment could be: "Calculate the cosine similarity between the following two word vectors."
Deliverable: Provide diverse, specific, and measurable assessments tailored to the unit, ensuring alignment with the content and coverage of a wide range of assessment types.
            """
        )
    )

    # Step 3: Initialize LLM and create a chain
    llama = Ollama(model="llama3.2")
    llm_chain = LLMChain(llm=llama, prompt=prompt_template)

    # Step 4: Generate specific assessments
    specific_assessments = []
    for entry in data:
        response = llm_chain.run({
            "unit": entry["Unit"],
            "contents": entry["Contents"],
            "assessments": entry["Assessments"]
        })
        specific_assessments.append({"Unit": entry["Unit"], "Specific Assessments": response})

    # Convert to DataFrame
    return pd.DataFrame(specific_assessments)

# Streamlit app
st.title("Educational Assessments Generator")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.write("Processing the file...")
    output_df = process_csv(tmp_file)

    # Downloadable output
    st.success("Processing complete!")
    st.write(output_df)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(output_df)

    st.download_button(
        label="Download specific assessments as CSV",
        data=csv,
        file_name="specific_assessments.csv",
        mime="text/csv",
    )

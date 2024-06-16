import streamlit as st
import openai
import pandas as pd

# Set your OpenAI API key
openai.api_key = "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"

# Model ID
model_id = "ft:gpt-3.5-turbo-0125:valonylabsz:finetune-itemclass:9aIqocEw"

def classify_text(input_text):
    response = openai.Completion.create(
        model=model_id,
        prompt=input_text,
        max_tokens=10
    )
    return response.choices[0].text.strip()

st.title("Text Classification with OpenAI Fine-Tuned Model")

st.write("This app classifies input text using a fine-tuned GPT-3.5 model.")

# Option to choose between single input and tabular input
input_type = st.radio("Choose input type:", ("Single Input", "Tabular Input"))

if input_type == "Single Input":
    single_input = st.text_area("Enter text to classify:")
    if st.button("Classify"):
        if single_input:
            result = classify_text(single_input)
            st.write("Classification Result:", result)
        else:
            st.write("Please enter some text to classify.")

elif input_type == "Tabular Input":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(df)
        
        if st.button("Classify"):
            df['Classification'] = df.iloc[:, 0].apply(classify_text)
            st.write("Classification Results:")
            st.write(df)

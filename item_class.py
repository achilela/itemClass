import streamlit as st
import openai
import pandas as pd
import os

# Load the OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the model ID
model_id = "ft:gpt-3.5-turbo-0125:valonylabsz:finetune-itemclass:9aIqocEw"

# Function to classify a single item
def classify_item(item):
    response = openai.Completion.create(
        model=model_id,
        prompt=f"Classify the following maintenance item: {item}",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
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
            result = classify_item("Please classify the following text:" single_input, model_id)
            if result is not None:
                st.write("Classification Result:", result)
            else:
                st.write("Failed to classify the text. Please check the error message above.")
        else:
            st.write("Please enter some text to classify.")

elif input_type == "Tabular Input":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(df)
        
        if st.button("Classify"):
            df['Classification'] = df.iloc[:, 0].apply(lambda x: Inference_func("Please classify the following text:", x, model_id))
            st.write("Classification Results:")
            st.write(df)

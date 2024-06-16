import openai
import streamlit as st
import pandas as pd
import os

# Set your OpenAI API key (replace 'YOUR_OPENAI_API_KEY' with your actual API key or use an environment variable)
openai.api_key = os.getenv("YOUR_OPENAI_API_KEY")

# Model ID
model_id = "ft:gpt-3.5-turbo-0125:valonylabsz:finetune-itemclass:9aIqocEw"

def classify_text(text, model):
    prompt = f"Please classify the following text: '{text}'"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=60
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

st.title("Text Classification with OpenAI Fine-Tuned Model")

st.write("This app classifies input text using a fine-tuned GPT-3.5 model.")

# Option to choose between single input and tabular input
input_type = st.radio("Choose input type:", ("Single Input", "Tabular Input"))

if input_type == "Single Input":
    single_input = st.text_area("Enter text to classify:")
    if st.button("Classify"):
        if single_input:
            result = classify_text(single_input, model_id)
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

        column_to_classify = st.selectbox("Select the column to classify", df.columns)

        if st.button("Classify"):
            df['Classification'] = df[column_to_classify].apply(lambda x: classify_text(x, model_id))
            st.write("Classification Results:")
            st.write(df)
            df.to_csv('classified_output.csv', index=False)
            st.write("Results saved to 'classified_output.csv'")

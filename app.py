import streamlit as st
import os
from huggingface_hub import InferenceClient #preinstalled native wrapper
# 1. Pull the classic 'Write' token out of your secret settings 
HF_TOKEN = os.getenv("HF_TOKEN") 
# 2. Fire up the native client using the token 
client = InferenceClient(token=HF_TOKEN)

st.title("😊 Mood Detector")
text = st.text_area("How are you feeling?")

if st.button("Analyze Mood"): 
    if not topic.strip(): 
        st.warning("Please tell us how are you feeling?") 
    elif not HF_TOKEN: 
        st.error("Your HF_TOKEN secret key is entirely missing from Settings!")
    else:
        with st.spinner("Processing natively through the cluster..."): 
            try:
                # 3. Route via internal endpoints rather than requests.post 
                response = client.chat.completions.create( model="j-hartmann/emotion-english-distilroberta-base", messages=[{"role": "user", "content": topic}], max_tokens=50 ) 
                # 5. Extract output clean string text safely 
                answer = response.choices[0].message.content 
                st.success(answer) 
            except Exception as e: 
                # This explicitly unpacks server messages instead of dropping HTML text 
                st.error(f"System Response: {e}")

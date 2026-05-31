import streamlit as st
import os
from huggingface_hub import InferenceClient #preinstalled native wrapper
# 1. Pull the classic 'Write' token out of your secret settings 
HF_TOKEN = os.getenv("HF_TOKEN") 
# 2. Fire up the native client using the token 
client = InferenceClient(token=HF_TOKEN)
MODEL = "j-hartmann/emotion-english-distilroberta-base"
st.title("😊 Mood Detector")
topic = st.text_area("How are you feeling?")

if st.button("Analyze Mood"): 
    if not topic.strip(): 
        st.warning("Please tell us how are you feeling?") 
    elif not HF_TOKEN: 
        st.error("Your HF_TOKEN secret key is entirely missing from Settings!") 
    else:
        with st.spinner("Processing natively through the cluster..."): 
            try:
                # 3. Route via internal endpoints rather than requests.post 
                response = client.text_classification(
                                    topic,
                                    model=MODEL
                                )
                # 5. Extract output clean string text safely 
                st.subheader("Detected Emotions")
                for emotion in response:
                    st.write(f"**{emotion.label}** : {emotion.score:.2%}")
                top_emotion = max(response, key=lambda x: x.score)
                st.success( f"Primary Emotion: {top_emotion.label}") 
            except Exception as e: 
                # This explicitly unpacks server messages instead of dropping HTML text 
                st.error(f"System Response: {e}")

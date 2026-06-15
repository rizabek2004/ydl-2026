import urllib.error

import streamlit as st

from generate_image import generate, load_image_creds

st.title("Text-to-Image Generator")

prompt = st.text_input("What image do you want to generate?")

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        model, url, key = load_image_creds()
        with st.spinner("Generating image..."):
            try:
                image_bytes = generate(model, url, key, prompt)
            except urllib.error.HTTPError as e:
                st.error(f"Request failed ({e.code}): {e.read().decode('utf-8', 'replace')}")
            else:
                st.image(image_bytes, caption=prompt)

import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Generative Design RAG", layout="wide", page_icon="🎨")

# --- 1. THE KNOWLEDGE BASE (Frontend Mapping) ---
# We map the names to the URLs so we can show the user what was picked.
# These MUST match the names you used in the Colab 'style_data' dictionary.
STYLE_DB = {
     "Cyberpunk": "https://images.unsplash.com/photo-1515036551567-bf1198cccc35?q=80&w=2076&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "Ghibli": "https://images.unsplash.com/photo-1630207831419-3532bcb828d7?q=80&w=1931&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "Sketch": "https://plus.unsplash.com/premium_vector-1711987527538-4ff22f0fdbbc?q=80&w=1922&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "Industrial": "https://images.unsplash.com/photo-1505577058444-a3dab90d4253?w=600"
}

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("🔗 Ngrok URL:", placeholder="https://....ngrok-free.app")
    st.info("Paste the URL from your running Colab instance here.")

# --- MAIN LAYOUT ---
st.title("🎨 Generative Design RAG Engine")
st.markdown("Transform sketches using **Text-to-Style Retrieval**.")

# A. PREVIEW THE DATABASE
with st.expander("📚 View Style Knowledge Base (Available Presets)"):
    st.caption("The RAG engine will search through these images to find the best match for your text.")
    cols = st.columns(4)
    for index, (name, url) in enumerate(STYLE_DB.items()):
        with cols[index]:
            st.image(url, caption=name, use_container_width=True)

st.divider()

# B. INPUTS
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Input Structure")
    uploaded_file = st.file_uploader("Upload Sketch/Wireframe", type=["png", "jpg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Your Structure", use_container_width=True)

with col2:
    st.subheader("2. Describe Desired Style")
    user_prompt = st.text_area(
        "What vibe do you want?", 
        height=150, 
        placeholder="e.g. A futuristic neon city with rain... \n(The AI will semantic search the database for the closest match)"
    )
    
    generate_btn = st.button("🚀 Run RAG Pipeline", type="primary", use_container_width=True)

# C. RESULTS DISPLAY
if generate_btn and uploaded_file and api_url:
    
    st.divider()
    st.subheader("3. RAG Pipeline Results")
    
    with st.spinner("🔍 Phase 1: Semantic Search... Phase 2: Generating..."):
        try:
            # Prepare payload
            files = {"file": uploaded_file.getvalue()}
            data = {"prompt": user_prompt}
            
            # --- FIX: BYPASS NGROK SECURITY CHECKS ---
            # 1. 'ngrok-skip-browser-warning': 'true' -> Bypasses the "Visit Site" button
            # 2. verify=False -> Ignores the SSL Decryption error
            
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # Silence the warning in logs

            headers = {"ngrok-skip-browser-warning": "true"}
            
            response = requests.post(
                f"{api_url}/generate", 
                files=files, 
                data=data, 
                headers=headers,    # <--- NEW HEADER
                verify=False        # <--- DISABLE SSL CHECK
            )
           
            
            if response.status_code == 200:
                # 1. Get the Image
                generated_img = Image.open(io.BytesIO(response.content))
                
                # 2. Get the RAG Metadata (Which style did it pick?)
                # We grab this from the custom header we added
                retrieved_style_name = response.headers.get("X-Retrieved-Style", "Unknown")
                
                # 3. Show the Flow
                r_col1, r_col2, r_col3 = st.columns(3)
                
                with r_col1:
                    st.info("Step 1: Input")
                    st.image(uploaded_file, use_container_width=True)
                    
                with r_col2:
                    st.success(f"Step 2: Retrieved '{retrieved_style_name}'")
                    # Show the image that matches the retrieved name
                    if retrieved_style_name in STYLE_DB:
                        st.image(STYLE_DB[retrieved_style_name], caption=f"Closest Match in DB", use_container_width=True)
                    else:
                        st.warning(f"Style '{retrieved_style_name}' not found in local preview.")
                        
                with r_col3:
                    st.balloons()
                    st.caption("Step 3: Final Output")
                    st.image(generated_img, use_container_width=True)
                    
            else:
                st.error(f"Server Error: {response.text}")

        except Exception as e:
            st.error(f"Connection Failed: {e}")
# import plotly
# print(plotly.__version__)


import streamlit as st
import sys
import subprocess

st.write("Python version:", sys.version)
st.write("Python executable:", sys.executable)

try:
    import plotly
    st.write("Plotly version:", plotly.__version__)
except ImportError:
    st.error("Plotly is not installed in the current Python environment.")
    
    # Try to install plotly
    st.write("Attempting to install Plotly...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        st.success("Plotly installed successfully. Please restart the Streamlit app.")
    except subprocess.CalledProcessError:
        st.error("Failed to install Plotly. Please install it manually.")
        
        
        
def main():
    st.title("RAG App with Chroma DB")
    st.write("This is a placeholder. The full app will be implemented once we resolve the Plotly issue.")

if __name__ == "__main__":
    main()
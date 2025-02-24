# FAQ Chatbot

## Overview
This project focuses on developing a **customer-centric chatbot** using the **Llama-2-7b-chat-hf** model. The chatbot is designed to provide **contextually accurate customer service responses** without requiring additional training datasets. The implementation integrates a **domain-specific policy document** using **LlamaIndex** and features a user-friendly **Streamlit** interface.

## Features
- **Powered by Llama 2**: Utilizes the transformer-based Llama-2 model for natural language processing.
- **Retrieval-Augmented Generation (RAG)**: Enhances the chatbot’s ability to provide domain-specific responses.
- **LlamaIndex Integration**: Allows dynamic retrieval of information from policy documents.
- **User-Friendly Interface**: Built with Streamlit for easy interaction.

## Project Architecture
The chatbot system consists of three primary components:
1. **Core NLP Model**
   - Uses **Llama 2** for language understanding and response generation.
2. **Knowledge Retrieval Layer**
   - Implements **LlamaIndex** to retrieve relevant policy document sections.
3. **User Interface Layer**
   - Developed with **Streamlit** to provide an interactive web-based chatbot experience.

## Implementation
### Technologies Used
- **Llama-2-7b-chat-hf** (pre-trained transformer model)
- **LlamaIndex** (for Retrieval-Augmented Generation)
- **Streamlit** (for the chatbot UI)
- **Python** (for backend processing)

### Steps to Run the Project
1. **Clone the Repository**
   ```bash
   git clone https://github.com/refun70/FAQ_Chatbot.git
   cd FAQ_Chatbot
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Chatbot**
   ```bash
   streamlit run app.py
   ```

## Evaluation and Results
The chatbot’s performance was evaluated using **BLEU score** and **Semantic Similarity**:
- **Semantic Similarity Score**: **0.68** (shows good contextual relevance)
- **BLEU Score**: **0.0015** (low due to the model’s flexible phrasing)

## Challenges Encountered
- Setting up **Jupyter Notebook** with GPU acceleration.
- Installing **LlamaCPP**, which relies on external libraries.
- **VRAM limitations** (only 6GB) restricting further fine-tuning.
- Discrepancy between **BLEU score** and **semantic similarity** results.

## Future Improvements
- Enhance **model fine-tuning** for better accuracy.
- Optimize **GPU memory usage** for large-scale processing.
- Improve **response coherence** using advanced tuning techniques.

## References
- [Llama-2-7b-chat-hf on Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/)
- [Streamlit Documentation](https://streamlit.io/)

---
🚀 **Developed by Rojen Dangol | Torrens University**


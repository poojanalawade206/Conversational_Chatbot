🧠 Conversational AI Chatbot

An AI-powered conversational chatbot built using FastAPI, LangChain, and OpenAI.
It can engage in natural language conversations, understand context, detect sentiment, and extract named entities from user input.

🚀 Features

Conversational AI – Handles general queries with context-aware responses
Context Management – Set and retrieve conversation context
Sentiment Analysis – Detects user sentiment (positive / negative / neutral)
Entity Extraction – Extracts key entities like names, locations, and organizations
RESTful API – Fully accessible via FastAPI’s interactive Swagger UI

🏗️ Tech Stack

FastAPI – REST API framework
LangChain – Context & conversation management
OpenAI GPT – Language understanding & response generation
spaCy – Entity recognition
Python 3.10+

📁 Project Structure
Conversational AI Chatbot/
│
├── fastapi_langchain_chatbot.py    # Main application file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignored files/folders
└── README.md                       # Project documentation

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/poojanalawade206/Conversational_Chatbot.git
cd Conversational_Chatbot

2️⃣ Create Virtual Environment
python -m venv .venv

3️⃣ Activate Virtual Environment
Windows:
.venv\Scripts\activate
macOS/Linux:
source .venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Add Your OpenAI API Key
Create a .env file in your project root:
OPENAI_API_KEY=your_openai_api_key_here

▶️ Run the Chatbot API
uvicorn fastapi_langchain_chatbot:app --reload

Once running, open the API documentation at:
👉 http://127.0.0.1:8000/docs

💬 API Endpoints
Endpoint	Method	Description
/chat	POST	Send user input and get chatbot response
/context	GET	Retrieve current conversation context
/context	POST	Set or update conversation context
/sentiment	POST	Analyze sentiment of input text
/entities	POST	Extract named entities from input

Example API Calls
🗨️ 1. Chat
POST /chat
{
  "input": "Who is the CEO of Google?"
}

Response
{
  "response": "The CEO of Google is Sundar Pichai."
}

2. Set Context
POST /context
{
  "context": "You are a helpful assistant that introduces itself politely."
}

Response
{
  "message": "Context updated successfully"
}

3. Sentiment
POST /sentiment
{
  "input": "I love working with AI!"
}

Response
{
  "sentiment": "positive"
}

4. Entities
POST /entities
{
  "input": "I work at Microsoft in Bangalore."
}

Response
{
  "entities": ["Microsoft", "Bangalore"]
}

🧰 Future Enhancements
Add database memory for long-term conversation history
Integrate text-to-speech (TTS) and speech-to-text (STT)
Build a frontend chat UI using React or Streamlit
Add multilingual support

Author

Pooja Nalawade
GitHub: @poojanalawade206
Project: Conversational_Chatbot

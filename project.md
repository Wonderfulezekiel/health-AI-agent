 Documentation: Personal Health & Wellness AI Chat Agent (LangChain + Gemini + Supabase)
📌 Project Overview
Create a personal AI assistant that helps users achieve their health and wellness goals by chatting with them, offering personalized suggestions, and storing their health goals and chat history using LangChain, Gemini, Supabase, FastAPI, and Streamlit.

⚙️ Tech Stack
Layer	Technology
LLM	Gemini Pro (Google Generative AI)
Agent Logic	LangChain
Memory/Storage	Supabase (PostgreSQL)
Backend	FastAPI
Frontend	Streamlit

🧠 Features
Chat with the agent about health goals (e.g., “How can I quit sugar?”)

Agent responds with step-by-step suggestions

Stores and remembers conversations using Supabase

Tracks long-term health goals and user behavior

Simple UI for interaction via Streamlit

✅ Setup Instructions
1. Environment
Install required packages:

bash
Copy
Edit
pip install langchain google-generativeai fastapi uvicorn streamlit supabase faiss-cpu python-dotenv
Create .env file:

env
Copy
Edit
GOOGLE_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_public_anon_key
2. Supabase Configuration
Create a table in Supabase SQL editor:

sql
Copy
Edit
create table chat_logs (
  id uuid primary key default gen_random_uuid(),
  user_id text,
  message text,
  role text,
  created_at timestamp with time zone default timezone('utc'::text, now())
);
3. LangChain + Gemini LLM Setup
Initialize Gemini LLM with LangChain

Connect Supabase client

Create functions to store and retrieve chat logs

4. FastAPI Backend
POST /chat endpoint

Accepts user_id and message

Retrieves user history from Supabase

Constructs prompt with chat history

Gets response from Gemini

Saves both user message and agent response to Supabase

Returns response as JSON

5. Streamlit Frontend
UI with input box for user query

Generates user_id with uuid on first session

Sends POST request to FastAPI /chat with user_id and message

Displays response from agent

6. Memory Management
Chat history retrieved from Supabase in chronological order

Combined into a conversation string to maintain context

Stored per user by user_id


✅ How to Run
Start FastAPI backend:

bash
Copy
Edit
uvicorn main:app --reload
Run Streamlit frontend:

bash
Copy
Edit
streamlit run chat_ui.py
📁 Suggested File Structure
bash
Copy
Edit
project/
│
├── .env
├── llm_setup.py          # LLM + Supabase logic
├── main.py               # FastAPI backend
├── chat_ui.py            # Streamlit UI
├── requirements.txt
└── README.md             # Project overview

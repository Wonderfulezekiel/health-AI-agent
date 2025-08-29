import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_react_agent
from tools.tavily_search import get_tavily_tools
from tools.openfoodfacts import get_openfoodfacts_tools
from tools.spoonacular import get_spoonacular_tools
from langchain import hub
import os
from uuid import uuid4
import re
from typing import Optional
import json
import urllib.parse
import urllib.request

# Optional Supabase import (do not break if library is missing)
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None  # type: ignore
    Client = None  # type: ignore

# Load environment variables
load_dotenv()

# Initialize Supabase client if credentials and library are available
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase_client = None
if create_client and SUPABASE_URL and SUPABASE_ANON_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception:
        supabase_client = None

# Helper functions for Supabase-backed memory

def _fetch_messages_from_supabase(conversation_id: str, limit: int = 50):
    if not supabase_client:
        return []
    try:
        result = (
            supabase_client
            .table("messages")
            .select("role, content, created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        rows = result.data or []
        messages = []
        for row in rows:
            role = row.get("role")
            content = row.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages
    except Exception:
        return []


def _save_message_to_supabase(conversation_id: str, role: str, content: str):
    if not supabase_client:
        return
    try:
        _ = (
            supabase_client
            .table("messages")
            .insert({
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
            })
            .execute()
        )
    except Exception:
        pass

# Optional: upload image to Supabase Storage and return a URL

def _upload_image_to_supabase(conversation_id: str, filename: str, file_bytes: bytes, content_type: str | None = None) -> str | None:
    if not supabase_client:
        return None
    try:
        bucket = os.getenv("SUPABASE_BUCKET", "meal-images")
        path = f"{conversation_id}/{uuid4()}-{filename}"
        file_opts = {"contentType": content_type or "application/octet-stream", "upsert": "true"}
        resp = supabase_client.storage.from_(bucket).upload(path=path, file=file_bytes, file_options=file_opts)
        # Some clients return dict with error key
        if isinstance(resp, dict) and resp.get("error"):
            return None
        # Prefer signed URL if bucket is private
        try:
            signed = supabase_client.storage.from_(bucket).create_signed_url(path, 60 * 60)
            url = signed.get("signedURL") or signed.get("signed_url")
            if url:
                return url
        except Exception:
            pass
        # Fallback to public URL (works if bucket is public)
        try:
            public = supabase_client.storage.from_(bucket).get_public_url(path)
            url = public.get("publicURL") or public.get("public_url")
            if url:
                return url
        except Exception:
            pass
        return None
    except Exception:
        return None

# Heuristics to guide image-only prompts without invoking the agent

def _has_recent_uploaded_image_message() -> bool:
    for msg in reversed(st.session_state.get("messages", [])):
        if msg.get("role") == "user" and msg.get("content", "").startswith("Uploaded image:"):
            return True
    return False


def _prompt_has_product_identifier(prompt: str) -> bool:
    """
    Checks if the prompt contains a specific product identifier (e.g., barcode, brand name).
    """
    text = (prompt or "").lower()
    # Barcode-like numeric token (>= 8 digits)
    if re.search(r"\b\d{8,}\b", text):
        return True
    # Contains non-generic words suggesting a product name
    generic = {"this", "that", "it", "food", "meal", "image", "photo", "uploaded", "healthy"}
    tokens = [t for t in re.findall(r"[a-z0-9]+", text) if t]
    meaningful = [t for t in tokens if t not in generic and len(t) >= 3]
    return len(meaningful) >= 1

# Helpers to support private Storage URLs and multimodal chat

def _parse_storage_url(url: str):
    try:
        m = re.search(r"/storage/v1/object/(?:sign|public)/([^/]+)/(.+)$", url)
        if not m:
            return None, None
        bucket = m.group(1)
        path_with_query = m.group(2)
        path = path_with_query.split("?")[0]
        # Decode percent-encoding (important for sign URLs)
        try:
            from urllib.parse import unquote
            path = unquote(path)
        except Exception:
            pass
        return bucket, path
    except Exception:
        return None, None


def _ensure_accessible_image_url(url: str) -> str | None:
    if not url:
        return None
    if not supabase_client:
        return url
    bucket, path = _parse_storage_url(url)
    if not bucket or not path:
        return url
    try:
        signed = supabase_client.storage.from_(bucket).create_signed_url(path, 60 * 60)
        fresh = signed.get("signedURL") or signed.get("signed_url")
        return fresh or url
    except Exception:
        return url


def _extract_image_urls(content: str) -> list[str]:
    urls: list[str] = []
    if not content:
        return urls
    for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content):
        urls.append(m.strip())
    if "Uploaded image:" in content:
        for m in re.findall(r"https?://[^\s)]+", content):
            urls.append(m.strip())
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def _get_latest_image_url() -> Optional[str]:
    for msg in reversed(st.session_state.get("messages", [])):
        if msg.get("role") == "user":
            urls = _extract_image_urls(msg.get("content", ""))
            if urls:
                fresh = _ensure_accessible_image_url(urls[-1])
                if fresh:
                    return fresh
    return None

# Vision product extraction

def _extract_product_from_image_with_vision(image_url: str, llm_client: ChatGoogleGenerativeAI, question: str) -> Optional[str]:
    """
    Analyzes an image of a meal and returns a detailed description of its contents.
    """
    try:
        prompt_text = f"""You are an expert nutritional analyst. Your task is to analyze the provided image of a meal and extract detailed information.

First, address the user's specific question: "{question}"

Then, provide a comprehensive analysis covering the following points:
1.  **Meal Identification:** What is the meal?
2.  **Key Ingredients:** List the primary ingredients you can identify.
3.  **Cooking Method:** How does it appear to be cooked? (e.g., grilled, fried, baked, steamed).
4.  **Estimated Nutritional Information:** Provide an estimate for Calories, Protein (g), Carbohydrates (g), and Fats (g).
5.  **Healthiness Assessment:** Briefly explain if the meal seems generally healthy and why.

Present your analysis clearly.
"""
        parts = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        msg = HumanMessage(content=parts)
        res = llm_client.invoke([msg])
        text = (res.content or "").strip()
        if text and text.lower() != "unknown":
            return text
        return None
    except Exception as e:
        print(f"Error in vision extraction: {e}")
        st.error(f"Vision analysis failed: {e}")
        return None

# Direct Open Food Facts search and reasoning

def _off_user_agent() -> str:
    app_name = os.getenv("OFF_APP_NAME", "WonderfulHealthAssistant")
    app_version = os.getenv("OFF_APP_VERSION", "0.1")
    contact = os.getenv("OFF_CONTACT_EMAIL", "contact@example.com")
    return f"{app_name}/{app_version} ({contact})"


def _off_search_top_one(query: str) -> Optional[dict]:
    try:
        base_url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 1,
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": _off_user_agent()})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        products = data.get("products") or []
        if not products:
            return None
        p = products[0]
        return {
            "product_name": p.get("product_name") or p.get("generic_name"),
            "brands": p.get("brands"),
            "nutri_score_grade": p.get("nutrition_grade_fr"),
            "nova_group": p.get("nova_group"),
            "nutriments": {
                "energy_kcal_100g": p.get("nutriments", {}).get("energy-kcal_100g"),
                "fat_100g": p.get("nutriments", {}).get("fat_100g"),
                "saturated_fat_100g": p.get("nutriments", {}).get("saturated-fat_100g"),
                "sugars_100g": p.get("nutriments", {}).get("sugars_100g"),
                "salt_100g": p.get("nutriments", {}).get("salt_100g"),
                "proteins_100g": p.get("nutriments", {}).get("proteins_100g"),
                "fiber_100g": p.get("nutriments", {}).get("fiber_100g"),
            },
            "additives_tags": p.get("additives_tags"),
            "ingredients_text": p.get("ingredients_text"),
            "url": p.get("url"),
            "image_url": p.get("image_url"),
            "code": p.get("code"),
        }
    except Exception:
        return None


def _reason_is_healthy(meal_description: str, question: str, llm_client: ChatGoogleGenerativeAI) -> str:
    """
    Analyzes the nutritional information of a meal and provides a comprehensive analysis of its healthiness.
    """
    instruction = (
        "You are a nutrition assistant. Based on the meal description provided, analyze if the meal is generally healthy. "
        "Explain your reasoning based on the ingredients, cooking method, and estimated nutritional breakdown. "
        "Provide actionable advice on how to make the meal healthier, if applicable."
    )
    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"{instruction}\n\nUser question: {question}"},
            {"type": "text", "text": "Meal description:"},
            {"type": "text", "text": meal_description},
        ]
    )
    res = llm_client.invoke([msg])
    return res.content if isinstance(res.content, str) else str(res.content)

# Set up the Streamlit page
st.set_page_config(page_title="Wonderful Health & Wellness AI Agent", layout="wide")
st.title("Wonderful Health & Wellness AI Chat Agent")

# Ensure a stable conversation id for persistence across reruns
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())

# Initialize the chat model
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
except Exception as e:
    st.error(f"Error initializing the language model: {e}")
    st.stop()

# Initialize a dedicated vision-capable model (fallback to llm if unavailable)
try:
    llm_vision = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
except Exception:
    llm_vision = llm

# Create the agent tools
tools = get_tavily_tools() + get_openfoodfacts_tools() + get_spoonacular_tools()

# Get the prompt from LangChain Hub
prompt_template = hub.pull("hwchase17/react-chat")

# Create the agent
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    if supabase_client:
        loaded_messages = _fetch_messages_from_supabase(st.session_state.conversation_id, limit=50)
        if not loaded_messages:
            greeting = "Hello! I'm your personal health and wellness assistant. How can I help you today?"
            _save_message_to_supabase(st.session_state.conversation_id, "assistant", greeting)
            loaded_messages = _fetch_messages_from_supabase(st.session_state.conversation_id, limit=50)
        if loaded_messages:
            st.session_state.messages = loaded_messages
        else:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your personal health and wellness assistant. How can I help you today?"
                }
            ]
    else:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your personal health and wellness assistant. How can I help you today?"
            }
        ]

# Optional uploader (non-blocking)
uploaded_image = st.file_uploader("Upload a meal/product image (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
if uploaded_image is not None:
    file_bytes = uploaded_image.getvalue()
    image_url = _upload_image_to_supabase(st.session_state.conversation_id, uploaded_image.name, file_bytes, uploaded_image.type)
    if image_url:
        # Append a user message with the image URL for context and display via normal chat loop
        user_image_msg = f"Uploaded image: ![]({image_url})"
        st.session_state.messages.append({"role": "user", "content": user_image_msg})
        _save_message_to_supabase(st.session_state.conversation_id, "user", f"Uploaded image: {image_url}")
        # Friendly acknowledgement without asking for a name
        assistant_ack = (
            "Image received. You can now ask questions like 'Is it healthy?' or 'What's its Nutri-Score?'."
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_ack})
        _save_message_to_supabase(st.session_state.conversation_id, "assistant", assistant_ack)
    else:
        st.error(f"Failed to upload image: {uploaded_image.name}")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        image_urls = _extract_image_urls(content)
        if image_urls:
            for url in image_urls:
                fresh_url = _ensure_accessible_image_url(url)
                if fresh_url:
                    content = content.replace(url, fresh_url)
        st.markdown(content)

# Chat input from the user
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    _save_message_to_supabase(st.session_state.conversation_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Intercept prompts lacking identifiers when an image was uploaded: run vision extraction automatically
        if _has_recent_uploaded_image_message():
            # Try to extract a product hint from the latest image using vision
            hint = None
            latest_url = _get_latest_image_url()
            if latest_url:
                hint = _extract_product_from_image_with_vision(latest_url, llm_vision, prompt)
            if hint:
                # Deterministic path: query OFF directly, then ask LLM to reason
                full_response = _reason_is_healthy(hint, prompt, llm)
                message_placeholder.markdown(full_response)
            else:
                full_response = "I received your image but couldn't read a product name or barcode. Please type the name or barcode so I can look it up."
                message_placeholder.markdown(full_response)
        else:
            # Prepare the chat history for the prompt (multimodal: include uploaded images)
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    image_urls = _extract_image_urls(content)
                    if image_urls:
                        parts = []
                        parts.append({"type": "text", "text": re.sub(r"!\[[^\]]*\]\([^)]*\)", "", content).strip() or "User uploaded an image."})
                        for u in image_urls:
                            fresh = _ensure_accessible_image_url(u)
                            if fresh:
                                parts.append({"type": "image_url", "image_url": {"url": fresh}})
                        chat_history.append(HumanMessage(content=parts))
                    else:
                        chat_history.append(HumanMessage(content=content))
                else:
                    chat_history.append(AIMessage(content=content))

            try:
                response = agent_executor.invoke({"chat_history": chat_history, "input": prompt})
                full_response = response["output"]
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
                full_response = "Sorry, I encountered an error. Please try again."
                message_placeholder.markdown(full_response)

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    _save_message_to_supabase(st.session_state.conversation_id, "assistant", full_response)

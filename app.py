# CONDA ENVIRONMENT: app_nasz_gpt
# IMPORTANT: Always run in app_nasz_gpt environment!

import os
import json
import io
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
from langfuse.openai import OpenAI  # Langfuse OpenAI wrapper dla automatycznego śledzenia
from dotenv import load_dotenv, dotenv_values
import PyPDF2
import docx
import time
import requests

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Aktualny cennik OpenAI API (05,07,2025)
MODEL_PRICING = {
    "gpt-5-nano": {
        "opis_modelu": "Klasyfikacja, podsumowania. Potrafi przetwarzać obrazy i tekst, ale nie jest dostępny w wersji audio.",
        "input_tokens": 0.05 / 1_000_000,   # $0.05 per 1M input tokens
        "output_tokens": 0.40 / 1_000_000,  # $0.40 per 1M output tokens
    },
    "gpt-5-mini": {
        "opis_modelu": "Workflow, szybkie zadania. Potrafi przetwarzać obrazy i tekst, ale nie jest dostępny w wersji audio.",
        "input_tokens": 0.25 / 1_000_000,  # $0.25 per 1M input tokens
        "output_tokens": 2.00 / 1_000_000, # $2.00 per 1M output tokens
    },
    "gpt-5": {
        "opis_modelu": "Programowanie, skomplikowane zadania. Potrafi przetwarzać obrazy i tekst, ale nie jest dostępny w wersji audio.",
        "input_tokens": 1.25 / 1_000_000,  # $1.25 per 1M input tokens
        "output_tokens": 10.00 / 1_000_000, # $10.00 per 1M output tokens
    }
}

# USD_TO_PLN = 3.6  # Aktualny kurs USD/PLN (05,07,2025)

DEFAULT_PERSONALITY = """
Jesteś pomocnym asystentem AI, który odpowiada na pytania użytkownika w sposób:
- Zwięzły i zrozumiały
- Merytoryczny i dokładny
- Przyjazny i profesjonalny
- Dostosowany do kontekstu rozmowy

Jeśli otrzymasz dokument do analizy, przeanalizuj go dokładnie i odpowiedz na pytania na jego podstawie.
""".strip()

DB_PATH = Path("db")
DB_CONVERSATIONS_PATH = DB_PATH / "conversations"

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
#
def load_environment():
    """Ładuje zmienne środowiskowe z pliku .env"""
    # Ustaw zmienne środowiskowe dla Langfuse (musi być przed innymi operacjami)
    load_dotenv()
    
    env = dotenv_values(".env")
    if not env.get("OPENAI_API_KEY"):
        st.error("❌ Brak klucza API OpenAI w pliku .env")
        st.stop()
    
    # Informacje o konfiguracji Langfuse (opcjonalne, nie blokuje działania)
    langfuse_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    missing_keys = [key for key in langfuse_keys if not env.get(key)]
    if missing_keys:
        st.warning(f"⚠️ Langfuse nie jest w pełni skonfigurowane - brakuje: {', '.join(missing_keys)}")
    
    return env

def extract_text_from_pdf(pdf_file) -> str:
    """Wyodrębnia tekst z pliku PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Błąd odczytu pliku PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file) -> str:
    """Wyodrębnia tekst z pliku DOCX"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Błąd odczytu pliku DOCX: {str(e)}")
        return ""

def extract_text_from_txt(txt_file) -> str:
    """Wyodrębnia tekst z pliku TXT"""
    try:
        return txt_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"❌ Błąd odczytu pliku TXT: {str(e)}")
        return ""

def process_uploaded_file(uploaded_file) -> Optional[str]:
    """Przetwarza przesłany plik i zwraca jego treść"""
    if not uploaded_file:
        return None
    
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    with st.spinner(f"📄 Przetwarzanie pliku {uploaded_file.name}..."):
        if file_extension == "pdf":
            return extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            return extract_text_from_docx(uploaded_file)
        elif file_extension == "txt":
            return extract_text_from_txt(uploaded_file)
        else:
            st.error(f"❌ Nieobsługiwany format pliku: {file_extension}")
            return None

def calculate_conversation_cost(messages: List[Dict], pricing: Dict) -> tuple[float, float]:
    """Oblicza koszt całej konwersacji oraz sumaryczny czas odpowiedzi (w sekundach)"""
    total_cost = 0.0
    total_time = 0.0
    for message in messages:
        if "usage" in message:
            usage = message["usage"]
            total_cost += (
                usage.get("prompt_tokens", 0) * pricing["input_tokens"] +
                usage.get("completion_tokens", 0) * pricing["output_tokens"]
            )
            total_time += usage.get("response_time", 0)
    # Zwracamy koszt oraz sumaryczny czas odpowiedzi
    return total_cost, total_time

@st.cache_data(show_spinner=False, ttl=3600)
def get_usd_to_pln_rate() -> tuple[float, Optional[str]]:
    """Pobiera aktualny kurs USD/PLN z NBP z cache (TTL=1h). Zwraca (kurs, data).
    W razie problemów zwraca (3.6, None) i wyświetla ostrzeżenie.
    """
    try:
        url = "https://api.nbp.pl/api/exchangerates/rates/A/USD/?format=json"
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"NBP HTTP {response.status_code}")
        data = response.json()
        rates = data.get("rates", [])
        if not rates or "mid" not in rates[0]:
            raise ValueError("Nieoczekiwana struktura odpowiedzi NBP")
        rate = float(rates[0]["mid"])
        rate_date = rates[0].get("effectiveDate")
        return rate, rate_date
    except Exception as e:
        st.warning(f"Nie udało się pobrać kursu USD/PLN z NBP: {e}")
        return 3.6, None  # wartość domyślna

# ==============================================================================
# CHATBOT FUNCTIONS
# ==============================================================================

def chatbot_reply(user_prompt: str, memory: List[Dict], file_content: Optional[str] = None) -> Dict:
    """Generuje odpowiedź chatbota"""
    # Przygotowanie wiadomości systemowej
    system_content = st.session_state.get("chatbot_personality", DEFAULT_PERSONALITY)
    
    # Jeśli jest plik, dodaj jego treść do systemu
    if file_content:
        system_content += f"\n\nOtrzymałeś również następujący dokument do analizy:\n\n{file_content}"
    
    messages = [{"role": "system", "content": system_content}]
    
    # Dodaj historię konwersacji
    for message in memory:
        messages.append({
            "role": message["role"], 
            "content": message["content"]
        })
    
    # Dodaj aktualną wiadomość użytkownika
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        import time
        start_time = time.time()
        with st.spinner("🤖 Generuję odpowiedź..."):
            response = st.session_state.openai_client.chat.completions.create(
                model=st.session_state.selected_model,
                messages=messages,
                temperature=1,
                max_completion_tokens=1000
            )
        elapsed = time.time() - start_time
        # Sprawdzenie użycia tokenów
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "response_time": elapsed,  # Dodaj czas odpowiedzi w sekundach
            }
        
        # Sprawdź czy odpowiedź została obcięta z powodu limitu tokenów
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        content = choice.message.content or ""
        
        # Jeśli odpowiedź została obcięta (finish_reason == "length"), dodaj informację
        if finish_reason == "length":
            content += "\n\n⚠️ **Odpowiedź została obcięta** - osiągnięto limit tokenów. Zwiększ `max_completion_tokens` aby uzyskać pełną odpowiedź."
        
        return {
            "role": "assistant",
            "content": content,
            "usage": usage,
        }
    
    except Exception as e:
        error_message = f"❌ Błąd API OpenAI: {str(e)}"
        st.session_state.error_message = error_message
        st.error(error_message)
        return {
            "role": "assistant",
            "content": f"Przepraszam, wystąpił błąd podczas generowania odpowiedzi.\n\n**Szczegóły błędu:**\n{str(e)}\n\nSpróbuj ponownie lub sprawdź ustawienia.",
            "usage": {},
        }

# ==============================================================================
# DATABASE FUNCTIONS
# ==============================================================================

def ensure_db_structure():
    """Zapewnia istnienie struktury bazy danych"""
    if not DB_PATH.exists():
        DB_PATH.mkdir(exist_ok=True)
        DB_CONVERSATIONS_PATH.mkdir(exist_ok=True)

def load_conversation_to_state(conversation: Dict):
    """Ładuje konwersację do stanu sesji"""
    st.session_state.update({
        "conversation_id": conversation["id"],
        "conversation_name": conversation["name"],
        "messages": conversation["messages"],
        "chatbot_personality": conversation["chatbot_personality"]
    })

def load_current_conversation():
    """Ładuje aktualną konwersację"""
    ensure_db_structure()
    
    current_file = DB_PATH / "current.json"
    
    if not current_file.exists():
        # Tworzenie pierwszej konwersacji
        conversation = {
            "id": 1,
            "name": "Pierwsza konwersacja",
            "chatbot_personality": DEFAULT_PERSONALITY,
            "messages": [],
        }
        
        # Zapisz konwersację
        with open(DB_CONVERSATIONS_PATH / "1.json", "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        # Zapisz jako aktualną
        with open(current_file, "w", encoding="utf-8") as f:
            json.dump({"current_conversation_id": 1}, f, indent=2)
    
    else:
        # Wczytaj ID aktualnej konwersacji
        with open(current_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            conversation_id = data["current_conversation_id"]
        
        # Wczytaj konwersację
        conversation_file = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
        if conversation_file.exists():
            with open(conversation_file, "r", encoding="utf-8") as f:
                conversation = json.load(f)
        else:
            # Jeśli plik nie istnieje, stwórz nową konwersację
            conversation = {
                "id": conversation_id,
                "name": f"Konwersacja {conversation_id}",
                "chatbot_personality": DEFAULT_PERSONALITY,
                "messages": [],
            }
    
    load_conversation_to_state(conversation)

def save_conversation():
    """Zapisuje aktualną konwersację"""
    conversation_id = st.session_state.get("conversation_id", 1)
    conversation = {
        "id": conversation_id,
        "name": st.session_state.get("conversation_name", f"Konwersacja {conversation_id}"),
        "chatbot_personality": st.session_state.get("chatbot_personality", DEFAULT_PERSONALITY),
        "messages": st.session_state.get("messages", []),
    }
    
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

def create_new_conversation():
    """Tworzy nową konwersację"""
    ensure_db_structure()
    
    # Znajdź następne ID
    conversation_ids = []
    for file_path in DB_CONVERSATIONS_PATH.glob("*.json"):
        try:
            conversation_ids.append(int(file_path.stem))
        except ValueError:
            continue
    
    new_id = max(conversation_ids, default=0) + 1
    
    # Ustaw ZAWSZE domyślną osobowość
    conversation = {
        "id": new_id,
        "name": f"Konwersacja {new_id}",
        "chatbot_personality": DEFAULT_PERSONALITY,
        "messages": [],
    }
    
    # Zapisz nową konwersację
    with open(DB_CONVERSATIONS_PATH / f"{new_id}.json", "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    
    # Ustaw jako aktualną
    with open(DB_PATH / "current.json", "w", encoding="utf-8") as f:
        json.dump({"current_conversation_id": new_id}, f, indent=2)
    
    load_conversation_to_state(conversation)
    st.session_state["chatbot_personality"] = DEFAULT_PERSONALITY  # Ustaw w session_state
    st.rerun()

def switch_conversation(conversation_id: int):
    """Przełącza na wybraną konwersację"""
    conversation_file = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
    
    if conversation_file.exists():
        with open(conversation_file, "r", encoding="utf-8") as f:
            conversation = json.load(f)
        
        # Ustaw jako aktualną
        with open(DB_PATH / "current.json", "w", encoding="utf-8") as f:
            json.dump({"current_conversation_id": conversation_id}, f, indent=2)
        
        load_conversation_to_state(conversation)
        st.rerun()

def list_conversations() -> List[Dict]:
    """Zwraca listę wszystkich konwersacji"""
    conversations = []
    
    for file_path in DB_CONVERSATIONS_PATH.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                conversation = json.load(f)
                conversations.append({
                    "id": conversation["id"],
                    "name": conversation["name"],
                    "message_count": len(conversation.get("messages", [])),
                })
        except (json.JSONDecodeError, KeyError):
            continue
    
    return sorted(conversations, key=lambda x: x["id"], reverse=True)

def delete_conversation(conversation_id: int):
    """Usuwa konwersację i przełącza na inną, jeśli to była aktualna"""
    conversation_file = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
    if conversation_file.exists():
        conversation_file.unlink()
        st.success(f"✅ Usunięto konwersację {conversation_id}")

    # Jeśli usunięto aktualną konwersację, przełącz na inną lub utwórz nową
    if conversation_id == st.session_state.get("conversation_id"):
        remaining = list_conversations()
        if remaining:
            # Przełącz na najnowszą pozostałą
            switch_conversation(remaining[0]["id"])
        else:
            # Utwórz nową, jeśli nie ma
            create_new_conversation()

# ==============================================================================
# UI FUNCTIONS
# ==============================================================================

def render_sidebar():
    """Renderuje sidebar z ustawieniami"""
    with st.sidebar:
        st.image("background/logo.png", use_container_width=True)
    
        st.markdown("## ⚙️ Ustawienia")


        # Osobowość chatbota
        st.session_state.chatbot_personality = st.text_area(
            "🎭 Osobowość chatbota" ,
            help="Opisz jak ma zachowywać się chatbot",
            value=st.session_state.get("chatbot_personality", DEFAULT_PERSONALITY),
            height=150,
            
        )

        # Wybór modelu
        st.session_state.selected_model = st.selectbox(
            "🧠  Model AI",
            options=list(MODEL_PRICING.keys()),
            index=0,
            help="Wybierz model OpenAI do wykorzystania"
        )
        
        # Informacje o cenach
        pricing = MODEL_PRICING[st.session_state.selected_model]
        st.markdown("### Parametry wybranego modelu modelu ")
        st.markdown(f"""
        **MODEL:** {st.session_state.selected_model}
        **OPIS:** {pricing['opis_modelu']}
        - **Input:** ${pricing['input_tokens']*1_000_000:.2f} / 1M tokenów
        - **Output:** ${pricing['output_tokens']*1_000_000:.2f} / 1M tokenów
        """)

        # Kurs walutowy
        usd_to_pln = st.number_input(
            "💱 Aktualny Kurs USD/PLN wg NBP",
            min_value=0.0,
            value=float(USD_TO_PLN),
            step=0.01,
            format="%.2f",
            key="usd_to_pln"
        )
        
        rate_date = st.session_state.get("usd_to_pln_date")
        if rate_date:
            st.caption(f"Kurs NBP z dnia: {rate_date}")
        else:
            st.caption("Użyto domyślnego kursu (pobranie z NBP nie powiodło się)")
        
        # Koszt aktualnej konwersacji
        messages = st.session_state.get("messages", [])
        total_cost, total_time = calculate_conversation_cost(messages, pricing)
        
        st.markdown("### 💳 Koszt konwersacji")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<span style='font-size: 0.8em;'>Koszt w USD</span><br><span style='font-size: 1.5em; font-weight: bold;'>{total_cost:.4f}</span>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<span style='font-size: 0.8em;'>Koszt w PLN</span><br><span style='font-size: 1.5em; font-weight: bold;'>{total_cost * usd_to_pln:.2f}</span>",
                unsafe_allow_html=True,
            )
        # Statystyki tokenów i czasu
        if messages:
            total_tokens = sum(msg.get("usage", {}).get("total_tokens", 0) for msg in messages)
            st.markdown(f"**Tokeny razem:** {total_tokens:,}")
            st.markdown(f"**Czas odpowiedzi razem:** {total_time:.2f} s")

def render_conversation_manager():
    """Renderuje zarządzanie konwersacjami"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("## 💬 Konwersacje")
        
        # Nazwa aktualnej konwersacji
        new_name = st.text_input(
            "📝 Nazwa konwersacji",
            value=st.session_state.get("conversation_name", ""),
            key="name_input"
        )
        
        if new_name != st.session_state.get("conversation_name"):
            st.session_state.conversation_name = new_name
            save_conversation()
        
        # Przycisk nowej konwersacji
        if st.button("➕ Nowa konwersacja", use_container_width=True):
            create_new_conversation()
        
        # Przycisk usuwania aktualnej konwersacji
        current_id = st.session_state.get("conversation_id", 1)
        if st.button("🗑️ Usuń aktualną konwersację", use_container_width=True):
            delete_conversation(current_id)
        
        # Lista konwersacji jako selectbox
        conversations = list_conversations()
        st.markdown("### 📋 Historia konwersacji")

        # Przygotuj listę nazw do selectboxa
        conversation_names = [f"{conv['name']} (id: {conv['id']}, msg: {conv['message_count']})" for conv in conversations[:10]]
        conversation_ids = [conv['id'] for conv in conversations[:10]]

        if conversation_names:
            selected_idx = conversation_ids.index(current_id) if current_id in conversation_ids else 0
            selected_conv = st.selectbox(
                "Wybierz konwersację",
                options=conversation_names,
                index=selected_idx
            )
            # Po wyborze znajdź id i przełącz, jeśli inna niż aktualna
            selected_id = conversation_ids[conversation_names.index(selected_conv)]
            if selected_id != current_id:
                switch_conversation(selected_id)  # Przełącz na wybraną konwersację
        
def render_main_chat():
    """Renderuje główny interfejs czatu"""
    # Wyświetl błąd jeśli istnieje
    if "error_message" in st.session_state:
        st.error(st.session_state.error_message)
        if st.button("❌ Zamknij błąd"):
            del st.session_state.error_message
            st.rerun()
    
    # pierwsza linijka w kolorze gradientu  wyśrodkowany
    st.markdown("<h1 style='text-align: center; font-size: 5em; background: linear-gradient(to right, #6a11cb 25%, #2575fc 75%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 4px 10px #2575fc25, 0 4px 20px rgba(37, 117, 252, 0.35);'>🤖 Twój czat GPT</h1>", unsafe_allow_html=True)
    # druga linijka pod spodem w kolorze niebieskim
    st.markdown("<h2 style='color: #2575fc; text-align: center; font-size: 1em;'>Inteligentny asystent AI z możliwością wyboru modelu, wcielający się w wybraną osobowość. <br>Aplikacja pozwala dodawać załączniki i analizować na bieżąco koszt użycia zgodnie z aktualnym kursem USD</h2>", unsafe_allow_html=True)

    # Pobierz plik z session state jeśli istnieje
    file_content = st.session_state.get("uploaded_file_content", None)
    
    # Wyświetl historię konwersacji
    messages = st.session_state.get("messages", [])
    
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Pokaż statystyki dla wiadomości asystenta
            if message["role"] == "assistant" and "usage" in message:
                usage = message["usage"]
                if usage:
                    st.caption(
                        f"Tokeny: {usage.get('total_tokens', 0)} | "
                        f"Input: {usage.get('prompt_tokens', 0)} | "
                        f"Output: {usage.get('completion_tokens', 0)} | "
                        f"Czas odpowiedzi: {usage.get('response_time', 0):.2f}s"
                    )
    
    # Kontener dla dolnego paska - zostanie na miejscu podczas przewijania
    bottom_container = st.container()
    with bottom_container:
        # Sekcja upload pliku i chat input w jednym rzędzie
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Upload pliku
            uploaded_file = st.file_uploader(
                "📁 Załącznik",
                type=["txt", "pdf", "docx"],
                help="Prześlij dokument do analizy",
                key="file_uploader"
            )
            
            # Przetwórz plik jeśli został przesłany
            if uploaded_file:
                file_content = process_uploaded_file(uploaded_file)
                if file_content:
                    st.session_state.uploaded_file_content = file_content
                    st.success(f"✅ {uploaded_file.name}")
                    with st.expander("📄 Podgląd"):
                        st.text_area("", file_content[:500] + "..." if len(file_content) > 500 else file_content, height=100, key="preview")
        
        with col2:
            # Input dla nowej wiadomości
            user_input = st.chat_input("Zadaj pytanie...")
    
    if user_input:
        # Dodaj wiadomość użytkownika
        with st.chat_message("user"):
            st.markdown(user_input)
        
        messages.append({"role": "user", "content": user_input})
        
        # Generuj odpowiedź
        with st.chat_message("assistant"):
            response = chatbot_reply(
                user_input, 
                memory=messages[-20:],  # Ostatnie 20 wiadomości jako kontekst
                file_content=st.session_state.get("uploaded_file_content", None)
            )
            st.markdown(response["content"])
            
            # Pokaż statystyki
            if response["usage"]:
                usage = response["usage"]
                st.caption(
                    f"Tokeny: {usage.get('total_tokens', 0)} | "
                    f"Input: {usage.get('prompt_tokens', 0)} | "
                    f"Output: {usage.get('completion_tokens', 0)} | "
                    f"Czas odpowiedzi: {usage.get('response_time', 0):.2f}s"
                )
        
        # Dodaj odpowiedź do historii
        messages.append(response)
        st.session_state.messages = messages
        
        # Zapisz konwersację
        save_conversation()
        st.rerun()

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Główna funkcja aplikacji"""
    # Konfiguracja strony
    st.set_page_config(
        page_title="Twój czat GPT",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Pobierz aktualny kurs USD/PLN (z cache) i zapisz datę kursu
    rate, rate_date = get_usd_to_pln_rate()
    global USD_TO_PLN
    USD_TO_PLN = rate
    st.session_state["usd_to_pln_date"] = rate_date


    # Dodaj CSS style
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stTextInput > div > div > input {
            background-color: #f0f2f6 !important;
            color: #262730 !important;
        }
        .stTextArea > div > div > textarea {
            background-color: #f0f2f6 !important;
            color: #262730 !important;
            border: 1px solid #d1d5db !important;
        }
        .stTextArea textarea {
            background-color: #f0f2f6 !important;
            color: #262730 !important;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .stButton > button {
            width: 100%;
            border-radius: 5px;
        }
        /* Dodatkowe style dla lepszej czytelności */
        .stTextArea label {
            color: #262730 !important;
            font-weight: 600;
        }
        .stSelectbox label {
            color: #262730 !important;
            font-weight: 600;
        }
        .stTextInput label {
            color: #262730 !important;
            font-weight: 600;
        }
        /* Style dla dolnego paska */
        .bottom-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 1rem;
            border-top: 1px solid #e1e5e9;
            z-index: 1000;
        }
        /* Dodaj miejsce na dole aby chat nie był zakryty */
        .main .block-container {
            padding-bottom: 120px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Inicjalizacja
    env = load_environment()
    st.session_state.openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])

    # Wczytaj konwersację
    if "conversation_id" not in st.session_state:
        load_current_conversation()

    # Renderuj interfejs
    render_sidebar()
    render_conversation_manager()
    render_main_chat()

if __name__ == "__main__":
    main()

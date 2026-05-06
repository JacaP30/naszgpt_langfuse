# CONDA ENVIRONMENT: app_nasz_gpt
# IMPORTANT: Always run in app_nasz_gpt environment!

import os
import json
import io
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
from openai import OpenAI as OpenAINative  # walidacja klucza bez Langfuse
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
    """Ładuje zmienne środowiskowe z pliku .env lub ze zmiennych środowiskowych"""
    # Ustaw zmienne środowiskowe dla Langfuse (musi być przed innymi operacjami)
    load_dotenv()
    
    # Najpierw sprawdź zmienne środowiskowe (działa na Streamlit Cloud)
    env = dict(os.environ)
    
    # Jeśli istnieje plik .env, nadpisz zmienne środowiskowe wartościami z pliku
    if Path(".env").exists():
        file_env = dotenv_values(".env")
        env.update(file_env)
    
    # Brak OPENAI_API_KEY nie przerywa — użytkownik może podać klucz na starcie lub wybrać tryb demo
    
    # Informacje o konfiguracji Langfuse (opcjonalne, nie blokuje działania)
    langfuse_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    missing_keys = [key for key in langfuse_keys if not env.get(key)]
    if missing_keys:
        st.warning(f"⚠️ Langfuse nie jest w pełni skonfigurowane - brakuje: {', '.join(missing_keys)}")
    
    return env


def validate_openai_api_key(api_key: str) -> tuple[bool, str]:
    """Sprawdza klucz przez lekkie wywołanie API (lista modeli). Zwraca (sukces, komunikat błędu)."""
    key = (api_key or "").strip()
    if not key:
        return False, "Podaj niepusty klucz."

    try:
        client = OpenAINative(api_key=key)
        client.models.list()
        return True, ""
    except Exception as e:
        msg = str(e).strip()
        lower = msg.lower()
        if (
            "401" in msg
            or "incorrect api key" in lower
            or "invalid api key" in lower
            or "authentication" in lower
            or "invalid_request_error" in lower and "key" in lower
        ):
            return False, "Klucz został odrzucony przez OpenAI — sprawdź, czy jest poprawny i aktywny."
        return False, msg or "Nie udało się zweryfikować klucza."


def get_raw_api_key(env: dict) -> str:
    """Klucz z pola użytkownika ma pierwszeństwo; niepoprawny klucz z .env jest ignorowany do czasu poprawy."""
    user_key = (st.session_state.get("user_api_key") or "").strip()
    if user_key:
        return user_key
    env_key = (env.get("OPENAI_API_KEY") or "").strip()
    ignored = (st.session_state.get("ignored_invalid_env_key") or "").strip()
    if env_key and ignored and env_key == ignored:
        return ""
    return env_key


def validate_openai_credentials(env: dict) -> None:
    """Przy pierwszym żądaniu weryfikuje dostępny klucz (env lub sesja). Ustawia stan przy sukcesie lub odrzuceniu."""
    if st.session_state.get("demo_mode"):
        return

    key = get_raw_api_key(env)
    if not key:
        st.session_state.pop("validated_openai_key", None)
        return

    if st.session_state.get("validated_openai_key") == key:
        return

    if st.session_state.get("rejected_api_key") == key:
        return

    with st.spinner("Sprawdzanie klucza OpenAI..."):
        ok, err_msg = validate_openai_api_key(key)

    if ok:
        st.session_state.validated_openai_key = key
        st.session_state.pop("rejected_api_key", None)
        st.session_state.pop("ignored_invalid_env_key", None)
        return

    st.session_state.rejected_api_key = key
    st.session_state.pop("validated_openai_key", None)

    user_key = (st.session_state.get("user_api_key") or "").strip()
    if user_key == key:
        st.session_state.pop("user_api_key", None)
    else:
        st.session_state.ignored_invalid_env_key = key

    st.error(f"Nieprawidłowy klucz API: {err_msg}")


def needs_startup_configuration(env: dict) -> bool:
    """True, gdy trzeba pokazać ekran startowy (brak działającego klucza lub tryb jeszcze nieustawiony)."""
    if st.session_state.get("demo_mode"):
        return False
    key = get_raw_api_key(env)
    if not key:
        return True
    return st.session_state.get("validated_openai_key") != key


def render_startup_configuration(env: dict) -> None:
    """Ekran startowy: klucz API lub tryb demo. Kończy wykonanie przez st.stop(), dopóki konfiguracja jest wymagana."""
    if not needs_startup_configuration(env):
        return

    st.markdown("## Konfiguracja dostępu")
    st.markdown(
        "Aby korzystać z czatu, podaj **klucz API OpenAI** "
        "(np. ze [strony OpenAI](https://platform.openai.com/api-keys)) "
        "lub wybierz **tryb demo**, aby obejrzeć interfejs bez wywołań API."
    )

    col_key, col_demo = st.columns(2)

    with col_key:
        st.markdown("##### Klucz API")
        api_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
            key="startup_api_key_field",
        )
        if st.button("Uruchom z kluczem", type="primary", use_container_width=True):
            key = (api_key_input or "").strip()
            if not key:
                st.warning("Wpisz klucz API albo wybierz tryb demo.")
            else:
                with st.spinner("Sprawdzanie klucza OpenAI..."):
                    ok, err_msg = validate_openai_api_key(key)
                if ok:
                    st.session_state.user_api_key = key
                    os.environ["OPENAI_API_KEY"] = key
                    st.session_state.demo_mode = False
                    st.session_state.validated_openai_key = key
                    st.session_state.pop("rejected_api_key", None)
                    st.session_state.pop("ignored_invalid_env_key", None)
                    st.rerun()
                else:
                    st.error(f"Klucz nie został zaakceptowany: {err_msg}")

    with col_demo:
        st.markdown("##### Tryb demo")
        st.caption(
            "Przeglądaj ustawienia, listę konwersacji i układ aplikacji. "
            "Czat oraz wywołania modelu są wyłączone."
        )
        if st.button("Wejdź w tryb demo", use_container_width=True):
            st.session_state.demo_mode = True
            st.session_state.pop("rejected_api_key", None)
            st.rerun()

    st.stop()


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
    if st.session_state.get("demo_mode") or not st.session_state.get("openai_client"):
        return {
            "role": "assistant",
            "content": (
                "**Tryb demo** — generowanie odpowiedzi wymaga klucza API OpenAI. "
                "Uruchom ponownie aplikację i wybierz „Uruchom z kluczem”, aby korzystać z czatu."
            ),
            "usage": {},
        }

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
                max_completion_tokens=5000
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

def exit_demo_to_configure_key() -> None:
    """Wyłącza tryb demo; przy następnym przebiegu pojawi się ekran wpisania klucza (jeśli nie ma ważnego klucza)."""
    st.session_state.demo_mode = False
    st.session_state.pop("rejected_api_key", None)
    st.rerun()


def render_sidebar():
    """Renderuje sidebar z ustawieniami"""
    with st.sidebar:
        if st.session_state.get("demo_mode"):
            st.info("Tryb demo — czat i wywołania API są wyłączone.")
            if st.button(
                "🔑 Wpisz klucz API",
                type="primary",
                use_container_width=True,
                key="demo_enter_key_sidebar",
            ):
                exit_demo_to_configure_key()

        # Wyświetl logo tylko jeśli plik istnieje
        logo_path = Path("background/logo.png")
        if logo_path.exists():
            try:
                st.image("background/logo.png", use_container_width=True)
            except TypeError:
                # Fallback dla starszych wersji Streamlit
                st.image("background/logo.png")
    
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

    # Tryb demo: wyraźny przycisk klucza nad tytułem (łatwo przeoczyć pod nagłówkiem strony)
    if st.session_state.get("demo_mode"):
        st.markdown(
            '<p style="margin-bottom:0.35rem;font-size:0.95rem;color:#555;">Tryb demo — aby pisać z modelem, dodaj klucz OpenAI:</p>',
            unsafe_allow_html=True,
        )
        if st.button(
            "🔑 Wpisz klucz API",
            type="primary",
            use_container_width=True,
            key="demo_enter_key_main_top",
        ):
            exit_demo_to_configure_key()
    
    # pierwsza linijka w kolorze gradientu  wyśrodkowany
    st.markdown("<h1 style='text-align: center; font-size: 5em; background: linear-gradient(to right, #6a11cb 25%, #2575fc 75%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 4px 10px #2575fc25, 0 4px 20px rgba(37, 117, 252, 0.35);'>🤖 Twój czat GPT</h1>", unsafe_allow_html=True)
    # druga linijka pod spodem w kolorze niebieskim
    st.markdown("<h2 style='color: #2575fc; text-align: center; font-size: 1em;'>Inteligentny asystent AI z możliwością wyboru modelu, wcielający się w wybraną osobowość. <br>Aplikacja pozwala dodawać załączniki i analizować na bieżąco koszt użycia zgodnie z aktualnym kursem USD</h2>", unsafe_allow_html=True)

    if st.session_state.get("demo_mode"):
        st.warning(
            "Tryb demo — przeglądasz interfejs bez wywołań OpenAI. Pole czatu jest wyłączone; "
            "możesz sprawdzać panel boczny, koszty (na zapisanych wiadomościach) i listę konwersacji."
        )

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
            demo = st.session_state.get("demo_mode", False)
            user_input = st.chat_input(
                "Tryb demo — czat wyłączony" if demo else "Zadaj pytanie...",
                disabled=demo,
            )
    
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
        initial_sidebar_state="expanded",
        menu_items={
            'Get help': None,
            'Report a bug': None,
            'About': "## Twój czat GPT\n\nAplikacja chat GPT zintegrowana z Langfuse do monitorowania wywołań OpenAI."
        }
    
    )

    HIDE_STREAMLIT_STYLE = """
        <style>
        /* Menu ⋮, stopka, toolbar — bez ukrywania całego header:
           w nagłówku jest przycisk rozwijania/zwijania sidebara; display:none na header
           sprawiał, że po zwinięciu panelu nie dało się go przywrócić. */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
        /* Nie ukrywaj stBaseButton-header — to często przycisk „Open sidebar” po zwinięciu */
        .stDeployButton, [data-testid="stDecoration"] {display: none !important;}

        /* PRAWY DOLNY RÓG — Manage app (Cloud) */
        button[aria-label="Manage app"] {display: none !important;}
        a[aria-label="Manage app"] {display: none !important;}
        [data-testid="manageAppButton"] {display: none !important;}
        [data-testid="stCloudManageApp"] {display: none !important;}
        /* Fallback (gdyby a11y/aria się zmieniło) */
        div[role="complementary"] [title="Manage app"] {display: none !important;}
        </style>
    """
    st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

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
        /* Pełna szerokość tylko w treści i sidebarze — nie w nagłówku Streamlit (menu panelu) */
        section[data-testid="stSidebar"] .stButton > button,
        .main .block-container .stButton > button {
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
    validate_openai_credentials(env)
    render_startup_configuration(env)

    if st.session_state.get("demo_mode"):
        st.session_state.openai_client = None
    else:
        api_key = get_raw_api_key(env)
        st.session_state.openai_client = OpenAI(api_key=api_key)

    # Wczytaj konwersację
    if "conversation_id" not in st.session_state:
        load_current_conversation()

    # Renderuj interfejs
    render_sidebar()
    render_conversation_manager()
    render_main_chat()

if __name__ == "__main__":
    main()

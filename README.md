# NaszGPT z Langfuse

Aplikacja chat GPT zintegrowana z Langfuse do monitorowania wywołań OpenAI.

## Wymagane środowisko

**CONDA ENVIRONMENT: `app_nasz_gpt`**

## Instalacja

1. Aktywuj środowisko Conda:
```bash
conda activate app_nasz_gpt
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

Lub utwórz środowisko z pliku:
```bash
conda env create -f environment.yml
conda activate app_nasz_gpt
```

## Konfiguracja

Utwórz plik `.env` z następującymi zmiennymi:

```
OPENAI_API_KEY=sk-proj-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Uruchomienie

Pamiętaj, aby **ZAWSZE** aktywować środowisko `app_nasz_gpt` przed uruchomieniem:

```bash
conda activate app_nasz_gpt
streamlit run app.py
```

## Funkcje

- Chat GPT z wyborem modelu
- Integracja z Langfuse dla śledzenia wywołań
- Analiza kosztów zgodnie z kursem NBP
- Zarządzanie konwersacjami
- Wsparcie dla załączników (PDF, DOCX, TXT)


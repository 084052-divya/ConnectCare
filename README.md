# ConnectCare: Telecom Grievance Redressal Chatbot

AI chatbot for **Nimbus Telecom** (a fictional Indian mobile operator) built on Streamlit + Google Gemini (free API key).
Use case #1 from the project menu: *Grievance redressal bot for a bank or telecom*.

**Features:** intent classification (Billing / Technical / Other + status, info, human, off-topic), multi-turn slot filling,
ticket ID generation (`NT-YYMMDD-XXXX`) with SLA dates, status lookup, outage linking, escalation to a human
(on request, fraud, legal threats, anger, disputes above Rs 5,000, or two failed clarifications), prompt-injection
blocking, OTP/card/Aadhaar redaction, answer grounding check, rule-based safe mode when the API fails,
agent console with ticket table, CSV export and intent analytics.

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # paste your Gemini key
streamlit run app.py
python -m pytest tests -q                                     # 17 offline tests, no key needed
```

## Deploy free (shareable link)
1. Get a free key at https://aistudio.google.com/app/apikey
2. Push this folder to a **public GitHub repo** (secrets.toml is git-ignored, never commit your key).
3. Go to https://share.streamlit.io > Create app > pick the repo, branch `main`, file `app.py`.
4. Advanced settings > Secrets: paste `GEMINI_API_KEY = "your-key"` > Deploy.
5. Share the `https://<your-app>.streamlit.app` link.

If no key is set, the sidebar lets a visitor paste their own key, and the bot still works in *Safe mode* (rules only).

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: chat, agent console, how-it-works |
| `engine.py` | Controller: guardrails, extraction, business rules, ticket store |
| `llm.py` | Gemini prompt, JSON validation, model fallback chain, rule-based NLU |
| `data/` | Sample customers, tickets, outages and 12 policy snippets (all fictional) |
| `tests/` | Scripted conversation tests |

## Sample data to try
Mobiles: 9999000101 (Delhi, bill Rs 1847 on a 599 plan, area outage), 9999000102 (Bengaluru, open recharge ticket),
9999000107 (Hyderabad, bill Rs 7420), 9999000104 (Delhi prepaid, 0 GB data left).
Tickets: NT-260928-K7QP, NT-261002-R9TB, NT-260915-M3XD, NT-260920-Z4HN.

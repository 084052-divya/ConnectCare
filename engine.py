"""Conversation controller for ConnectCare.

Flow for every user message:
  1. Guardrails before the model: empty/long input, prompt-injection
     patterns, sensitive data redaction (OTP, Aadhaar, card numbers).
  2. Deterministic extraction: mobile number, ticket ID (regex).
  3. Context building: customer record, their tickets, area outages,
     top policy snippets from the knowledge base (lightweight RAG).
  4. NLU: Gemini (or rule-based fallback) -> intent, category, entities,
     sentiment, short grounded answer.
  5. Business rules decide the action: create ticket, look up status,
     escalate to human, ask a clarifying question.
  6. Output guardrail: numbers in the model's answer must exist in the
     context (no invented amounts/dates/IDs).
"""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm import rule_based_nlu

IST = timezone(timedelta(hours=5, minutes=30))
DATA = Path(__file__).parent / "data"
RUNTIME_TICKETS = DATA / "tickets_runtime.json"

SLA_HOURS = {"Billing": 72, "Technical": 48, "Other": 120}
HIGH_VALUE_DISPUTE = 5000
QUEUE_FOR = {"Billing": "Billing", "Technical": "Network operations", "Other": "Customer care"}

INJECTION_PATTERNS = [
    r"ignore (all |any |the |your )?(previous |prior |above )?(instructions|rules|prompt)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"system prompt", r"you are now", r"developer mode", r"jailbreak", r"\bDAN\b",
    r"pretend (to be|you are)", r"act as (a|an) (?!customer)", r"disregard (your|the|all)",
    r"new instructions", r"override", r"forget (your|all|previous)",
]
FRAUD_PATTERNS = [r"fraud", r"sim ?swap", r"unauthori[sz]ed", r"hack", r"scam", r"someone (else )?(took|stole|is using)",
                  r"stolen", r"money (was )?stolen", r"identity theft"]
LEGAL_PATTERNS = [r"consumer (court|forum)", r"lawyer", r"legal (notice|action)", r"\btrai\b", r"police", r"\bsue\b"]
YES = re.compile(r"^\s*(yes|y|yeah|yep|haan|ha|han|correct|right|sure|ok|okay|ji)\b", re.I)
NO = re.compile(r"^\s*(no|n|nope|nahi|nahin|wrong|not really)\b", re.I)


def now_ist() -> datetime:
    return datetime.now(IST)


def fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def mask_mobile(m: str | None) -> str:
    return f"{m[:2]}XXXXXX{m[-2:]}" if m and len(m) == 10 else "unknown"


# ---------------------------------------------------------------- data store
class Store:
    def __init__(self):
        if not RUNTIME_TICKETS.exists():
            self.reset()
        with open(DATA / "customers.csv", newline="", encoding="utf-8") as f:
            self.customers = {r["mobile"]: r for r in csv.DictReader(f)}
        with open(DATA / "outages.csv", newline="", encoding="utf-8") as f:
            self.outages = [r for r in csv.DictReader(f) if r["status"]]
        self.kb = json.loads((DATA / "knowledge_base.json").read_text(encoding="utf-8"))

    @staticmethod
    def reset():
        shutil.copy(DATA / "tickets_seed.json", RUNTIME_TICKETS)

    def tickets(self) -> list[dict]:
        try:
            return json.loads(RUNTIME_TICKETS.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.reset()
            return json.loads(RUNTIME_TICKETS.read_text(encoding="utf-8"))

    def save(self, tickets: list[dict]):
        tmp = RUNTIME_TICKETS.with_suffix(".tmp")
        tmp.write_text(json.dumps(tickets, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(RUNTIME_TICKETS)

    def get_ticket(self, tid: str) -> dict | None:
        return next((t for t in self.tickets() if t["ticket_id"] == tid), None)

    def tickets_for(self, mobile: str) -> list[dict]:
        return [t for t in self.tickets() if t["mobile"] == mobile]

    def new_ticket_id(self) -> str:
        existing = {t["ticket_id"] for t in self.tickets()}
        alphabet = "".join(c for c in string.ascii_uppercase + string.digits if c not in "O0I1L")
        while True:
            tid = f"NT-{now_ist():%y%m%d}-{''.join(random.choices(alphabet, k=4))}"
            if tid not in existing:
                return tid

    def create_ticket(self, **fields) -> dict:
        tickets = self.tickets()
        created = now_ist()
        t = {
            "ticket_id": self.new_ticket_id(),
            "created_at": fmt(created),
            "sla_due": fmt(created + timedelta(hours=SLA_HOURS.get(fields["category"], 120))),
            "updates": [],
            "channel": "Chatbot",
            **fields,
        }
        tickets.append(t)
        self.save(tickets)
        return t

    def add_update(self, tid: str, note: str):
        tickets = self.tickets()
        for t in tickets:
            if t["ticket_id"] == tid:
                t["updates"].append({"at": fmt(now_ist()), "note": note})
        self.save(tickets)

    def outage_for(self, city: str, area: str) -> dict | None:
        for o in self.outages:
            if o["city"].lower() == (city or "").lower() and o["area"].lower() in (area or "").lower() \
                    and o["issue"] != "No outage":
                return o
        return None

    def retrieve(self, text: str, k: int = 2) -> list[dict]:
        """Keyword-overlap retrieval over the policy knowledge base."""
        t = text.lower()
        tokens = set(re.findall(r"[a-z0-9]+", t))
        scored = []
        for doc in self.kb:
            score = sum(2 for kw in doc["keywords"] if kw in t)
            score += len(tokens & set(re.findall(r"[a-z]+", doc["title"].lower())))
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:k]]


# ---------------------------------------------------------------- helpers
def redact(text: str) -> tuple[str, list[str]]:
    """Remove secrets before anything leaves the app."""
    found = []
    if re.search(r"\b(otp|pin|password|cvv)\b\D{0,15}\d{3,8}", text, re.I):
        text = re.sub(r"(\b(?:otp|pin|password|cvv)\b\D{0,15})\d{3,8}", r"\1[REDACTED]", text, flags=re.I)
        found.append("OTP/PIN")
    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b", text):
        text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b", "[CARD REDACTED]", text)
        found.append("card number")
    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text):
        text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[AADHAAR REDACTED]", text)
        found.append("Aadhaar-like number")
    return text, found


def extract_mobile(text: str) -> tuple[str | None, bool]:
    """Returns (valid_mobile, looked_like_a_number_but_invalid).
    Accepts 9999000101, +91 9999000101, 09999000101, 99990 00101, 999-900-0101."""
    pats = [r"(?<![\d])(?:\+?91[\s-]?|0)?([6-9]\d{9})(?!\d)",
            r"(?<![\d])(?:\+?91[\s-]?)?([6-9]\d{4})[\s-](\d{5})(?!\d)",
            r"(?<![\d])(?:\+?91[\s-]?)?([6-9]\d{2})[\s-](\d{3})[\s-](\d{4})(?!\d)"]
    for p in pats:
        m = re.search(p, text)
        if m:
            return "".join(m.groups()), False
    return None, bool(re.search(r"(?<!\d)\d{8,12}(?!\d)", text))


def extract_ticket(text: str) -> str | None:
    m = re.search(r"\bNT[\s\-]?(\d{6})[\s\-]?([A-Z0-9]{4})\b", text, re.I)
    return f"NT-{m.group(1)}-{m.group(2).upper()}" if m else None


def ungrounded_numbers(answer: str, grounding: str) -> list[str]:
    """Numbers (2+ digits) in the answer that do not appear anywhere in context."""
    g = grounding.replace(",", "")
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", answer)
    return [n for n in nums if len(n.replace(",", "").replace(".", "")) >= 2 and n.replace(",", "") not in g]


def matches(patterns, text) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def new_state() -> dict:
    return {"mobile": None, "customer_name": None, "pending_complaint": None, "awaiting": None,
            "fallback_count": 0, "escalated": False, "handoff_id": None}


# ---------------------------------------------------------------- the bot
class ConnectCareBot:
    def __init__(self, store: Store, gemini=None):
        self.store = store
        self.gemini = gemini

    # public entry point ---------------------------------------------------
    def handle(self, raw: str, state: dict, history: list[dict]) -> dict:
        meta = {"guardrails": [], "action": None, "nlu_source": None}
        msg = (raw or "").strip()
        if not msg:
            return self._reply("Please type your question or complaint.", meta)
        if len(msg) > 1200:
            msg = msg[:1200]
            meta["guardrails"].append("input truncated to 1200 chars")

        if matches(INJECTION_PATTERNS, msg):
            meta["guardrails"].append("prompt-injection pattern blocked before model call")
            meta.update(intent="out_of_scope", action="refused_adversarial")
            return self._reply(
                "I can't change how I work or share my internal instructions. I'm here only for Nimbus Telecom "
                "support: billing or network complaints, SIM and recharge issues, complaint status, or connecting "
                "you to an agent. What can I help you with?", meta)

        msg, secrets = redact(msg)
        warn = ""
        if secrets:
            meta["guardrails"].append(f"redacted {', '.join(secrets)}")
            warn = ("For your safety I removed the " + " and ".join(secrets) +
                    " you shared. Nimbus will never ask for OTPs, PINs, card or Aadhaar numbers in chat.\n\n")

        # deterministic entity extraction
        mobile, bad_number = extract_mobile(msg)
        ticket_id = extract_ticket(msg)
        if mobile:
            self._identify(mobile, state, meta)

        # quick yes/no handling for a pending confirmation
        if state.get("awaiting") == "confirm" and state.get("pending_complaint"):
            if YES.match(msg):
                state["awaiting"] = None
                state["pending_complaint"]["confirmed"] = True
                return self._continue_complaint(state, meta, prefix=warn)
            if NO.match(msg):
                state["pending_complaint"] = None
                state["awaiting"] = None
                meta["action"] = "complaint_discarded"
                return self._reply(warn + "No problem, I've discarded that. Could you tell me in your own words "
                                          "what the issue is?", meta)

        # NLU
        context = self._context(msg, state)
        nlu = self._nlu(msg, state, context, history, meta)
        if nlu["entities"].get("mobile") and not mobile:
            m2, _ = extract_mobile(str(nlu["entities"]["mobile"]))
            if m2:
                mobile = m2
                self._identify(m2, state, meta)
        ticket_id = ticket_id or extract_ticket(str(nlu["entities"].get("ticket_id") or ""))
        meta.update(intent=nlu["intent"], category=nlu["category"], sub_category=nlu["sub_category"],
                    confidence=nlu["confidence"], sentiment=nlu["sentiment"],
                    entities={**nlu["entities"], "mobile": mask_mobile(mobile) if mobile else None,
                              "ticket_id": ticket_id})
        empathy = (nlu["empathy_line"] + " ") if nlu["empathy_line"] else ""

        if bad_number and not mobile and state.get("awaiting") == "mobile":
            meta["action"] = "invalid_mobile"
            return self._reply(warn + "That doesn't look like a valid 10-digit Indian mobile number (it should "
                                      "start with 6, 7, 8 or 9). Could you re-enter it?", meta)
        if mobile and not meta.get("identified"):
            meta["action"] = "mobile_not_found"
            state["awaiting"] = "mobile"
            return self._reply(warn + f"I couldn't find {mask_mobile(mobile)} in Nimbus records. Please check "
                                      "the number. It should be the Nimbus number the issue is about.", meta)

        # ---------------- escalation rules (deterministic, checked first)
        reason = self._escalation_reason(msg, nlu, state)
        if reason:
            return self._escalate(reason, nlu, state, meta, prefix=warn + empathy)

        intent = nlu["intent"]
        # continuing a complaint: a bare number or details count as part of it
        continuing = bool(state.get("pending_complaint")) and intent in {"unclear", "complaint", "info_query"} \
            and (bool(mobile) or state.get("awaiting") in {"mobile", "details"})
        if continuing:
            intent = "complaint"

        if intent == "status_lookup" or (ticket_id and intent in {"unclear", "info_query"}):
            state["fallback_count"] = 0
            return self._status(ticket_id, state, meta, prefix=warn + empathy)

        if intent == "complaint":
            state["fallback_count"] = 0
            self._merge_complaint(nlu, msg, state)
            pc = state["pending_complaint"]
            if nlu["confidence"] < 0.55 and not continuing and not pc.get("confirmed") and pc.get("category"):
                state["awaiting"] = "confirm"
                meta["action"] = "confirm_low_confidence"
                return self._reply(warn + empathy + f"Just to confirm: is this a **{pc['category']}** issue "
                                   f"about \"{pc.get('summary') or msg[:120]}\"? (yes / no)", meta)
            return self._continue_complaint(state, meta, prefix=warn + empathy)

        if intent == "unclear":
            state["fallback_count"] += 1
            meta["action"] = f"clarify (fallback {state['fallback_count']}/2)"
            return self._reply(warn + (nlu["answer"] or "Sorry, I didn't understand. Is this about a bill, a "
                               "network problem, a complaint status, or talking to an agent?"), meta)

        # info / greeting / goodbye / out_of_scope -> grounded model answer
        state["fallback_count"] = 0
        answer = nlu["answer"] or "How can I help you with your Nimbus connection today?"
        bad = ungrounded_numbers(answer, json.dumps(context, ensure_ascii=False) + msg)
        if bad and intent == "info_query":
            meta["guardrails"].append(f"answer withheld: ungrounded numbers {bad}")
            answer = ("I couldn't verify that detail from your account records, so I won't guess. I can raise a "
                      "query for the billing team or connect you to an agent. Which would you prefer?")
        meta["action"] = f"answer_{intent}"
        if intent == "info_query" and context.get("policy_snippets"):
            meta["sources"] = [p["id"] + " " + p["title"] for p in context["policy_snippets"]]
        return self._reply(warn + answer, meta)

    # internals ------------------------------------------------------------
    def _reply(self, text: str, meta: dict, card: dict | None = None) -> dict:
        return {"text": text.strip(), "card": card, "meta": meta}

    def _identify(self, mobile: str, state: dict, meta: dict):
        cust = self.store.customers.get(mobile)
        if cust:
            state["mobile"], state["customer_name"] = mobile, cust["name"]
            if state.get("awaiting") == "mobile":
                state["awaiting"] = None
        meta["identified"] = bool(cust)

    def _context(self, msg: str, state: dict) -> dict:
        ctx = {"policy_snippets": [{"id": d["id"], "title": d["title"], "text": d["text"]}
                                   for d in self.store.retrieve(msg)]}
        if state.get("mobile"):
            c = dict(self.store.customers[state["mobile"]])
            c["mobile"] = mask_mobile(c["mobile"])
            ctx["customer"] = {k: v for k, v in c.items() if v}
            ctx["their_tickets"] = [{k: t[k] for k in ("ticket_id", "category", "sub_category", "status",
                                                       "created_at", "sla_due")}
                                    for t in self.store.tickets_for(state["mobile"])][-5:]
            o = self.store.outage_for(c["city"], c["area"])
            if o:
                ctx["area_outage"] = o
        return ctx

    def _nlu(self, msg, state, context, history, meta) -> dict:
        safe_state = {k: state[k] for k in ("customer_name", "pending_complaint", "awaiting")}
        if self.gemini and self.gemini.available:
            try:
                nlu, model = self.gemini.analyse(msg, safe_state, context, history)
                meta["nlu_source"] = model
                return nlu
            except RuntimeError as e:
                meta["guardrails"].append("Gemini failed, rule-based fallback used")
                meta["llm_error"] = str(e)[:300]
        meta["nlu_source"] = "rule-based fallback"
        return rule_based_nlu(msg, context.get("policy_snippets", []), state)

    def _escalation_reason(self, msg: str, nlu: dict, state: dict) -> str | None:
        if state.get("escalated") and nlu["intent"] != "human_request":
            return None
        if matches(FRAUD_PATTERNS, msg):
            return "Possible fraud / SIM-swap (critical)"
        if nlu["intent"] == "human_request":
            return "Customer asked for a human agent"
        if matches(LEGAL_PATTERNS, msg):
            return "Legal or regulatory threat mentioned"
        if nlu["sentiment"] == "angry" and nlu["intent"] in {"complaint", "status_lookup", "unclear"}:
            return "Customer is very upset"
        amt = nlu["entities"].get("amount_inr") or 0
        if nlu["intent"] == "complaint" and nlu["category"] == "Billing" and amt > HIGH_VALUE_DISPUTE:
            return f"High-value billing dispute (Rs {amt:,.0f} > Rs {HIGH_VALUE_DISPUTE:,})"
        if nlu["intent"] == "unclear" and state.get("fallback_count", 0) >= 2:
            return "Bot could not understand after 2 clarification attempts"
        return None

    def _escalate(self, reason, nlu, state, meta, prefix="") -> dict:
        priority = "Critical" if "fraud" in reason.lower() else "High"
        pc = state.get("pending_complaint") or {}
        category = pc.get("category") or nlu["category"] or "Other"
        state["escalated"] = True
        meta["action"] = f"escalate_to_human: {reason}"
        summary = pc.get("summary") or nlu["entities"].get("issue_summary") or "Customer requested agent support."
        card = {"type": "escalation", "reason": reason, "priority": priority, "queue": QUEUE_FOR[category]}
        if state.get("mobile"):
            t = self.store.create_ticket(mobile=state["mobile"], category=category,
                                         sub_category=pc.get("sub_category") or nlu["sub_category"] or "Agent handoff",
                                         description=summary, priority=priority,
                                         status="Escalated - Human agent", escalated=True,
                                         escalation_reason=reason)
            state["handoff_id"] = t["ticket_id"]
            state["pending_complaint"], state["awaiting"] = None, None
            card.update(ticket_id=t["ticket_id"], sla_due=t["sla_due"])
            ref = f"Your reference is **{t['ticket_id']}**."
        else:
            hid = "ESC-" + "".join(random.choices(string.digits, k=6))
            state["handoff_id"] = hid
            card.update(ticket_id=hid)
            ref = f"Your handoff reference is **{hid}**. Please keep your registered mobile number ready."
        hour = now_ist().hour
        wait = "about 5 minutes (simulated)" if 8 <= hour < 22 else "a callback after 8 AM IST (agents are offline)"
        extra = (" If this involves fraud, please also call **198** right away to block your SIM, and report "
                 "money loss on the cybercrime helpline **1930**.") if priority == "Critical" else ""
        text = (f"{prefix}I'm handing this conversation to a human agent from our {card['queue']} team. {ref} "
                f"Expected wait: {wait}. The agent will see a summary of this chat, so you won't need to repeat "
                f"yourself.{extra}")
        return self._reply(text, meta, card)

    def _status(self, ticket_id, state, meta, prefix="") -> dict:
        if ticket_id:
            t = self.store.get_ticket(ticket_id)
            if not t:
                meta["action"] = "status_not_found"
                return self._reply(prefix + f"I couldn't find a complaint with ID **{ticket_id}**. Ticket IDs look "
                                            "like NT-260928-K7QP. You can also share your registered mobile number "
                                            "and I'll list your complaints.", meta)
            own = state.get("mobile") == t["mobile"]
            meta["action"] = "status_lookup"
            return self._reply(prefix + "Here's the latest on that complaint:", meta,
                               {"type": "status", "ticket": t, "full": own})
        if state.get("mobile"):
            ts = self.store.tickets_for(state["mobile"])
            meta["action"] = "status_list"
            if not ts:
                return self._reply(prefix + "You have no complaints on record. Would you like to raise one?", meta)
            return self._reply(prefix + f"I found {len(ts)} complaint(s) on your number:", meta,
                               {"type": "status_list", "tickets": ts[-5:]})
        meta["action"] = "ask_ticket_or_mobile"
        state["awaiting"] = None
        return self._reply(prefix + "Please share your ticket ID (for example NT-260928-K7QP) or your registered "
                                    "10-digit mobile number.", meta)

    def _merge_complaint(self, nlu, msg, state):
        pc = state.get("pending_complaint") or {"category": None, "sub_category": None, "summary": None,
                                                "amount_inr": None, "details": []}
        pc["category"] = pc["category"] or nlu["category"]
        pc["sub_category"] = pc["sub_category"] or nlu["sub_category"]
        if nlu["entities"].get("issue_summary"):
            pc["summary"] = nlu["entities"]["issue_summary"] if not pc["summary"] else pc["summary"]
        if nlu["entities"].get("amount_inr"):
            pc["amount_inr"] = nlu["entities"]["amount_inr"]
        bare_number = re.fullmatch(r"[\s\+\d\-]+", msg) is not None
        if not bare_number:
            pc["details"].append(msg[:300])
        state["pending_complaint"] = pc

    def _continue_complaint(self, state, meta, prefix="") -> dict:
        pc = state["pending_complaint"]
        if not pc.get("category"):
            state["awaiting"] = "details"
            meta["action"] = "ask_category"
            return self._reply(prefix + "Is this about your **bill/charges**, a **network or internet** problem, "
                                        "or something else like SIM or recharge?", meta)
        detail_words = len(" ".join(pc.get("details", [])).split())
        if not pc.get("summary") and detail_words < 5:
            state["awaiting"] = "details"
            meta["action"] = "ask_details"
            ask = {"Billing": "Which bill or charge is wrong, and by roughly how much?",
                   "Technical": "What exactly happens (calls dropping, no signal, slow data), where, and since when?",
                   "Other": "Could you describe what happened in a sentence or two?"}[pc["category"]]
            return self._reply(prefix + ask, meta)
        if not state.get("mobile"):
            state["awaiting"] = "mobile"
            meta["action"] = "ask_mobile"
            return self._reply(prefix + "To register this complaint, please share the 10-digit Nimbus mobile "
                                        "number it's about. (Never share OTPs or passwords.)", meta)

        # duplicate check: same number + category open in last 24h
        cutoff = fmt(datetime.now(IST) - timedelta(hours=24))
        dup = next((t for t in reversed(self.store.tickets_for(state["mobile"]))
                    if t["category"] == pc["category"] and t["created_at"] >= cutoff
                    and t["status"] not in {"Resolved", "Closed"}), None)
        summary = pc.get("summary") or " ".join(pc["details"])[:250]
        if dup:
            self.store.add_update(dup["ticket_id"], f"Customer added via chat: {summary}")
            state["pending_complaint"], state["awaiting"] = None, None
            meta["action"] = "duplicate_merged"
            return self._reply(prefix + f"You already have an open {pc['category']} complaint "
                                        f"**{dup['ticket_id']}** from {dup['created_at']}. I've added these details "
                                        "to it rather than opening a duplicate.", meta,
                               {"type": "status", "ticket": self.store.get_ticket(dup["ticket_id"]), "full": True})

        cust = self.store.customers[state["mobile"]]
        outage = self.store.outage_for(cust["city"], cust["area"]) if pc["category"] == "Technical" else None
        priority = "High" if outage or (pc.get("amount_inr") or 0) > 2000 else "Medium"
        t = self.store.create_ticket(mobile=state["mobile"], category=pc["category"],
                                     sub_category=pc.get("sub_category") or "General",
                                     description=summary, priority=priority, status="Open", escalated=False,
                                     linked_outage=outage["issue"] if outage else None)
        state["pending_complaint"], state["awaiting"] = None, None
        meta["action"] = "ticket_created"
        note = ""
        if outage:
            note = (f"\n\nThere is a known issue in {outage['area']}: {outage['issue']}. Engineers expect it "
                    f"fixed by {outage['eta']} IST. I've linked your complaint to it.")
        return self._reply(prefix + f"I've registered your complaint, {cust['name'].split()[0]}." + note, meta,
                           {"type": "ticket", "ticket": t})

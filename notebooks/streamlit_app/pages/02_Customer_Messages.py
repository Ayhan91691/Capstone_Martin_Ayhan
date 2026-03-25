import streamlit as st
import sys
import os
import resend
import uuid
import json
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime

# =============================
# ENV
# =============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
resend.api_key = RESEND_API_KEY

# =============================
# PATH
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from utils.sentiment import predict_sentiment_batch
from utils.state import init_state

init_state()

# =============================
# 🤖 MULTI-TICKET AI
# =============================
def generate_tickets_from_message(msg):

    prompt = f"""
You are a senior product manager.

Analyze the customer message and split it into distinct issues.

Return a JSON ARRAY of tickets.

Each ticket must include:
- title (short, specific)
- priority (High, Medium, Low)
- summary (clear description)
- labels (max 3)
- subtasks (4-6 professional product tasks including technical + UX + validation)

IMPORTANT:
- If multiple problems exist → create MULTIPLE tickets
- Subtasks must be specific, actionable, and realistic
- No generic tasks like "Investigate"
- Write like a senior product owner

Customer message:
{msg['text']}

Return ONLY JSON ARRAY:
[
  {{
    "title": "...",
    "priority": "...",
    "summary": "...",
    "labels": ["...", "..."],
    "subtasks": ["...", "..."]
  }}
]
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        data = json.loads(response.choices[0].message.content)

    except:
        data = [{
            "title": msg["text"][:50],
            "priority": "Medium",
            "summary": msg["text"],
            "labels": ["general"],
            "subtasks": [
                "Analyze user issue in detail",
                "Identify root cause in system",
                "Design solution approach",
                "Implement fix and validate"
            ]
        }]

    tickets = []

    for item in data:
        tickets.append({
            "id": str(uuid.uuid4()),
            "title": item["title"],
            "description": item["summary"],
            "priority": item["priority"],
            "status": "Backlog",
            "customer_name": msg["name"],
            "labels": item.get("labels", []),
            "comments": [],
            "subtasks": [{"text": s, "done": False} for s in item["subtasks"]]
        })

    return tickets

# =============================
# 🤖 SUBJECT
# =============================
def generate_subject(msg):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"Write a short email subject (max 8 words): {msg['text']}"
            }]
        )

        subject = response.choices[0].message.content.strip()
        subject = subject.replace("\n", " ")
        return subject[:80]

    except:
        return "Support Response"

# =============================
# 🤖 EMAIL BODY
# =============================
def generate_ai_reply(msg):

    prompt = f"""
You are a professional support agent at MA Analytics.

Write ONLY the email body.

Rules:
- No greeting
- No name
- No subject
- No signature

Structure:
1. Appreciation
2. "Following action points we have taken:"
3. Bullet points
4. Closing text

Message:
{msg['text']}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.choices[0].message.content.strip()

        text = text.replace(msg["name"], "")
        text = text.replace("Hello", "")
        text = text.replace("Dear", "")

        text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")

        return text

    except:
        return "Thank you for your feedback. We are working on improvements."

# =============================
# ✉️ EMAIL TEMPLATE
# =============================
def send_email(to_email, subject, name, reply):

    subject = subject.replace("\n", " ")

    LOGO_URL = "https://raw.githubusercontent.com/Ayhan91691/Capstone_Martin_Ayhan/main/notebooks/streamlit_app/logo.jpeg"

    html = f"""
    <div style="font-family:Arial; max-width:600px; margin:auto; background:#f9fafb; border-radius:10px; overflow:hidden;">

        <div style="background:#0f172a; padding:20px;">
            <img src="{LOGO_URL}" style="height:50px;">
        </div>

        <div style="padding:30px; background:white;">
            <h2>Hello {name},</h2>

            <p style="line-height:1.6;">
                {reply}
            </p>

            <p style="margin-top:20px;">
                Best regards,<br>
                Customer Support Agent<br>
                <strong>MA Analytics</strong>
            </p>
        </div>

        <div style="padding:15px; font-size:12px; text-align:center; color:#777;">
            MA Analytics GmbH · Berlin · support@ma-analytics.ai
        </div>
    </div>
    """

    resend.Emails.send({
        "from": "MA Analytics <onboarding@resend.dev>",
        "to": to_email,
        "subject": subject,
        "html": html
    })

# =============================
# STATE
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tickets" not in st.session_state:
    st.session_state.tickets = []

if "selected_msg" not in st.session_state:
    st.session_state.selected_msg = None

# =============================
# UI
# =============================
st.title("📩 Customer Messages AI")

col1, col2 = st.columns([2, 1])

# =============================
# INPUT
# =============================
with col1:

    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")

    if st.button("Send Message"):

        if message:

            sentiment = predict_sentiment_batch([message])[0]

            msg = {
                "name": name or "Customer",
                "email": email,
                "text": message,
                "sentiment": sentiment,
                "time": datetime.now().strftime("%H:%M"),
                "date": datetime.now().strftime("%d.%m.%Y")
            }

            st.session_state.messages.append(msg)

            tickets = generate_tickets_from_message(msg)
            for t in tickets:
                st.session_state.tickets.append(t)

            st.success("✅ Message → AI Tickets created")
            st.rerun()

# =============================
# INBOX
# =============================
with col2:

    st.markdown("### 📥 Inbox")

    for i in range(len(st.session_state.messages) - 1, -1, -1):

        msg = st.session_state.messages[i]

        if st.button(
            f"{msg['name']} ({msg['time']}) — {msg['text'][:30]}",
            key=i
        ):
            st.session_state.selected_msg = i

# =============================
# DETAILS
# =============================
if st.session_state.selected_msg is not None:

    msg = st.session_state.messages[st.session_state.selected_msg]

    st.markdown("---")
    st.subheader("📨 Message Details")

    st.write(f"👤 {msg.get('name', 'N/A')}")
    st.write(f"📧 {msg.get('email', 'N/A')}")
    st.write(f"📅 {msg.get('date', 'N/A')}")
    st.write(f"⏰ {msg.get('time', 'N/A')}")

    if msg["sentiment"] == "Negative":
        st.error(f"🧠 Sentiment: {msg['sentiment']}")
    elif msg["sentiment"] == "Positive":
        st.success(f"🧠 Sentiment: {msg['sentiment']}")
    else:
        st.warning(f"🧠 Sentiment: {msg['sentiment']}")

    st.markdown("### Message")
    st.info(msg["text"])

    # AI
    subject = generate_subject(msg)
    reply = generate_ai_reply(msg)

    st.markdown("### ✉️ AI Reply")
    edited = st.text_area("Edit Reply", reply, height=200)

    if st.button("📤 Send Email"):

        send_email(
            msg["email"],
            subject,
            msg["name"],
            edited
        )

        st.success("🚀 Email sent!")
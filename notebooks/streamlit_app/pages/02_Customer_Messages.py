import streamlit as st
import pandas as pd
import sys
import os
import resend
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from datetime import datetime
from utils.sentiment import predict_sentiment_batch
from utils.state import init_state
init_state()

resend.api_key = "re_4k5WuXY6_Tsrdqb5f2t71mPJSi7exLw6n"

def send_email(to_email, subject, name, reply):

    import resend

    r = resend.Emails.send({
        "from": "MA Analytics <onboarding@resend.dev>",
        "to": to_email,
        "subject": subject,
        "html": f"""
<html>
<body style="margin:0; padding:0; background:#f3f4f6; font-family:Arial, sans-serif;">

<!-- PREVIEW TEXT (Inbox Vorschau) -->
<span style="display:none; max-height:0; overflow:hidden;">
Your request has been received – MA Analytics Support Team
</span>

<!-- CONTAINER -->
<table width="100%" cellspacing="0" cellpadding="0">
<tr>
<td align="center">

<!-- CARD -->
<table width="600" cellspacing="0" cellpadding="0" style="background:white; border-radius:12px; overflow:hidden;">

<!-- HEADER -->
<tr>
<td style="background:#111827; padding:20px 30px;">
    <table width="100%">
        <tr>
            <td align="left">
                <img src="https://raw.githubusercontent.com/Ayhan91691/Capstone_Martin_Ayhan/main/logo.jpeg"
                     alt="MA Analytics"
                     style="height:40px;">
            </td>
            <td align="right" style="color:#9ca3af; font-size:12px;">
                Customer Intelligence
            </td>
        </tr>
    </table>
</td>
</tr>

<!-- BODY -->
<tr>
<td style="padding:40px 30px;">

    <h2 style="margin:0 0 20px; font-size:22px; color:#111827;">
        Hello {name},
    </h2>

    <p style="
        font-size:15px;
        line-height:1.6;
        color:#374151;
        white-space:pre-line;
        margin:0 0 25px;
    ">
{reply}
    </p>

    <!-- BUTTON -->
    <a href="#" style="
        display:inline-block;
        padding:12px 20px;
        background:#2563eb;
        color:white;
        text-decoration:none;
        border-radius:8px;
        font-size:14px;
        font-weight:500;
    ">
        View your request
    </a>

</td>
</tr>

<!-- FOOTER -->
<tr>
<td style="background:#f9fafb; padding:25px; text-align:center;">

    <p style="margin:0; font-size:13px; color:#6b7280;">
        MA Analytics GmbH<br>
        Data Science & AI Solutions
    </p>

    <p style="margin:10px 0 0; font-size:12px; color:#9ca3af;">
        📍 Berlin, Germany<br>
        📧 support@ma-analytics.ai<br>
        🌐 www.ma-analytics.ai
    </p>

    <p style="margin-top:15px; font-size:11px; color:#d1d5db;">
        © 2026 MA Analytics — All rights reserved
    </p>

</td>
</tr>

</table>

</td>
</tr>
</table>

</body>
</html>
"""
    })

    return r

st.title("📩 Customer Messages AI")

# -------------------------------
# SESSION STATE (Speicher)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_msg" not in st.session_state:
    st.session_state.selected_msg = None

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns([2, 1])

# ===============================
# LEFT → FORM
# ===============================
with col1:
    st.header("📝 New Customer Message")

    name = st.text_input("First Name")
    email = st.text_input("Email")
    message = st.text_area("Message")

    if st.button("Send Message"):
        if message:

            sentiment = predict_sentiment_batch([message])[0]

            new_msg = {
                "name": name,
                "email": email,
                "text": message,
                "sentiment": sentiment,
                "time": datetime.now().strftime("%H:%M"),
                "read": False
            }

            st.session_state.messages.append(new_msg)

            st.success("✅ Message received!")
            st.rerun()

# ===============================
# RIGHT → INBOX
# ===============================
with col2:
    # 🔥 CLEAR BUTTON STATE
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
# BUTTON
    if st.button("🗑️ Clear Inbox"):
        st.session_state.confirm_clear = True
# CONFIRMATION
    if st.session_state.confirm_clear:
        st.warning("Are you sure you want to delete all messages?")

        col_yes, col_no = st.columns(2)

        with col_yes:
            if st.button("✅ Yes, delete all"):
                st.session_state.messages = []
                st.session_state.selected_msg = None
                st.session_state.confirm_clear = False
                st.rerun()

        with col_no:
            if st.button("❌ Cancel"):
                st.session_state.confirm_clear = False

## 📬 GMAIL STYLE INBOX
    st.markdown("### 📥 Inbox")

    for real_index in range(len(st.session_state.messages) - 1, -1, -1): 
        msg = st.session_state.messages[real_index]

        is_unread = not msg["read"]

        col_del, col_main = st.columns([1, 10])

        with col_del:
            if st.button("🗑️", key=f"delete_{real_index}"):
                st.session_state.messages.pop(real_index)
                if st.session_state.selected_msg == real_index:
                    st.session_state.selected_msg = None
                elif(
                    st.session_state.selected_msg is not None
                    and st.session_state.selected_msg > real_index
                ):
                    st.session_state.selected_msg -= 1
                st.rerun()
        
        with col_main:
            if st.button(
                f"{'🔴 ' if is_unread else ''}{msg['name']} — {msg['text'][:40]}...",
                key=f"msg_{real_index}"
            ):
                st.session_state.selected_msg = real_index
                st.session_state.messages[real_index]["read"] = True

# ===============================
# MESSAGE VIEW
# ===============================
if st.session_state.selected_msg is not None:

    if st.session_state.selected_msg >= len(st.session_state.messages):
        st.session_state.selected_msg = None
    else:
        msg = st.session_state.messages[st.session_state.selected_msg]

        st.markdown("---")
        st.subheader("📨 Message Details")

        st.write(f"**Name:** {msg['name']}")
        st.write(f"**Email:** {msg['email']}")
        st.write(f"**Time:** {msg['time']}")
        st.write(f"**Sentiment:** {msg['sentiment']}")

        st.markdown("### Message")
        st.info(msg["text"])

    # ---------------------------
    # AI INSIGHT
    # ---------------------------
    st.markdown("### 🧠 AI Insight")

    if msg["sentiment"] == "Negative":
        insight = "Customer is unhappy. Likely issue needs urgent attention."
        reply = f"""
        Dear {msg['name']},

        we are sorry for your experience.
        Our team is already working on improving this issue.

        Best regards  
        Support Team
        """
    else:
        insight = "Customer feedback is positive."
        reply = f"""
        Dear {msg['name']},

        thank you for your positive feedback!
        We are happy you enjoy our service.

        Best regards  
        Support Team
        """

    st.warning(insight)

    # ---------------------------
    # AUTO RESPONSE
    # ---------------------------
    st.markdown("### ✉️ Suggested Reply")

    edited_reply = st.text_area("Edit reply:", reply, height=200)

    if st.button("📤 Send Reply"):
        send_email(
            to_email=msg["email"],
            subject="Response to your feedback",
            name=msg["name"],
            reply=edited_reply
    )
    st.success("Reply sent!")
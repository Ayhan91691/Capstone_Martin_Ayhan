import streamlit as st
import uuid

st.set_page_config(layout="wide")

# =============================
# INIT
# =============================
if "tickets" not in st.session_state:
    st.session_state.tickets = []

if "selected_ticket" not in st.session_state:
    st.session_state.selected_ticket = None

# =============================
# HELPERS
# =============================
def move_ticket(ticket_id, new_status):
    for t in st.session_state.tickets:
        if t["id"] == ticket_id:
            t["status"] = new_status

def get_progress(ticket):
    subtasks = ticket.get("subtasks", [])
    if not subtasks:
        return 0
    done = sum(1 for s in subtasks if s["done"])
    return int((done / len(subtasks)) * 100)

def get_priority_color(priority):
    return {
        "High": "🔴",
        "Medium": "🟠",
        "Low": "🟢"
    }.get(priority, "⚪")

# =============================
# HEADER
# =============================
st.title("📊 Kanban Board")

col1, col2, col3, col4 = st.columns(4)

columns = {
    "Backlog": col1,
    "Todo": col2,
    "In Progress": col3,
    "Done": col4
}

# =============================
# BOARD
# =============================
for status, col in columns.items():
    with col:
        st.subheader(status)

        for t in st.session_state.tickets:
            if t["status"] != status:
                continue

            progress = get_progress(t)

            with st.container():
                st.markdown(f"### {t['title']}")
                st.caption(f"👤 {t['customer_name']}")

                st.write(f"{get_priority_color(t['priority'])} Priority: {t['priority']}")
                st.write(f"🏷️ Labels: {', '.join(t.get('labels', []))}")
                st.progress(progress / 100)

                c1, c2, c3 = st.columns(3)

                with c1:
                    if st.button("⬅️", key=f"l_{t['id']}_{status}"):
                        if status == "Todo":
                            move_ticket(t["id"], "Backlog")
                        elif status == "In Progress":
                            move_ticket(t["id"], "Todo")
                        elif status == "Done":
                            move_ticket(t["id"], "In Progress")

                with c2:
                    if st.button("➡️", key=f"r_{t['id']}_{status}"):
                        if status == "Backlog":
                            move_ticket(t["id"], "Todo")
                        elif status == "Todo":
                            move_ticket(t["id"], "In Progress")
                        elif status == "In Progress":
                            move_ticket(t["id"], "Done")

                with c3:
                    if st.button("Details", key=f"d_{t['id']}_{status}"):
                        st.session_state.selected_ticket = t["id"]

# =============================
# DETAILS VIEW
# =============================
if st.session_state.selected_ticket:

    ticket = next(
        (t for t in st.session_state.tickets if t["id"] == st.session_state.selected_ticket),
        None
    )

    if ticket:
        st.markdown("---")
        st.subheader("🎫 Ticket Details")

        # =====================
        # EDIT BASIC INFO
        # =====================
        ticket["title"] = st.text_input("Title", ticket["title"])
        ticket["priority"] = st.selectbox(
            "Priority",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(ticket["priority"])
        )

        # LABELS
        labels_input = st.text_input(
            "Labels (comma separated)",
            ", ".join(ticket.get("labels", []))
        )
        ticket["labels"] = [l.strip() for l in labels_input.split(",") if l.strip()]

        st.markdown("### Description")
        ticket["description"] = st.text_area("Edit Description", ticket["description"])

        # =====================
        # SUBTASKS
        # =====================
        st.markdown("### ✅ Subtasks")

        new_task = st.text_input("Add subtask")

        if st.button("Add Subtask"):
            if new_task:
                ticket["subtasks"].append({"text": new_task, "done": False})

        for i, sub in enumerate(ticket["subtasks"]):
            col1, col2, col3 = st.columns([6, 1, 1])

            with col1:
                checked = st.checkbox(
                    sub["text"],
                    sub["done"],
                    key=f"sub_{ticket['id']}_{i}"
                )
                ticket["subtasks"][i]["done"] = checked

            with col2:
                if st.button("✏️", key=f"edit_{ticket['id']}_{i}"):
                    new_text = st.text_input(
                        "Edit subtask",
                        sub["text"],
                        key=f"edit_input_{ticket['id']}_{i}"
                    )
                    ticket["subtasks"][i]["text"] = new_text

            with col3:
                if st.button("❌", key=f"del_{ticket['id']}_{i}"):
                    ticket["subtasks"].pop(i)
                    st.rerun()

        # =====================
        # COMMENTS
        # =====================
        st.markdown("### 💬 Comments")

        new_comment = st.text_input("Add comment")

        if st.button("Add Comment"):
            if new_comment:
                ticket["comments"].append(new_comment)

        for i, c in enumerate(ticket["comments"]):
            col1, col2 = st.columns([8, 1])

            with col1:
                st.write(f"💬 {c}")

            with col2:
                if st.button("❌", key=f"del_comment_{ticket['id']}_{i}"):
                    ticket["comments"].pop(i)
                    st.rerun()

        # =====================
        # PROGRESS
        # =====================
        progress = get_progress(ticket)
        st.markdown("### 📊 Progress")
        st.progress(progress / 100)
        st.write(f"{progress}% complete")

        # =====================
        # BACK BUTTON
        # =====================
        if st.button("⬅ Back"):
            st.session_state.selected_ticket = None
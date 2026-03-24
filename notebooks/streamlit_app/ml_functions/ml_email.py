from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import List, Optional, Sequence

from dotenv import load_dotenv


# =========================
# Load environment
# =========================

load_dotenv()


# =========================
# Helpers
# =========================

def str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        raise ValueError(f"Missing required environment variable: {name}")
    return value or ""


# =========================
# Config / Models
# =========================

@dataclass
class SMTPConfig:
    host: str
    port: int
    username: str
    password: str
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30

    def validate(self) -> None:
        if not self.host:
            raise ValueError("SMTP host is required.")
        if not self.port:
            raise ValueError("SMTP port is required.")
        if not self.username:
            raise ValueError("SMTP username is required.")
        if not self.password:
            raise ValueError("SMTP password is required.")
        if self.use_tls and self.use_ssl:
            raise ValueError("use_tls and use_ssl cannot both be True.")


@dataclass
class EmailContent:
    subject: str
    sender: str
    recipients: List[str]
    text_body: Optional[str] = None
    html_body: Optional[str] = None
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    reply_to: Optional[str] = None

    def validate(self) -> None:
        if not self.subject:
            raise ValueError("Email subject is required.")
        if not self.sender:
            raise ValueError("Sender email is required.")
        if not self.recipients:
            raise ValueError("At least one recipient is required.")
        if not self.text_body and not self.html_body:
            raise ValueError("Either text_body or html_body must be provided.")


# =========================
# Provider presets
# =========================

def gmail_smtp_config(username: str, password: str) -> SMTPConfig:
    return SMTPConfig(
        host="smtp.gmail.com",
        port=587,
        username=username,
        password=password,
        use_tls=True,
        use_ssl=False,
    )


def outlook_smtp_config(username: str, password: str) -> SMTPConfig:
    return SMTPConfig(
        host="smtp.office365.com",
        port=587,
        username=username,
        password=password,
        use_tls=True,
        use_ssl=False,
    )


def smtp_config_from_env() -> SMTPConfig:
    provider = get_env("SMTP_PROVIDER", default="custom").lower()

    username = get_env("SMTP_USERNAME", required=True)
    password = get_env("SMTP_PASSWORD", required=True)

    if provider == "gmail":
        return gmail_smtp_config(username=username, password=password)

    if provider == "outlook":
        return outlook_smtp_config(username=username, password=password)

    return SMTPConfig(
        host=get_env("SMTP_HOST", required=True),
        port=int(get_env("SMTP_PORT", default="587")),
        username=username,
        password=password,
        use_tls=str_to_bool(get_env("SMTP_USE_TLS", default="true"), default=True),
        use_ssl=str_to_bool(get_env("SMTP_USE_SSL", default="false"), default=False),
        timeout=int(get_env("SMTP_TIMEOUT", default="30")),
    )


# =========================
# Message builder
# =========================

class EmailBuilder:
    @staticmethod
    def build(email_data: EmailContent) -> EmailMessage:
        email_data.validate()

        msg = EmailMessage()
        msg["Subject"] = email_data.subject
        msg["From"] = email_data.sender
        msg["To"] = ", ".join(email_data.recipients)

        if email_data.cc:
            msg["Cc"] = ", ".join(email_data.cc)

        if email_data.reply_to:
            msg["Reply-To"] = email_data.reply_to

        if email_data.text_body and not email_data.html_body:
            msg.set_content(email_data.text_body)
        elif email_data.html_body and not email_data.text_body:
            msg.set_content(
                "This email contains HTML content. Please use an HTML-capable email client."
            )
            msg.add_alternative(email_data.html_body, subtype="html")
        else:
            msg.set_content(email_data.text_body or "")
            msg.add_alternative(email_data.html_body or "", subtype="html")

        return msg


# =========================
# SMTP sender
# =========================

class SMTPSender:
    def __init__(self, config: SMTPConfig):
        self.config = config
        self.config.validate()

    def _connect(self) -> smtplib.SMTP | smtplib.SMTP_SSL:
        context = ssl.create_default_context()

        if self.config.use_ssl:
            server = smtplib.SMTP_SSL(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout,
                context=context,
            )
        else:
            server = smtplib.SMTP(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout,
            )
            server.ehlo()
            if self.config.use_tls:
                server.starttls(context=context)
                server.ehlo()

        server.login(self.config.username, self.config.password)
        return server

    def send(self, email_data: EmailContent) -> dict:
        msg = EmailBuilder.build(email_data)
        all_recipients = self._merge_recipients(
            email_data.recipients,
            email_data.cc,
            email_data.bcc,
        )

        try:
            with self._connect() as server:
                server.send_message(msg, to_addrs=all_recipients)

            return {
                "success": True,
                "message": "Email sent successfully.",
                "recipients": all_recipients,
            }
        except smtplib.SMTPAuthenticationError as exc:
            return {
                "success": False,
                "message": "SMTP authentication failed.",
                "error": str(exc),
            }
        except smtplib.SMTPException as exc:
            return {
                "success": False,
                "message": "SMTP error occurred while sending email.",
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "success": False,
                "message": "Unexpected error occurred while sending email.",
                "error": str(exc),
            }

    @staticmethod
    def _merge_recipients(
        recipients: Sequence[str],
        cc: Sequence[str],
        bcc: Sequence[str],
    ) -> List[str]:
        merged = []
        seen = set()

        for address in list(recipients) + list(cc) + list(bcc):
            normalized = address.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)

        return merged


# =========================
# Example usage
# =========================

def main() -> None:
    smtp_config = smtp_config_from_env()

    email_data = EmailContent(
        subject="Test email from Python SMTP",
        sender=get_env("EMAIL_SENDER", required=True),
        recipients=[
            "ayhannf91@gmail.com",
            "martin.lohr@outlook.com",
        ],
        html_body="""
        <html>
            <body>
                <h2>Hello,</h2>
                <p>This is a <b>HTML test email</b>.</p>
                <p>Best regards</p>
            </body>
        </html>
        """,
        reply_to=get_env("EMAIL_REPLY_TO", default=""),
    )

    sender = SMTPSender(smtp_config)
    result = sender.send(email_data)
    print(result)


if __name__ == "__main__":
    main()
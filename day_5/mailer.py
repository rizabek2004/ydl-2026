"""Optional email feature (requirements.md "Email — по желанию").

Sends a short conversation summary to the administrator via MailerSend.

Hard safety rules enforced here, matching requirements.md:
  - Recipient is ALWAYS config.ADMIN_EMAIL (your own address). The caller cannot
    send to an arbitrary address.
  - Sending happens only when this function is called, which the UI does only on an
    explicit button press. There is NO loop and NO automatic resend.
"""
import config


def send_summary(
    summary_text: str,
    subject: str = "Chat summary / Әңгіме қорытындысы / Саммари разговора",
) -> str:
    """Send `summary_text` to the admin (yourself). Returns the message id.

    `summary_text` is expected to be the trilingual (EN/KZ/RU) summary.
    Raises RuntimeError if MailerSend is not configured.
    """
    if not config.email_enabled():
        raise RuntimeError(
            "Email disabled: set MAILERSEND_API_KEY env var to enable sending."
        )

    # Imported lazily so the rest of the app runs without the mailersend package.
    from mailersend import MailerSendClient, EmailBuilder

    ms = MailerSendClient(api_key=config.MAILERSEND_API_KEY)

    html_body = summary_text.replace("\n", "<br>")

    # Recipient is hardcoded to the admin address — never a user-supplied value.
    email = (
        EmailBuilder()
        .from_email(config.FROM_EMAIL, config.FROM_NAME)
        .to_many([{"email": config.ADMIN_EMAIL, "name": "Admin"}])
        .subject(subject)
        .html(f"<h2>Conversation summary (EN / KZ / RU)</h2><p>{html_body}</p>")
        .text(summary_text)
        .build()
    )
    response = ms.emails.send(email)
    return getattr(response, "message_id", "sent")

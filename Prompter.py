import email
from email.policy import default
from email.header import decode_header
from bs4 import BeautifulSoup, Comment
import tiktoken  # For token counting


class Prompter:
    INSTRUCTIONS = """
        I want you to act as a spam detector to determine whether a given
    email is a phishing email or a legitimate email. Your analysis
    should be thorough and evidence-based. Phishing emails often
    impersonate legitimate brands and use social engineering techniques
    to deceive users. These techniques include, but are not limited to:
    fake rewards, fake warnings about account problems, and creating
    a sense of urgency or interest. Spoofing the sender address and
    embedding deceptive HTML links are also common tactics.
    Analyze the email by following these steps:
    1. Identify any impersonation of well-known brands.
    2. Examine the email header for spoofing signs, such as
    discrepancies in the sender name or email address.
    Evaluate the subject line for typical phishing characteristics
    (e.g., urgency, promise of reward). Note that the To address has
    been replaced with a dummy address.
    3. Analyze the email body for social engineering tactics designed to
    induce clicks on hyperlinks. Inspect URLs to determine if they are
    misleading or lead to suspicious websites.
    4. Provide a comprehensive evaluation of the email, highlighting
    specific elements that support your conclusion. Include a detailed
    explanation of any phishing or legitimacy indicators found in the
    email.
    5. Summarize your findings and provide your final verdict on the
    legitimacy of the email, supported by the evidence you gathered.
    Email:
    """

    def count_tokens(self, text, model="llama2"):
        """Count the number of tokens in a text using tiktoken."""
        tokenizer = tiktoken.get_encoding(
            "cl100k_base"
        )  # Adjust encoding if required for a different LLM
        return len(tokenizer.encode(text))

    def simplify_html(self, html, token_limit):
        """Simplify HTML content to reduce token count."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove unnecessary tags and attributes
        for tag in soup(["style", "script", "meta", "link"]):
            tag.decompose()

        # Remove comments
        for comment in soup.find_all(
            string=lambda text: isinstance(text, Comment)
        ):
            comment.extract()

        # Remove all attributes except for major ones
        allowed_attrs = {"src", "href", "alt", "title", "name", "id", "class"}
        for tag in soup.find_all(True):
            tag.attrs = {
                key: value for key, value in tag.attrs.items() if key in allowed_attrs
            }

        # Remove tags with no text content
        for tag in soup.find_all():
            if not tag.get_text(strip=True):
                tag.decompose()

        # Unwrap certain tags
        for tag_name in ["font", "strong", "b"]:
            for tag in soup.find_all(tag_name):
                tag.unwrap()

        # Reduce src and href attributes
        for tag in soup.find_all(["img", "a"]):
            if tag.name == "img" and "src" in tag.attrs:
                tag["src"] = "/".join(tag["src"].split("/")[:2])[:10]
            if tag.name == "a" and "href" in tag.attrs:
                tag["href"] = "/".join(tag["href"].split("/")[:2])[:10]

        # Iteratively remove HTML elements from the center if token limit is exceeded
        while self.count_tokens(str(soup)) > token_limit:
            all_tags = soup.find_all(True)
            if not all_tags:
                break
            all_tags[len(all_tags) // 2].decompose()

        return str(soup)

    def simplify_text(self, text, token_limit):
        """Simplify plain text content to reduce token count."""
        lines = text.splitlines()
        while self.count_tokens("\n".join(lines)) > token_limit:
            if len(lines) > 2:
                del lines[len(lines) // 2]  # Remove lines from the middle
            else:
                break
        return "\n".join(lines)

    def process_eml_file_with_limit(self, eml_path, token_limit=3000):
        """
        Processes an .eml file, simplifies content to fit within a token limit.

        :param eml_path: Path to the .eml file.
        :param token_limit: The maximum allowed number of tokens for the email content.
        :return: A dictionary with processed email components.
        """
        try:
            with open(eml_path, "r", encoding="utf-8", errors="ignore") as eml_file:
                email_message = email.message_from_file(eml_file, policy=default)

            # Decode headers
            headers = {}
            for key, value in email_message.items():
                decoded_key = key
                decoded_value = decode_header(value)
                headers[decoded_key] = "".join(
                    (
                        fragment.decode(encoding or "utf-8", errors="replace")
                        if isinstance(fragment, bytes)
                        else fragment
                    )
                    for fragment, encoding in decoded_value
                )
            headers = {
                k: v
                for k, v in headers.items()
                if not (
                    k.startswith("X-")
                    or k in {"DKIM-Signature", "Authentication-Results", "Received-SPF"}
                )
            }

            # Process body
            body = None
            if email_message.is_multipart():
                html_part = None
                plain_part = None
                for part in email_message.walk():
                    try:
                        if part.get_content_type() == "text/html":
                            html_part = part.get_payload(decode=True).decode(
                                part.get_content_charset() or "utf-8", errors="replace"
                            )
                        elif part.get_content_type() == "text/plain":
                            plain_part = part.get_payload(decode=True).decode(
                                part.get_content_charset() or "utf-8", errors="replace"
                            )
                    except (UnicodeDecodeError, AttributeError):
                        pass  # Handle decoding errors gracefully

                if html_part:  # Prioritize HTML
                    body = self.simplify_html(html_part, token_limit)
                elif plain_part:
                    body = self.simplify_text(plain_part, token_limit)
            else:
                try:
                    if email_message.get_content_type() == "text/html":
                        body = self.simplify_html(
                            email_message.get_payload(decode=True).decode(
                                email_message.get_content_charset() or "utf-8",
                                errors="replace",
                            ),
                            token_limit,
                        )
                    elif email_message.get_content_type() == "text/plain":
                        body = self.simplify_text(
                            email_message.get_payload(decode=True).decode(
                                email_message.get_content_charset() or "utf-8",
                                errors="replace",
                            ),
                            token_limit,
                        )
                except (UnicodeDecodeError, AttributeError):
                    pass  # Handle decoding errors gracefully

            return {
                "headers": headers,
                "body": body,
            }

        except Exception as e:
            print(f"Error processing .eml file: {e}")
            return None

    def get_prompt(self, text: str) -> str:
        return f"""
        {self.INSTRUCTIONS}
        \'\'\'{text}\'\'\'
        """

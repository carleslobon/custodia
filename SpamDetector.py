import google.generativeai as genai
import json
from Prompter import Prompter

API_KEY = "AIzaSyDZyBoLtY9TKlrDQlEYWYf6H4mZKSv2Uj8"
FLASH_MODEL = "gemini-1.5-flash"
PRO_MODEL = "gemini-1.5-pro"


class SpamDetector:
    def __init__(self):
        self.api_key = API_KEY
        self.model_name = FLASH_MODEL
        self.model = None
        self._configure()

    def _configure(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name, tools=[self.print_phishing_result]
        )
        self.prompter = Prompter()

    @staticmethod
    def print_phishing_result(
        is_phishing: bool,
        phishing_score: float,
        brand_impersonated: str,
        rationales: str,
        brief_reason: str,
    ):
        """Outputs whether a given email is phishing or legitimate.

        Args:
            is_phishing (bool): True if the email is classified as phishing, otherwise False.
            phishing_score (int): Confidence score between 1 and 10.
            brand_impersonated (str): The brand being impersonated, if any.
            rationales (str): Detailed explanation of the decision.
            brief_reason (str): Short summary of the decision reasons.

        Returns:
            dict: A structured output with phishing detection results.
        """
        return {
            "is_phishing": is_phishing,
            "phishing_score": phishing_score,
            "brand_impersonated": brand_impersonated,
            "rationales": rationales,
            "brief_reason": brief_reason,
        }

    def parse_response(self, response: str) -> dict:
        fc = response.candidates[0].content.parts[0].function_call
        args_dict = dict(fc.args)
        return args_dict

    def send_to_model(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return self.parse_response(response)
    
    def is_email_phishing(self, email_path: str) -> dict:
        prompt = self.prompter.process_eml_file_with_limit(email_path)
        prompt = self.prompter.get_prompt(json.dumps(prompt))
        return self.send_to_model(prompt)

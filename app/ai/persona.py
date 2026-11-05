"""Persona manager for loading and dynamically tailoring chatbot personalities."""

from pathlib import Path
from typing import Dict
from app.core.logging import logger

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read_prompt_file(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    logger.warning("Prompt file %s not found in %s", filename, PROMPTS_DIR)
    return ""


class PersonaManager:
    """Loads prompt templates and adjusts persona directives."""

    def __init__(self):
        self.system_prompt = _read_prompt_file("system.txt")
        self.personas: Dict[str, str] = {
            "professional": _read_prompt_file("professional.txt"),
            "casual": _read_prompt_file("casual.txt"),
            "humorous": _read_prompt_file("humorous.txt"),
        }

    def get_persona_prompt(self, persona_name: str = "casual", is_sensitive: bool = False) -> str:
        """Retrieve the persona prompt instructions, adjusting for sensitivity if needed.

        Args:
            persona_name: 'professional' | 'casual' | 'humorous'.
            is_sensitive: If True, tones down humorous or casual style to empathetic & direct.

        Returns:
            str: Persona instructions prompt text.
        """
        persona = persona_name.lower().strip()
        if persona not in self.personas:
            persona = "casual"

        if is_sensitive and persona == "humorous":
            return (
                "TONE OVERRIDE: The user is discussing a sensitive or serious topic. "
                "Tone down humor completely. Respond with genuine empathy, calmness, and helpful clarity."
            )

        return self.personas[persona]

    def build_system_directive(self, persona_name: str = "casual", is_sensitive: bool = False) -> str:
        """Compose the full system directive combining core system rules and persona styling."""
        persona_directive = self.get_persona_prompt(persona_name, is_sensitive)
        return f"{self.system_prompt}\n\n[Persona Guidelines]\n{persona_directive}"


persona_manager = PersonaManager()

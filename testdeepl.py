import os
import re
import deepl

# -------------------------------
# Pre-processing functions
# -------------------------------

def protect_bold_sections(text):
    """
    Replace **section** with <b>section</b>
    """
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

def restore_bold_sections(text):
    """
    Replace <b>section</b> with **section**
    """
    return re.sub(r"<b>(.*?)</b>", r"**\1**", text)
# -------------------------------
# Translation function
# -------------------------------

def translate_text_deepl(text, target_lang="FR"):
    """
    Translate text using DeepL API
    """
    auth_key = os.getenv("DEEPL_API_KEY")
    if not auth_key:
        raise ValueError("DEEPL_API_KEY environment variable not set")
    
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(text, target_lang=target_lang)
    return result.text

# -------------------------------
# Example main script
# -------------------------------

if __name__ == "__main__":
    original_text = """
**Context**
Oasis, a beloved band known for their hits in the 90s, has been missed by many fans over the years.

**What happened**
toto
""".strip()

    print("🔹 Original Text:\n")
    print(original_text)
    print("\n" + "="*50 + "\n")

    # Step 1: Protect formatting
    protected = protect_bold_sections(original_text)
    print(protected)
    # Step 2: Translate
    translated = translate_text_deepl(protected, target_lang="FR")
    print(translated)
    # # Step 3: Restore formatting
    final = restore_bold_sections(translated)

    print("🔹 Translated Text:\n")
    print(final)

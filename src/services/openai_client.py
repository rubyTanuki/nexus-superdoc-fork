import os
from openai import OpenAI

class OpenAIProcessor:
    def __init__(self, emb_model="text-embedding-ada-002", gen_model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model
        self.gen_model = gen_model

    # --- Stage 3: Embedding Logic ---
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts: return []
        cleaned = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(input=cleaned, model=self.emb_model)
        return [data.embedding for data in response.data]

    # --- Stage 5.2: LLM Heading Generation ---
    def generate_headings_batch(self, text_snippets: list[str]) -> list[str]:
        """
        Takes a list of straggler content snippets and returns AI titles.
        Using .batch style logic or parallel calls.
        """
        results = []
        for snippet in text_snippets:
            response = self.client.chat.completions.create(
                model=self.gen_model,
                messages=[
                    {"role": "system", "content": "Generate a concise, title-case heading (4-7 words) for this text."},
                    {"role": "user", "content": snippet}
                ],
                temperature=0.1
            )
            results.append(response.choices[0].message.content.strip())
        return results
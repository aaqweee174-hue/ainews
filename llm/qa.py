from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_llm(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
Answer based on the context below.
If the answer is not directly in the context, summarize related info or say 'Not enough info'.

Context:
{context_text}

Question: {query}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

  
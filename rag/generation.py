from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are a precise question-answering system.

    Answer the question using ONLY the provided context.

    If the answer is not explicitly contained in the context, say:
    "I don't have enough information to answer this."

    Context:
    {context}

    Question
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
        )
    
    return response.choices[0].message.content.strip()
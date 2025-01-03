"""
Minimal placeholder for GPT-based text generation.
Expand with actual OpenAI calls if desired.
"""

import openai

def generate_speech(role, seat_id, intent, context):
    """
    Call OpenAI API for speech generation. 
    Currently returns a placeholder string.
    """
    # If you want real GPT calls, do something like:
    # prompt = f"You are seat {seat_id}, a {role}. Your intent: {intent}.\nContext: {context}"
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "system", "content": prompt}],
    # )
    # text_out = response.choices[0].message.content
    # return text_out

    return f"[Seat {seat_id} / {role}] says: I'm doing {intent}!"

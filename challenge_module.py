from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_questions(text, num=3):
    prompt = f"Generate {num} logical questions from this text:\n{text[:1000]}"
    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]['generated_text'].split('\n')[:num]

def evaluate_answer(user_answer, correct_text):
    return "Correct" if user_answer.lower() in correct_text.lower() else f"Incorrect. Reference: {correct_text}"


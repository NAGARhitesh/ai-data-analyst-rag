from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    device=-1
)

def get_response(prompt):
    try:
        response = generator(
            prompt,
            max_length=200,
            do_sample=False
        )
        return response[0]["generated_text"]

    except Exception as e:
        return f"⚠️ Error: {str(e)}"
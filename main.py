from generator import ImageGenerator
from utils import validate_prompt

def run():
    prompt = "A futuristic trading dashboard overlooking a city skyline"
    prompt = validate_prompt(prompt)
    generator = ImageGenerator()
    result = generator.generate(prompt)

    print(result["message"])
    print(f"Saved as: {result['filename']}")

if __name__ == "__main__":
    run()

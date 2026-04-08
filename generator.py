class ImageGenerator:
    def generate(self, prompt):
        filename = "generated_output.png"
        message = f"Generated mock image for prompt: {prompt}"
        return {"filename": filename, "message": message}

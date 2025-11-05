import requests
import io, base64
from typing import List

try:
    from PIL import Image
except ImportError:
    print("Pillow library not found. Please install it using: pip install Pillow")
    Image = None


def caption_images_via_gemini(image_bytes_list: List[bytes], api_key: str) -> List[str]:
    """
    Captions a list of images using the Google Gemini Pro Vision API.

    Args:
        image_bytes_list: A list where each item is the raw bytes of an image.
        api_key: Your Google AI API key.

    Returns:
        A list of strings, where each string is the generated caption for the
        corresponding image. Returns an error message string for any failed image.
    """
    if not Image:
        # Pillow is not installed, return an error for all images.
        return ["Error: Pillow library is required but not installed."] * len(image_bytes_list)

    captions = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    for img_bytes in image_bytes_list:
        try:
            # --- 1. Dynamically Detect Image Format (The Fix) ---
            image = Image.open(io.BytesIO(img_bytes))
            # The format attribute gives 'JPEG', 'PNG', etc.
            image_format = image.format.lower()
            if image_format not in ["jpeg", "png", "webp", "gif", "heic"]:
                captions.append(f"Error: Unsupported image format '{image_format}'")
                continue

            mime_type = f"image/{image_format}"

            # --- 2. Construct Payload with a Text Prompt (The Fix) ---
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                # This tells Gemini WHAT to do with the image
                                "text": "Describe this image in brief. Analyze any charts, graphs, or text present."
                            },
                            {
                                "inline_data": {
                                    "mimeType": mime_type,
                                    "data": b64_img
                                }
                            }
                        ]
                    }
                ]
            }

            # --- 3. Make the API Call ---
            response = requests.post(url, headers=headers, json=payload)

            # --- 4. Robust Error Handling (The Fix) ---
            if response.status_code == 200:
                data = response.json()
                # Safely extract the text using .get() to avoid crashes
                try:
                    caption = data["candidates"][0]["content"]["parts"][0]["text"]
                    captions.append(caption.strip())
                except (KeyError, IndexError):
                    # This happens if the model refuses to answer (e.g., safety settings)
                    captions.append("Caption could not be generated (possible safety block or empty response).")
            else:
                # Provide a detailed error message
                error_details = response.text
                captions.append(f"Error {response.status_code}: {error_details}")

        except Exception as e:
            # This catches errors like invalid image bytes
            captions.append(f"An unexpected error occurred: {e}")

    return captions


if __name__ == '__main__':
    GOOGLE_API_KEY = ""

    if GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
        print("Please replace 'YOUR_GOOGLE_AI_API_KEY' with your actual API key.")
    else:
        # Create a dummy 1x1 red pixel PNG image for testing
        dummy_png_bytes = base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
        )

        # Create a list with one image to test the function
        images_to_caption = [dummy_png_bytes]

        # Call the function
        generated_captions = caption_images_via_gemini(images_to_caption, GOOGLE_API_KEY)

        # Print the results
        for i, caption in enumerate(generated_captions):
            print(f"Caption for Image {i + 1}:")
            print(caption)
            print("-" * 20)
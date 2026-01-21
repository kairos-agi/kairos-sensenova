      
import os
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class PromptRewriter:
    def __init__(self, model_path: str, dtype="auto",device_map="auto"):
        print("Loading VLM from:", model_path)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"

        self.prompt_alignment = """
            You are an expert in rewriting and expanding user prompts to match the detailed, narrative style of a professional video captioning model.

            Your task is to take a short user-provided prompt (and an optional reference image that represents the video's initial frame) and expand it into a coherent, model-aligned description prompt that instructs the captioning model to generate high-quality video captions.

            Follow these rules strictly:

            1. Expand the user’s prompt so that it clearly guides the captioning model to describe the full visual content of the video, including:
            - Key subjects such as people, animals, and objects
            - The surrounding environment, setting, and background details
            - Lighting conditions, atmosphere, and any relevant framing or camera behavior
            - A clear sequence of actions, events, or changes taking place throughout the video

            2. If a reference image is provided, use the visual content of the image together with the original user prompt as the basis for expansion; if no image is provided, base the expansion only on the original user prompt.

            3. Ensure the rewritten prompt flows naturally as a single, unified narrative paragraph that integrates both static scene elements and dynamic actions seamlessly.

            4. Use vivid, factual, and objective language. Avoid speculation, emotional interpretation, or assumptions that go beyond what is suggested by the user’s original input or visible in the image.

            5. The final rewritten prompt must be in English, in plain text, and contain no added information that is not implied or supported by the user's input or the reference image.

            Example input:
            "A man walking on the street."

            Example rewritten prompt:
            "A man walks along a quiet city street lined with low-rise buildings and scattered greenery. The camera follows him from behind at a steady pace, capturing his casual movements and the soft afternoon light reflecting off the pavement. Cars pass occasionally in the background, and the surrounding storefronts and signs create a calm urban atmosphere as he continues down the sidewalk."
            """

        self.instruction_alignment = (
            "\n\nBelow is the user-provided short prompt. "
            "Expand it according to the style and guidance above:\n"
        )

    def load_image(self, image_path: str, max_size=None):
        img = Image.open(image_path).convert("RGB")
        if max_size is not None:
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h))
                # print(f"[load_image] Resized image from ({w}, {h}) to ({new_w}, {new_h})")
            else:
                # print(f"[load_image] Image smaller than max_size, kept original size ({w}, {h})")
                pass
        else:
            w, h = img.size
            # print(f"[load_image] Using original image size ({w}, {h})")

        return img

    def rewrite_prompt(
        self,
        user_prompt: str,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
        max_image_size: int | None = None,
    ) -> str:

        text = self.prompt_alignment + self.instruction_alignment + user_prompt

        if image_path is None or image_path == "":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
            # print("[Mode] Text-only prompt expansion")

        else:
            image = self.load_image(image_path, max_image_size)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            # print(f"[Mode] Multimodal prompt expansion, image={image_path}")

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()



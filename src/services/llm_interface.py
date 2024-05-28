import openai

class LLMInterface():
    def __init__(self, grounding, text_prompt):
        self.grounding = grounding
        self.text_prompt = text_prompt

    def api_call(self, type):
        openai.api_base = "https://uksouth-openai-shareable.openai.azure.com/"
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = "8579bb167434413ca8a52aed547f5c23"
        deployment_name = "gpt-35-turbo"

        if type == 'letter_generation':
            system_context = f'''PROMPTING'''
            llm_text_prompt = f"Context: {self.grounding} Question: {self.text_prompt}"
        else: 
            system_context = f'''PROMPTING'''
            llm_text_prompt = f"Question: {self.text_prompt}"
            
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": llm_text_prompt},
            ],
            temperature=1,
            max_tokens=1000,
        )

        generated_text = response.to_dict()["choices"][0]["message"]["content"]

        return generated_text
from transformers import AutoModelForCausalLM, AutoTokenizer
class VidCapModel():
    def __init__(self, model, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.get_captions = self.model.get_captions

    def get_all_captions(self, file_path):
        return self.get_captions()
    
    def fuse_captions(self):
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        prompt = "Combine the following:"

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
        )
        gen_text = tokenizer.batch_decode(gen_tokens)[0]


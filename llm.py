# LLM API over huggingface/vLLM/...
from transformers import StoppingCriteria


######## Huggingface API ########

class StopByWords(StoppingCriteria):
    # Stopping-words criteria
    def __init__(self, stop_words, tokenizer):
        StoppingCriteria.__init__(self)
        # NOTE stop words应在stirng层面而非token ids层面判断
        self.tokenizer = tokenizer
        self.stop_words = stop_words

    def __call__(self, input_ids, scores=None):
        # NOTE 提高decoding性能, 限制suffix在最后10个token内
        suffix_str = self.tokenizer.decode(input_ids[0][-10:]).strip()
        for sw in self.stop_words:
            if suffix_str.endswith(sw):
                return True
        return False


class gptj_interface:
    # Wrapper for Huggingface's GPT-J generation, including customization of the generation config & stopping-words
    def __init__(self, gptj_tokenizer, gptj_model, stop_words, gen_config={}):
        self.tokenizer = gptj_tokenizer
        self.model = gptj_model
        self.cfg = gen_config
        self.stopper = StopByWords(stop_words, gptj_tokenizer)

    def call_gptj_local(self, cur_prompt):
        # <1, prompt_len>
        inputs_dict = self.tokenizer(cur_prompt, return_tensors='pt').to(self.model.device)
        output_dict = self.model.generate(**inputs_dict,
                return_dict_in_generate=True,
                #output_attentions=True,
                repetition_penalty = self.cfg.get('repetition_penalty', 0.),
                temperature = self.cfg.get('temperature', 1.),
                top_k = self.cfg.get('top_k', 1),
                top_p = self.cfg.get('top_p', 1.),
                max_new_tokens = self.cfg.get('max_new_tokens', 32),
                do_sample = self.cfg.get('do_sample', True),
                pad_token_id = 50256,
                stopping_criteria = [self.stopper,],
                num_return_sequences = 5,)
        topk_output_ids = output_dict.sequences
        output_sents = []
        for output_ids in topk_output_ids:
            output_sent = self.tokenizer.decode(output_ids)
            output_sents.append(output_sent)
        return output_sents
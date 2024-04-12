# LLM API over huggingface/vLLM/...

######## Huggingface API ########
from transformers import StoppingCriteria


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
        with torch.no_grad():
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


######## vLLM API ########
import torch
from vllm import LLM, SamplingParams


class vllm_gptj_interface:
    def __init__(self, stop_words, gen_config={}):
        self.llm = LLM(
            model = "EleutherAI/gpt-j-6B",
            tokenizer = "EleutherAI/gpt-j-6B",
            # dtype = 'float32',
            # Use all GPUs
            tensor_parallel_size = torch.cuda.device_count(),
            # tensor_parallel_size = 1,
            gpu_memory_utilization = 0.6,
        )
        self.sampling_config = SamplingParams(
            # repetition_penalty = gen_config.get('repetition_penalty', 0.),
            temperature = gen_config.get('temperature', 1.),
            best_of = gen_config.get('best_of', 3),
            # n = gen_config.get('top_k', 1),
            # top_k = gen_config.get('top_k', 1),
            # top_p = gen_config.get('top_p', 1.),
            max_tokens = gen_config.get('max_tokens', 32),
            stop = stop_words,
            include_stop_str_in_output = True,
            use_beam_search = gen_config.get('use_beam_search', True),
        )

    def call_gptj_local(self, cur_prompt):
        llm_output = self.llm.generate(cur_prompt, self.sampling_config, use_tqdm=False)
        # For BSZ=1, directly return llm_output[0] (possibly with multi-samples)
        all_samples = [cur_prompt + out_sample.text for out_sample in llm_output[0].outputs]
        return all_samples


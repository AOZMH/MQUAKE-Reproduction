import sys


class MelloContext:

    def __init__(self, qid, question, task_prompt, total_iters=4,):
        self.qid = qid
        self.qu = question.strip()
        self.tp = task_prompt.strip()
        self.context = ''   # Active context after question
        self.left_iters = total_iters   # Max iteration
        self.status = 'Running'
        self.ans = None

        self.prefix = self.tp + "\n\nQuestion: " + self.qu + '\n'
    
    def make_prompt(self):
        return self.tp + "\n\nQuestion: " + self.qu + '\n' + self.context
    
    def make_context_log(self):
        return 'Question: ' + self.qu + '\n' + self.context
    
    def update_llm_response(self, llm_resp):
        # Update results from LLM to get next subquestion / set self.status
        # NOTE llm_resp includes the original context (to comply with huggingface)
        last_sent, sub_qu = llm_resp.strip().split('\n')[-1], llm_resp.strip().split('\n')[-3]
        self.context = llm_resp[len(self.prefix):]

        # Stopped by '\n\n'
        if last_sent.startswith('Final answer: '):
            ans = last_sent[len("Final answer: "):]
            self.status = 'Finished'
            self.ans = ans
            return
        
        # Otherwise stopped by 'Retr...fact:', extract the generated subquestion
        # NOTE StoppingCriteria会保留stop words, 此处更新逻辑以跳过最后Retrieved fact行
        if not sub_qu.startswith('Subquestion: '):
            self.status = 'Error'
            # print('\n\n', llm_resp, '\n\n')
            return # failed case
        self.sub_qu = sub_qu[len("Subquestion: "):]
        self.left_iters -= 1
        if self.left_iters == 0:
            self.status = 'Exhausted'

        # Update context if next iteration (retrieval) is needed
        self.context = self.context[:-len('\nRetrieved fact:')]

    def update_retrieved_fact(self, fact_sent):
        # Append retrieved facts to context
        self.context += '\nRetrieved fact: ' + fact_sent + '.'


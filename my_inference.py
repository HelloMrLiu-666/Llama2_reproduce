from my_model import RMSNorm
from transformers import AutoTokenizer
from my_model import ModelArgs,Transformer
import torch.nn.functional as F

class Llama:
    def __init__(self, model: Transformer, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def build(
            ckpt_dir: str= None,
            tokenizer_path: str= None,
            max_seq_len: int,
            max_batch_size: int,
            model_parallel_size: Optional[int] = None,
            seed: int = 1,
        ) -> "Llama":
        # seed must be the same in all processes
        torch.manual_seed(seed)

        #because I load the tokenizer from huggingface, so no need the tokenizer_path
        tokenizer = AutoTokenizer.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded")
        h_model = AutoModelForMultimodalLM.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded")
        vocab_size=tokenizer.vocab_size
        my_ModelArgs=ModelArgs(vocab_size=vocab_size,max_batch_size=max_batch_size,max_seq_len=max_seq_len)
        m_model=Transformer(params=args)
        #the same, since I load the model from huggingface, so no need the ckpt_dir
        #check if the  parameter names match between my_model & huggingface model     
        d_h=h_model.state_dict()
        d_m=m_model.state_dict()

        #load the parameter from huggingface model
        m_model.load_state_dict(d_h, strict=True)#strict=True, then no need to check key name manually
        # check if each parameter have the same shape
        value_shape_identical = all([d_h.get(key).shape == value.shape for key, value in d_m.items()])
        assert value_shape_identical==True, "my_model & huggingface model parameter should have the same shape"

        return Llama(m_model, tokenizer)


    def text_completion(
            self,
            prompts: List[str],
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_gen_len: Optional[int] = None,
            logprobs: bool = False,
            echo: bool = False,
        ) -> List[CompletionPrediction]:
        
        max_seq_len=self.model.params.max_seq_len
        max_batch_size=self.model.params.max_batch_size
        tokenizer=self.model.tokenizer

        #determine the max_total_seq to initialize the prompt_gen matrix
        id_list=[tokenizer.encode(str_i) for str_i in prompts] #List[str]->List[List[int]]
        assert (len(id_list)<max_batch_size), "the prompt batch size exceed the model max_batch_size"
        max_promp_len=max([len(list_i) for list_i in id_list])
        min_promp_len=min([len(list_i) for list_i in id_list])
        assert (max_promp_len<max_seq_len) and (min_promp_len<max_seq_len), "the prompt length exceed the model max_seq_len"
        max_total_seq=min(max_seq_len, max_promp_len+max_gen_len) #determine the max_total_seq
        pad_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token) #get the pad_id from the tokenizer
        prompt_gen=torch.full((len(id_list), max_total_seq), pad_id,dtype=torch.long)#initialize the prompt_gen matrix of shape(b,max_total_seq)
        for index, id_i in enumerate(id_list): #copy prompt to prompt_gen matrix
            prompt_gen[index,:len(id_i)]=torch.tensor(id_i,dtype=torch.long)


        eos_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token) #get the pad_id from the tokenizer

        pad_mask = prompt_gen==pad_id# (b,max_total_seq)
        eos_mask_reached=torch.full((len(id_list),), True, dtype=torch.bool)# (b,), default to be True
        
        #starting from 0 instead of min_promp_len to generate text, cause the KV cache need to build from 0
        for cur_pos in range(0,max_total_seq):
            input=prompt_gen[:,cur_pos:cur_pos+1]#(b,max_total_seq)->(b,1), don't use prompt_gen[:,cur_pos],or shape would be 1 dimensional (b,)
            logits=self.model.forward(tokens=input, start_pos=cur_pos)#(b,1,vocab) ， umar's cur_pos start from 1, it should be OK, since rope depend on relative position
            logits=logits/temperature#appy the temperature
            probs = F.softmax(logits, dim=-1)#(b,1,vocab) 
            next_token=sample_top_p(probs, top_p)#(b,seq_1,vocab_1) 
            next_token=torch.squeeze(next_token)#(b,seq_1,vocab_1) ->#(b,) 
        # eos_mask_local judgue is only local(single seq),not total seq, so even reach the eos next token could would not be stopped
        # that's why need eos_mask_reached*eos_mask_local
            eos_mask_local = ~(next_token==eos_id)# (b,) ,Eos to be False; not Eos True
            eos_mask_reached=eos_mask_reached and eos_mask_local#Eos to be False,not Eos True; once get to the EOS, all next_token result False, because boolean and property
            #fill the next token prediction back to prompt_gen matrix, if it's padding& not EOS, then copy the next_token; if not keep the original
            original=prompt_gen[:,cur_pos+1]
            #(b,)=where (b,),(b,),(b,)
            prompt_gen[:,cur_pos+1]=torch.where(pad_mask[:,cur_pos+1], next_token, original)

            all_eos = all(~eos_mask_reached)#if all sentence reach the EOS before the max_length, exit the for loop
            if all_eos:
                break
        
        # cut to eos tok if any
        for gen_i in prompt_gen.tolist():
            if eos_id in gen_i:
                first_index=gen_i.index(eos_id)
                gen_i[first_index+1:]=pad_id


            
        #decode the batch id and print
        list_str=tokenizer.batch_decode(prompt_gen)
        # for str_i in list_str:
        #     print(str_i)
        return list_str

    def sample_top_p(probs, p):# (b,seq,vocab)->(b,seq,1)
        # g_enerator = torch.Generator(device=probs.get_device())
        probs, indices=torch.sort(probs, dim=-1, descending=True)# indices are the index in unsorted probs
        probs_sum=torch.cumsum(probs, dim=-1) #(b,seq,vocab)
        mask=probs_sum-probs >p
        #keep the top P
        probs[mask]=0.0 #utilizing the boolean indexing
        #normalize the top P
        probs=probs/torch.sum(probs, dim=-1, keepdim=True)#(b,seq,vocab)/(b,seq,1)->(b,seq,vocab)
        #sample from the top P, recent version pytorch support 3D input tensor for multinomial, the last dim is the sampling dim
        probs_index=torch.multinomial(probs, num_samples=1, generator=None)#(b,seq,vocab)->index (b,seq,1) 
        next_token=torch.gather(indices, dim=-1, probs_index)#(b,seq,vocab),index (b,seq,1) ->(b,seq,1)

        return next_token




if __name__ == '__main__':
    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]
    args=ModelArgs(vocab_size=vocab_size,max_batch_size=max_batch_size,max_seq_len=max_seq_len)
    my_Llma=Llama.build(args=args)
    text=my_Llma.text_completion()
    for str_i in text:
        print(str_i)

    

    
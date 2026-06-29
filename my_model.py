# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import os

import torch
import torch.nn.functional as F

from torch import nn
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM






@dataclass
class ModelArgs:
    dim: int = 10  # 10 is just for test, should changed to 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(data=torch.ones([1,1,dim], dtype=torch.float32))
        # self.weight=nn.Parameter(data=torch.range(start=0, end=dim-1, step=1))#just for test
        

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        # temp=x[0,0,:]#just for test
        x_powered=torch.pow(x, 2) #element wise power 2, (B,seq,dim)
        x_mean=torch.mean(x_powered, dim=2, keepdim=True)# (B,seq,dim)-> (B,seq,1)
        x_rms=torch.pow(x_mean, 0.5) #(B,seq,1)
        output=x/(x_rms+self.eps) #(B,seq,dim)/(B,seq,1) ->broadcasting to (B,seq,dim)
        return output

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output=self.weight * self._norm(x) # (B,seq,dim)*(1,1,dim) -> broadcastint to (B,seq,dim)
        return output


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0): 
    # this function returns the freqs matrix of the model capatity/ability with full max_length, during inference just crop a segment from this full matrix 
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials. the shape should be (max_seq_len,dim/2)

    """
    assert dim%2==0, "the embedding dimension should be even"
    i=torch.range(start=1, end=(dim//2), step=1) #(dim/2,)
    expo=(-2*(i-1)/dim) #(dim/2,)
    theta_i=torch.pow(theta, expo) #(dim/2,)
    m=torch.range(start=0, end=end-1, step=1) #(m,)
    m_theta=torch.outer(m,theta_i).float() #(m,)*(dim/2,) ->(m,dim/2)
    freqs_cis=torch.polar(torch.ones_like(m_theta,dtype=m_theta.dtype), m_theta)  #(m,dim/2), actually is (max_seq_len,dim/2)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor): #check the dimension before apply_rotary_emb
    #following steps should finished before apply_rotary_emb
    #1. (max_seq_len,dim/2) is cropt to->(seq_len,dim/2) but this step is finished at the beginning(Top level Transformer forward, thus the position para no need to passed through multiple layers)
    #2.                                  (seq_len,dim/2) and->(1,seq_len,1,dim/2)
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility. the x input shape is (b,seq,n_heads,head_dim/2)

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    freqs_cis=freqs_cis.unsqueeze(0).unsqueeze(2) #(seq_len,dim/2) ->(1,seq_len,dim/2) ->(1,seq_len,1,dim/2)complex
                                                        # in order to match xq()complex(b,seq,n_heads,head_dim/2)

    assert (freqs_cis.dim() == x.dim()) and (x.dim()==4 ), f"the freqs_cis & x tensor should 4 dimentional tensor"
    assert freqs_cis.shape[1]==x.shape[1], f"the freqs_cis & x's 2nd dimension(seq) should be the same"
    assert freqs_cis.shape[3]==x.shape[3], f"the freqs_cis & x's 4th dimension(dim/2) should be the same"

    return freqs_cis


#
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.(b,seq,n_heads,head_dim)
        xk (torch.Tensor): Key tensor to apply rotary embeddings. (b,seq,n_kv_heads,head_dim)
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials. #(m,dim/2) / #(seq,dim/2) complex 

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings. no shape changing(b,seq,n_heads,head_dim)
    """

    xq_shape = xq.shape
    xk_shape = xk.shape
    xq=torch.view_as_complex(xq.contiguous().view(*xq_shape[:-1],-1,2))   #real(b,seq,n_heads,head_dim)->real(b,seq,n_heads,head_dim/2,2)->complex(b,seq,n_heads,head_dim/2)
    xk=torch.view_as_complex(xk.contiguous().view(*xk_shape[:-1],-1,2))     #complex(b,seq,n_kv_heads,head_dim/2)
    freqs_cis=reshape_for_broadcast(freqs_cis, xq) #(seq,dim/2) ->(1,seq_len,1,dim/2) complex
    xq=xq * freqs_cis #(b,seq,n_heads,head_dim/2) * (1,seq_len,1,dim/2) -> broadcasting (b,seq,n_heads,head_dim/2)
    xk=xk * freqs_cis
    xq=torch.view_as_real(xq).contiguous().view(*xq_shape) # complex (b,seq,n_heads,head_dim/2)->real(b,seq,n_heads,head_dim/2,2)->(b,seq,n_heads,head_dim)
    xk=torch.view_as_real(xk).contiguous().view(*xk_shape)
    return (xq, xk)  

                                                            
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:#(b,seq,n_kv_heads,head_dim)->(b,seq,n_rep*n_kv_heads,head_dim)
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]#(b,seq,n_kv_heads,head_dim)->(b,seq,n_kv_heads,1,head_dim)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)#(b,seq,n_kv_heads,1,head_dim)->(b,seq,n_kv_heads,1*n_rep,head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)#(b,seq,n_kv_heads,1*n_rep,head_dim)->(b,seq,n_rep*n_kv_heads,head_dim)
    )

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    #following is meta code for parellel computing
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        # self.n_local_heads = args.n_heads // model_parallel_size
        # self.n_local_kv_heads = self.n_kv_heads // model_parallel_size

        assert args.n_heads % self.n_kv_heads==0, "the n_heads should be divisable by n_kv_heads"
        assert args.dim % args.n_heads==0, "the dim should be divisable by n_heads"
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = torch.nn.Linear(args.dim, self.head_dim*args.n_heads, bias=False)
        self.wk = torch.nn.Linear(args.dim, self.head_dim*self.n_kv_heads, bias=False)
        self.wv = torch.nn.Linear(args.dim, self.head_dim*self.n_kv_heads, bias=False)
        self.wo = torch.nn.Linear(self.head_dim*args.n_heads, args.dim, bias=False)
        #in real senario, if the b<max_batch_size then the cache_k should cliped along batch dimension, same to the seq_length dimension
        self.cache_k = torch.zeros(args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention. (B,seq,dim)->(B,seq,dim) no shape changing

        """

        bsz, seqlen, _ = x.shape

        xq=self.wq(x) #(b,seq,dim)*(dim,head_dim*n_heads)->     (b,seq,head_dim*n_heads)
        xk=self.wk(x) #(b,seq,dim)*(dim,head_dim*n_kv_heads)->  (b,seq,head_dim*n_kv_heads)
        xv=self.wv(x) #(b,seq,dim)*(dim,head_dim*n_kv_heads)->  (b,seq,head_dim*n_kv_heads)

        xq=xq.view(bsz, seqlen,-1,self.head_dim) #(b,seq,head_dim*n_heads)->(b,seq,n_heads,head_dim)
        xk=xk.view(bsz, seqlen,-1,self.head_dim) #(b,seq,head_dim*n_kv_heads)->(b,seq,n_kv_heads,head_dim)
        xv=xv.view(bsz, seqlen,-1,self.head_dim)

        #apply ROPE should before cache in, the position information should also stored in cache
        #because the cached K&V should have position information, otherwise each attention matrix calculation would recalculate the rotation
        #apply ROPE should after QKV split into multiple heads
        #because the dot product is happened between each small head, head is the smallest operators for the dot product(or inner product in the ROPE)
        #and the dot product take into account the relative angle position(the apply _rotary comes into play)
        #the successive transpose operation would not effect the dimention dim, thus no effect to the positional information
        xq, xk=apply_rotary_emb(xq,xk,freqs_cis) # no shape changing

        #kv cache in
        self.cache_k[:bsz,start_pos:start_pos+seqlen,:,:]=xk #(max_batch_size,max_seq_len,n_kv_heads,head_dim)=(b,seq_1,n_kv_heads,head_dim)
        self.cache_v[:bsz,start_pos:start_pos+seqlen,:,:]=xv
        #kv cache out
        xk=self.cache_k[:bsz,:start_pos+seqlen,:,:]#(b,seq_1,n_kv_heads,head_dim)->(b,seq_cache,n_kv_heads,head_dim)
        xv=self.cache_v[:bsz,:start_pos+seqlen,:,:]

        xk=repeat_kv(xk, self.n_rep)#(b,seq,n_kv_heads,head_dim)->(b,seq,n_rep*n_kv_heads,head_dim)
        xv=repeat_kv(xv, self.n_rep)#(b,seq,n_kv_heads,head_dim)->(b,seq_cache,n_rep*n_kv_heads,head_dim)

        xq=xq.transpose(1,2)#(b,seq,n_heads,head_dim)->(b,n_heads,seq,head_dim)
        xk=xk.transpose(1,2).transpose(2,3)#(b,seq_cache,n_rep*n_kv_heads,head_dim)->(b,n_rep*n_kv_heads,seq_cache,head_dim)->(b,n_rep*n_kv_heads,head_dim,seq_cache)
        xv=xv.transpose(1,2)#(b,seq_cache,n_rep*n_kv_heads,head_dim)->(b,n_rep*n_kv_heads,seq_cache,head_dim)
        output=torch.matmul(xq,xk)/(self.head_dim**0.5)#(b,n_heads,seq,head_dim) @ (b,n_rep*n_kv_heads,head_dim,seq_cache)->(b,n_heads,seq_1,seq_cache)
        output=torch.matmul(output,xv) #(b,n_heads,seq_1,seq_cache)@(b,n_rep*n_kv_heads,seq_cache,head_dim)->(b,n_heads,seq_1,head_dim)
        output=output.transpose(1,2).contiguous().view(bsz,seqlen,-1)#(b,n_heads,seq_1,head_dim)->(b,seq_1,n_heads,head_dim)->(b,seq_1,n_heads*head_dim)
        output=self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            self: (b,seq,dim)
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x): #(B,seq,dim)->(B,seq,dim) no shape changing
        swish1 = nn.SiLU()
        return self.w3(swish1(self.w1(x))*self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(self.dim,self.hidden_dim, args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(self.dim)
        self.ffn_norm = RMSNorm(self.dim)





    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers. (B,seq,dim)->(B,seq,dim) no shape changing

        """

        x=x+self.attention(self.attention_norm(x),start_pos,freqs_cis)
        x=x+self.feed_forward(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)

        self.layers =nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm =RMSNorm(params.dim)
        self.output =  nn.Linear(params.dim, self.vocab_size, bias=False)

        # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
        # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        self.freqs_cis =precompute_freqs_cis(self.params.dim // self.params.n_heads, 2*self.params.max_seq_len)


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.--shape(B,1)
            start_pos (int): Starting position for attention caching. and it's a global variable during interencing, 
                                                                        it should start from 0 because need to build th cache from scratch

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        b,seq=self.shape

        freqs_cis=self.freqs_cis[start_pos:start_pos+seq,:] #(max_seq_len,dim/2)->(seq,dim/2), it should start from 0

        x=self.tok_embeddings(tokens) #(B,seq)->(B,seq,dim), actually the seq is equal to 1
        for layer in self.layers: # input& output shape identical, (B,seq,dim)
            x= layer(x,start_pos,freqs_cis)
        x= self.norm(x) # no chaging shape, (B,seq,dim)
        x=self.output(x) #(B,seq,dim)->(B,seq,vocab), this is the logits

        return x


def check_my_model(): 
    #following code is to login huggingface so as to make use the official Llama model, but Meta refused 
    # Login with your token
    # login(token="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")
    # api = HfApi()
    # print(api.whoami())
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # #printing the visual studio huggingface model download path
    # print("HF_HOME:", os.environ.get("HF_HOME"))
    # print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
    # print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
    
    tokenizer = AutoTokenizer.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded")
    hug_model =  AutoModelForCausalLM.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded")
    vocab_size=tokenizer.vocab_size
    my_ModelArgs=ModelArgs(vocab_size=vocab_size)    
    

    hug_config = hug_model.config
    print(f"hug_config:{hug_config}")
    return hug_model


    #test the RMS norm
        # my_model=Transformer( params=my_ModelArgs)
        # tokens=tokenizer.encode("Hello this is a test")
        # tokens=torch.tensor(tokens,dtype=torch.long)
        # x=tokens.unsqueeze(0).repeat(3,1)
        # output=my_model(tokens=x, start_pos=1)
        # RMS_NORM=RMSNorm(dim=my_ModelArgs.dim, eps= 1e-6)
        # output=RMS_NORM(output)

    # #test repeat_kv
    #     b,seq,n_kv_heads,head_dim=2,4,6,8
    #     x=torch.rand(b,seq,n_kv_heads,head_dim)
    #     repeated_kv=repeat_kv(x=x, n_rep=2)#(b,seq,n_kv_heads,head_dim)->(b,seq,n_rep*n_kv_heads,head_dim)
    #     print("hahah")

    # #test ROPE!!
    # #test precompute_freqs_cis
    #     b,seq,n_heads,head_dim=1,1,1,6
    #     freqs_cis=precompute_freqs_cis(dim=head_dim, end=4, theta = 10000.0)
    # #test reshape_for_broadcast
    #     freqs_cis=freqs_cis[2:3,:] #
    #     # x=torch.rand(b,seq,n_heads,head_dim//2)
    #     # reshape_freqs_cis=reshape_for_broadcast(freqs_cis=freqs_cis, x=x)
    # #test apply_rotary_emb
    #     n_kv_heads=1
    #     xq=torch.range(start=0, end=head_dim-1, step=1)
    #     xq=xq[None, None, None, :]#(b,seq,n_heads,head_dim)
    #     xk=torch.range(start=1, end=head_dim, step=1)
    #     xk=xk[None, None, None, :]#(b,seq,n_kv_heads,head_dim)
    #     xq, xk=apply_rotary_emb(xq=xq,xk=xk,freqs_cis=freqs_cis)

    
# if __name__ == '__main__':
#     check_my_model() 




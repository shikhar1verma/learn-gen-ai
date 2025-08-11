## LLM tasks and use cases

1. It can be used not only for text generation but variety of other tasks.
2. Can be used for summarization.
3. Can be used for language translation.
4. Can be used for coding AI assistant.
5. Can be used for named entity extraction.
6. Can be used for categorization.
7. And many other tasks.

## Transformer Architecture (Encoder Decoder model)

1. All LLMs are based on transformer architecture which came from attention is all you need paper. There can be various variations to these models in architecture. Like some are encoder only, some are decoder only some are both.
    - **Encoder**: Processes full input in parallel, learns contextual embeddings.
    - **Decoder**: Uses masked self-attention (only sees past tokens), predicts next token step-by-step; in encoder-decoder, decoder also attends to encoder outputs.
    - **Architectural variants**: encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5, original Transformer).
2. The deep learning models can have any architecture and researcher tested many. But this self attention concepts was just intuitive and was working with scale as well.
3. The transformer has two components encoder and decoder. Both having similar structure but different workings.
4. Encoder is basically more of finding hidden patterns and prompt language in depth.
5. Decoder is more of better in finding out next word according to some hidden patter it found out.
6. These both are usually work in conjunction. Where encoder output is given to decoder in between of decoder’s self attentions layers and feed forward layers.
7. The basic architecture of both are:
Encoder: Input text → tokenized → Embedding layer → Positional encoding → multi head self attention layer → Feed forward layer → (repeat N layers) → Contextual embeddings.
Decoder: Input text → tokenized → Embedding layer → Positional encoding → Masked Self-Attention multi headed → Cross-Attention (from encoder outputs) → Feed forward layer → (repeat N layers) → Linear function → soft max function → Output text
8. LLM models can be decoder only, encoder only or both.

## Generative AI project Lifecycle

1. Scope: Define use case
2. Select model: Choose existing model or pretrain your own.
3. Adapt and align model:
    1. Prompt engineering with evaluation
    2. Fine tuning with evaluation
    3. Human feedback alignment with evaluation
4. Application integration:
    1. Optimize and deploy model for inference
    2. Augment model and build LLM powered application

## Pretraining of models

1. This is very compute intensive step.
2. It takes thousands of GPUs which run in parallel to train a model.
3. With different techniques training data is split between GPUs and model is trained efficiently.
    - **Data parallelism** (each GPU trains on a different mini-batch)
    - **Model parallelism** (split model layers/tensors across GPUs when too large)
    - Often combined into **pipeline + tensor parallelism** in modern LLM training.
4. Chinchilla is a scaling law of how many tokens are need for specific number of parameters.
    - It says the *optimal* number of training tokens grows linearly with model parameters for fixed compute.
    - *For a fixed compute budget, it’s better to train smaller models on more tokens than huge models on fewer tokens.*
    - Here “tokens” means **training dataset token count**, not prompt tokens at inference.
5. One can calculate most of the compute and money related calculations before hand approximately to train a model.

## Full Fine Tuning

1. The full model can be fine-tuned after general pretraining, either for a single downstream task or for multiple tasks.
2. This is done by providing instructions as prompt and update weights according to tasks.
    - Instruction fine-tuning is *not the same* as task-specific fine-tuning — instruction tuning uses many diverse tasks to improve zero-shot generalization.
3. Due instruction fine tuning on various tasks the model is also called instruct model.
4. There different ways to evaluate the models:
    1. Like with exact word matching, with ROUGE score of n grams, BLUE score of ngrams.
    2. There are also different benchmarks where the models are evaluated like GLUE/SuperGlue, MMLU, Big-Bench-Hard.
    3. Benchmarks: BLEU/ROUGE are mainly for summarization/translation; MMLU/Big-Bench are general LLM evals.
5. But the biggest drawbacks of full fine tuning is Catastrophic forgetting. Where LLM starts to forget the things which it learn from large amount of data which it previously trained on.
6. To tackle this we use parameter efficient fine tuning short for PEFT.

## Performance efficient fine tuning PEFT

1. Basically rather than training all the parameter either we train some of the parameters or train more smaller number of parameters according to tasks.
    - “Selective” fine-tuning usually refers to **partial layer training** (e.g., last few transformer blocks), not just “low information” params.
2. There can be 3 main types of PEFT:
    1. Selective: Train only a subset of the model parameters (e.g., last few transformer layers) instead of the entire model.
    2. Reparameterization: Reparameterize model weights using low rank representation. LoRA or QLoRa. In this a low rank matrices are trained which are product into each other to become same dimension of matrix as of original model parameter matrix. When the inference is given these two were summated together. Advantage is we just need to train those low rank matrices which is like 1-10% of the model size.
        - LoRA: low-rank matrices are **added** to frozen weights (“summated together” after projection).
    3. Additive: So there are two types of methods in this.
        1. Adapters: Which are trainable layers added after self attention or feed forward layers. This changes the architecture of model.
        2. Soft prompt: In this transformer architecture is intact. We add some dimensions in input layers. And keeping the architecture frozen.
        A more popular method is called prompt tuning. Where either trainable layers are added in embedding layer. Or embedding layer is retrained according to inputs.
            
            Just a note soft prompt is not prompt engineering. It is training the embedding layers in different ways.
            
            - Soft prompt tuning modifies **input embeddings only**; doesn’t touch core transformer layers.

## Reinforcement Learning with Human Feedback (RLHF)

1. In this basically we have an agent and an environment. The agent will do some actions on environment then the state of environment is check according to optimized results. If it performed good then reward is provided to agent. If it performed bad then very less reward (in some punishment) is provided to agent.
2. In language terms. Agent is LLM instruct model. Environment is LLM context. Action is token vocabulary. State is current context. Reward is provided by humans or some reward LLM model.
3. It is more mathematical regarding reward policy. And other reward techniques. But it basically does the above.
4. It also helps model to align 3 Hs helpful, honest and harmless.
5. Below are the main steps:
    1. **SFT(supervised fine tuning)**: Train base model with human-written responses.
    2. **Reward model**: Train on human rankings of model outputs.
    3. **PPO(proximal policy optimization)/DPO(direct preference optimization)**: Optimize the LLM to maximize reward model score while staying close to original behavior (KL (Kullback-Leibler Divergence) penalty).
    4. PPO is reinforcement learning–based; DPO is a simpler supervised objective that avoids running full RL.
    - The “environment” here is usually simulated — the LLM isn’t interacting with the real world but with prompts & human ratings.

## Application Integration

1. So there are two main part of it.
    1. First model is optimized and ready to be deployed for inference.
    2. Model is augmented by provided external sources data so applications can be created through LLM powered.
2. LLM optimizations, there are 3 methods:
    1. Distilled: Here teacher LLM trains a smaller student LLM model.
    2. Quantized: Here 32 bit floating point is quantized to 16 bit floating point or 8 bit integer.
        1. 32-bit FP → 16-bit FP (BF16/FP16) or 8/4-bit INT; each has trade-offs in speed vs accuracy.
    3. Pruning: Here parameters of model is pruned which provided little to model performance.
3. RAG (retrieval augmented generation): It is a technique where model is provided with a information text which might have the answer. It is provided in model context so model can better tell answers. As models have cutoff dates. It is better to provided the latest information in the context and let model inference from that.
In this the information is provided by various external sources. Like internet/websearch, databases, documents etc. Or external applications can be used to process or get the answers/information.
    - retrieval step usually uses **vector embeddings + similarity search**,
    - Retrieval flow: Query → embed → vector DB similarity search → retrieve top-k documents → inject into LLM context → generate answer.
4. Chain of Though (CoT) prompting: Here we provided steps in one or few shots of the reasoning task we wanted to do. Then the model also mimics it and tried to provide the correct answer. Again keeping this in note that LLM models only predicts next word they don’t have any reasoning mechanism. But providing them with some reasoning steps in context itself they tries to mimic the same and provide a better reasoning answers.
    - it’s **prompting strategy**, not architectural change.
5. Program aided language models (PAL): In this the LLM models are provided with some actions which they can perform over the code interpreter. Os some other tools. Here we provide code like structure in the prompt context. Then the LLM also mimics the same and give us the code to get the answer of the same the answers steps are run in python interpreter and we get the answer correctly. This can be done in single shot or few shots prompting.
    - these generated programs are *executed externally*, not inside LLM.
6. Reason And Thinking (ReAct): It is combination of chain of though reasoning and action planing. In this basically an LLM given a question first do thinking by having thoughts. Then according to those thoughts having plan to execute an actions. Once action is taken it will have observation. If the answer is questioned in observation then it will stop. Else the loop will continue till it get an answer to the quesiton.
This whole process is called ReAct.
    - “Reasoning + Acting” loop, uses observation to refine next step.

## Responsible AI

1. Toxicity as challenge: Here model exhibits toxic behaviour. By providing answers that is feels as toxic. For this guardrails can be rule-based filters or **classifier models** trained to detect unsafe content.
2. Hallucinations as challenge: Models might provide you answers that looks correct but it actually a wrong answer. So awareness must be there while using these models.
3. Intellectual property as challenge: Making sure there are not copyright issues. People are plagarizing the content.
4. **Bias/Fairness**: unintended demographic biases in outputs.
5. **Privacy**: avoid leaking sensitive training data.
6. **Transparency**: model cards, explainable outputs.
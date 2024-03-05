# Improving Context-Aware Transducers by Early Context Injection and Text Perturbation
# 
# It's been shown by existing research that transducer speech recognition models can benefit from contexts 
# (e.g., contact lists, user specified vocabulary) in addition to acoustic signals. 
# In particular, rare words and named entities can be better recognized with the presence of contexts.
# In this work, we propose novel techniques to improve context-aware transducer models.
# First, we propose to inject contexts into the encoder at an early stage instead of merely at the last layer.
# We further use auxiliary ctc loss to regularize the intermediate context-injected layers.
# Second, we observe that the word error rates on the training data is already very low, even for the rare words, 
# without providing any contexts.
# To enforce the model to leverage the contexts during training, we propose to perturb the reference transcription
# such that the model must rely on the contexts to make correct predictions. 
# We show that our proposed techniques can significantly improve the performance of the context-aware transducer models.
# On LibriSpeech, our proposed techniques together can reduce the word error rate by 5.5% relative to the baseline model,
# making the new state-of-the-art performance for the models of similar architecture.
# On a real-world dataset, ConEC, our proposed techniques can reduce the word error rate by 3.5% relative to the baseline model.

# Existing research suggests that transducer speech recognition models benefit from additional contexts (e.g., contact lists, user specified vocabulary). Rare words and named entities can be better recognized with context.
# In this work, we propose novel techniques to improve context-aware transducer models. First, we propose to inject contexts into the encoder at an early stage instead of merely at the last layer. We further use auxiliary ctc loss to regularize the intermediate context-injected layers. Second, to enforce the model to leverage the contexts during training, we perturb the reference transcription so that the model must rely on the contexts to make correct predictions. On LibriSpeech, our proposed techniques together can reduce the word error rate by 5.5\% relative to the baseline model, making the new state-of-the-art performance. On a real-world dataset, ConEC, our proposed techniques can reduce the word error rate by 3.5\% relative to the baseline model.

# Introduction

# End-to-end (E2E) automatic speech recognition (ASR) has emerged as the dominant solution of ASR, due to its simplicity of modeling and impressive performance.
# % Transducer models have been widely used in automatic speech recognition (ASR) due to their effectiveness and streaming nature.
# % The transducer model has the ability to directly model the alignment between the input speech and the output text.
# A conventional E2E ASR model takes merely acoustics features as input and outputs the corresponding text transcription. 
# However, human speech recognition doesn’t occur in isolation. In addition to acoustic cues, we often rely on various contextual resources to aid in understanding and interpreting spoken content. 
# In particular, these contextual cues play a significant role in recognizing rare words and named entities. 
# Therefore, various contextual biasing techniques have been proposed to improve standard ASR models, 
# including \cite{Fox2022ImprovingCR, Dingliwal2023PersonalizationOC, Lei2023PersonalizationOC} for connectionist temporal classification (CTC) models, 
# \cite{Pundak2018DeepCE, Bruguier2019PhoebePC, Huber2021InstantOW, Zhang2022EndtoendCA} for attention-based encoder-decoder (LAS) models, 
# \cite{Jain2020ContextualRF, Chang2021ContextAwareTT, Kantha2022, Yang2023PromptASRFC, Tang2024ImprovingAC} for transducer models 
# and more recently \cite{Chen2023SALMSL, Sun2023ContextualBO, Lakomkin2023EndtoEndSR, Everson2024TowardsAR} for (or with) large language models.
# As of the types of contexts, we consider lists biasing words in this paper (e.g., contact lists, user specified vocabulary). There are other types of contexts, e.g., visual contexts~\cite{}, date-time and location~\cite{}.

# In general, contextual biasing can be achieved in two ways (or the hybrid of the two), depending on whether the internal representations of the E2E model are modified or not by the contexts.
# If the internal representations are unchanged, contextual biasing most likely happens only during the decoding process, where the contexts are used to guide the beam search, e.g., shallow fusion~\cite{Fox2022ImprovingCR, Zhao2019ShallowFusionEC, Wang2023ContextualBW}.
# Otherwise, the contexts are injected into the E2E model, e.g., with the outputs from cross-attention layers over the biasing lists~\cite{Pundak2018DeepCE, Chang2021ContextAwareTT}. This is called deep biasing or neural biasing, which will be the focus of this paper. 
# With the emergence of LLMs, the internal states of the neural networks can also be modified by prompts, e.g., keyword boosting as in~\cite{Chen2023SALMSL}. Neural biasing has been reported outperforming shallow fusion~\cite{Chang2021ContextAwareTT}, as it has specialized parameters trained to accommodate the contexts.

# This paper proposes two generic techniques for neural biasing. 
# Typically, neural biasing approaches are built on top of existing E2E ASR models. The biasing module employs a cross-attention mechanism to inject the contexts into the models.
# More specifically, each biasing word in the contexts are first encoded into some fixed-size vectors respectively, which are the queries and values for the cross-attention layer. 
# Then, for each frame (or selected frames) of the acoustic features or encoder/decoder embeddings, the frame's embedding is used as the key to attend to the cross-attention layer.
# Ideally, some biasing words (or none of them) are related to the current frame by the attention.
# Then, the attention output is used to modify the frame's embedding, which will be subsequently fed into the next layer of the model or the decoder.

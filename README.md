# Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models

> 🔍 See our paper: [**"Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models"**](https://arxiv.org/abs/2402.17177) [![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightblue?style=flat-square)](https://arxiv.org/abs/2402.17177)
> 
> 📧 Please let us know if you find a mistake or have any suggestions by e-mail: lis221@lehigh.edu

## Table of Contents
- 💡 [About](#about)
- ✨ [Updates](#updates)
- 🕰️ [History of Generative AI in the Vision Domain](#history-of-generative-ai-in-the-vision-domain)
- 📑 [Paper List](#paper-list)
    - [Technology](#technology)
        - [Data Pre-processing](#data-pre-processing)
        - [Modeling](#modeling)
        - [Language Instruction Following](#language-instruction-following)
        - [Prompt Engineering](#prompt-engineering)
        - [Trustworthiness](#trustworthiness)
    - [Application](#application)
        - [Movie](#movie)
        - [Education](#education)
        - [Gaming](#gaming)
        - [Healthcare](#healthcare)
        - [Robotics](#robotics)
- 🔗 [Citation](#citation)
## About
Sora is a text-to-video generative AI model, released by OpenAI in February 2024. The model is trained to generate videos of realistic or imaginative scenes from text instructions and shows potential in simulating the physical world. Based on public technical reports and reverse engineering, this paper presents a comprehensive review of the model's background, related technologies, applications, remaining challenges, and future directions of text-to-video AI models. We first trace Sora's development and investigate the underlying technologies used to build this "world simulator". Then, we describe in detail the applications and potential impact of Sora in multiple industries ranging from filmmaking and education to marketing. We discuss the main challenges and limitations that need to be addressed to widely deploy Sora, such as ensuring safe and unbiased video generation. Lastly, we discuss the future development of Sora and video generation models in general, and how advancements in the field could enable new ways of human-AI interaction, boosting productivity and creativity of video generation.
<div align="center">
<img src="https://raw.githubusercontent.com/lichao-sun/SoraReview/main/image/sora_framework.png" width="85%"></div>

## Updates
- 📄 [07/03/2024] Our paper has been featured in the [AI Tidbits weekly roundup](https://www.aitidbits.ai/p/march-7th-2024#:~:text=consistency%20and%20stability-,Lehigh,-University%20presents%20an).
- 📄 [28/02/2024] Our paper has been uploaded to arXiv and was selected as the Daily Paper by Hugging Face.

## History of Generative AI in the Vision Domain
<div align="center">
<img src="https://raw.githubusercontent.com/lichao-sun/SoraReview/main/image/history.png" width="85%"></div>

## Paper List
<div align="center">
<img src="https://raw.githubusercontent.com/lichao-sun/SoraReview/main/image/paper_list_structure.png" width="70%"></div>

### Technology
####  Data Pre-processing
- (*NeurIPS'23*) Patch n’Pack: Navit, A Vision Transformer for Any Aspect Ratio and Resolution [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/06ea400b9b7cfce6428ec27a371632eb-Paper-Conference.pdf)
- (*ICLR'21*) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [[paper]](https://arxiv.org/abs/2010.11929)[[code]](https://github.com/google-research/vision_transformer)
- (*arXiv 2013.12*) Auto-Encoding Variational Bayes [[paper]](https://arxiv.org/abs/1312.6114)
- (*ICCV'21*) Vivit: A Video Vision Transformer [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.html?ref=https://githubhelp.com)[[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)
- (*ICML'21*) Is Space-Time Attention All You Need for Video Understanding? [[paper]](https://arxiv.org/abs/2102.05095)[[code]](https://github.com/facebookresearch/TimeSformer)
- (*NeurIPS'17*) Neural Discrete Representation Learning [[paper]](https://proceedings.neurips.cc/paper_files/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)[[code]](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)
- (*CVPR'22*) High-Resolution Image Synthesis with Latent Diffusion Models [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)[[code]](https://github.com/CompVis/latent-diffusion)
#### Modeling
- (*JMLR'22*) Cascaded Diffusion Models for High Fidelity Image Generation [[paper]](https://dl.acm.org/doi/abs/10.5555/3586589.3586636)
- (*ICLR'22*) Progressive Distillation for Fast Sampling of Diffusion Models [[paper]](https://arxiv.org/abs/2202.00512)[[code]](https://github.com/google-research/google-research/tree/master/diffusion_distillation)
- Imagen Video: High Definition Video Generation with Diffusion Models [[paper]](https://arxiv.org/abs/2210.02303)
- (*CVPR'23*) Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf)
- (*ICCV'23*) Scalable Diffusion Models with Transformers [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)
- (*CVPR'23*) All Are Worth Words: A ViT Backbone for Diffusion Models [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Bao_All_Are_Worth_Words_A_ViT_Backbone_for_Diffusion_Models_CVPR_2023_paper.pdf)[[code]](https://github.com/baofff/U-ViT)
- (*ICCV'23*) Masked Diffusion Transformer Is a Strong Image Synthesizer [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Masked_Diffusion_Transformer_is_a_Strong_Image_Synthesizer_ICCV_2023_paper.pdf)[[code]](https://github.com/sail-sg/mdt)
- (*arXiv 2023.12*) DiffiT: Diffusion Vision Transformers for Image Generation [[paper]](https://arxiv.org/abs/2312.02139)[[code]](https://github.com/nvlabs/diffit)
- (*CVPR'24*) GenTron: Delving Deep into Diffusion Transformers for Image and Video Generation [[paper]](https://arxiv.org/abs/2312.04557)
- (*arXiv 2023.09*) LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models [[paper]](https://arxiv.org/abs/2309.15103)[[code]](https://github.com/Vchitect/LaVie)
- (*arXiv 2024.01*) Latte: Latent Diffusion Transformer for Video Generation [[paper]](https://arxiv.org/abs/2401.03048)[[code]](https://github.com/Vchitect/Latte)
- (*arXiv 2024.03*) Scaling Rectified Flow Transformers for High-Resolution Image Synthesis [[paper]](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf)
#### Language Instruction Following
- Improving Image Generation with Better Captions [[paper]](https://cdn.openai.com/papers/dall-e-3.pdf)
- (*arXiv 2022.05*) CoCa: Contrastive Captioners are Image-Text Foundation Models [[paper]](https://arxiv.org/abs/2205.01917)[[code]](https://github.com/lucidrains/CoCa-pytorch)
- (*arXiv 2022.12*) VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners [[paper]](https://arxiv.org/abs/2212.04979)
- (*CVPR'23*) InstructPix2Pix: Learning to Follow Image Editing Instructions [[paper]](https://arxiv.org/abs/2211.09800)[[code]](https://github.com/timothybrooks/instruct-pix2pix)
- (*NeurlPS'23*) Visual Instruction Tuning [[paper]](https://arxiv.org/abs/2304.08485)[[code]](https://github.com/haotian-liu/LLaVA)
- (*ICML'23*) mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image, and Video [[paper]](https://arxiv.org/abs/2302.00402)[[code]](https://github.com/X-PLUG/mPLUG-2)
- (*arXiv 2022.05*) GIT: A Generative Image-to-text Transformer for Vision and Language [[paper]](https://arxiv.org/abs/2205.14100)[[code]](https://github.com/microsoft/GenerativeImage2Text)
- (*CVPR'23*) Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning [[paper]](https://arxiv.org/abs/2302.14115)[[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq)
#### Prompt Engineering
- (*arXiv 2023.10*) Unleashing the Potential of Prompt Engineering in Large Language Models: A Comprehensive Review [[paper]](https://arxiv.org/abs/2310.14735)
- (*arXiv 2023.04*) Boosted Prompt Ensembles for Large Language Models [[paper]](https://arxiv.org/abs/2304.05970)[[code]](https://github.com/awwang10/llmpromptboosting)
- (*NeurIPS'23*) Optimizing Prompts for Text-to-Image Generation [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/d346d91999074dd8d6073d4c3b13733b-Paper-Conference.pdf)[[code]](https://github.com/microsoft/LMOps)
- (*CVPR'23*) VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_VoP_Text-Video_Co-Operative_Prompt_Tuning_for_Cross-Modal_Retrieval_CVPR_2023_paper.pdf)[[code]](https://github.com/bighuang624/vop)
- (*ICCV'23*) Tune-a-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Tune-A-Video_One-Shot_Tuning_of_Image_Diffusion_Models_for_Text-to-Video_Generation_ICCV_2023_paper.pdf)[[code]](https://github.com/showlab/Tune-A-Video)
- (*CVPR'22*) Image Segmentation Using Text and Image Prompts [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Luddecke_Image_Segmentation_Using_Text_and_Image_Prompts_CVPR_2022_paper.pdf)[[code]](https://github.com/timojl/clipseg)
- (*ACM Computing Surveys'23*) Pre-Train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing [[paper]](https://dl.acm.org/doi/pdf/10.1145/3560815)[[code]](https://github.com/pfliu-nlp/NLPedia-Pretrain)
- (*EMNLP'21*) The Power of Scale for Parameter-Efficient Prompt Tuning [[paper]](https://aclanthology.org/2021.emnlp-main.243/)[[code]](https://github.com/google-research/prompt-tuning)
#### Trustworthiness
- (*arXiv 2024.02*) Jailbreaking Attack Against Multimodal Large Language Models [[paper]](https://arxiv.org/abs/2402.02309)[[code]](https://github.com/abc03570128/jailbreaking-attack-against-multimodal-large-language-model)
- (*arXiv 2023.09*) A Survey of Hallucination in Large Foundation Models [[paper]](link-to-paper)[[code]](https://github.com/vr25/hallucination-foundation-model-survey)
- (*arXiv 2024.01*) TrustLLM: Trustworthiness in Large Language Models [[paper]](https://arxiv.org/abs/2401.05561)[[code]](https://github.com/HowieHwong/TrustLLM)
- (*ICLR'24*) AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models [[paper]](https://arxiv.org/abs/2310.04451)[[code]](https://github.com/sheltonliu-n/autodan)
- (*NeurIPS'23*) DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models [[paper]](https://arxiv.org/abs//2306.11698)[[code]](https://github.com/AI-secure/DecodingTrust)
- Jailbroken: How Does LLM Safety Training Fail? [[paper]](link-to-paper)[[code]](link-to-code)
- (*arXiv 2023.10*) HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination & Visual Illusion in Large Vision-Language Models [[paper]](https://www.researchgate.net/profile/Fuxiao-Liu-2/publication/376072740_HALLUSIONBENCH_An_Advanced_Diagnostic_Suite_for_Entangled_Language_Hallucination_Visual_Illusion_in_Large_Vision-Language_Models/links/6568af0e3fa26f66f43abf17/HALLUSIONBENCH-An-Advanced-Diagnostic-Suite-for-Entangled-Language-Hallucination-Visual-Illusion-in-Large-Vision-Language-Models.pdf)[[code]](https://github.com/tianyi-lab/hallusionbench)
- (*arXiv 2023.09*) Bias and Fairness in Large Language Models: A Survey [[paper]](https://arxiv.org/abs/2309.00770)[[code]](https://github.com/i-gallegos/fair-llm-benchmark)
- (*arXiv 2023.02*) Fair Diffusion: Instructing Text-to-Image Generation Models on Fairness [[paper]](https://arxiv.org/abs/2302.10893)[[code]](https://github.com/ml-research/fair-diffusion)


### Application

#### Movie
- (*arXiv 2023.06*) MovieFactory: Automatic Movie Creation from Text Using Large Generative Models for Language and Images [[paper]](https://arxiv.org/abs/2306.07257) 
- (*ACM Multimedia'23*) MobileVidFactory: Automatic Diffusion-Based Social Media Video Generation for Mobile Devices from Text [[paper]](https://dl.acm.org/doi/abs/10.1145/3581783.3612667)
- (*arXiv 2024.01*) Vlogger: Make Your Dream A Vlog [[paper]](https://arxiv.org/abs/2401.09414) [[code]](https://github.com/Vchitect/Vlogger)

#### Education
- (*arXiv 2023.09*) CCEdit: Creative and Controllable Video Editing via Diffusion Models [[paper]](https://arxiv.org/abs/2309.16496) [[code]](https://github.com/RuoyuFeng/CCEdit)
- (*TVCG'24*) Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance [[paper]](https://arxiv.org/abs/2306.00943) [[code]](https://github.com/AILab-CVC/Make-Your-Video)
- (*ICLR'24*) AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models Without Specific Tuning [[paper]](https://arxiv.org/abs/2307.04725) [[code]](https://github.com/guoyww/animatediff/)
- (*arXiv 2023.07*) Animate-a-Story: Storytelling with Retrieval-Augmented Video Generation [[paper]](https://arxiv.org/abs/2307.06940) [[code]](https://github.com/AILab-CVC/Animate-A-Story)
- (*CVPR'23*) Conditional Image-to-Video Generation with Latent Flow Diffusion Models [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.html) [[code]](https://github.com/nihaomiao/CVPR23_LFDM)
- (*arXiv 2023.11*) Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation [[paper]](https://arxiv.org/abs/2311.17117) [[code]](https://github.com/HumanAIGC/AnimateAnyone)
- (*CVPR'22*) Make It Move: Controllable Image-to-Video Generation with Text Descriptions [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Make_It_Move_Controllable_Image-to-Video_Generation_With_Text_Descriptions_CVPR_2022_paper.html) [[code]](https://github.com/Youncy-Hu/MAGE)
#### Gaming
- (*AAAI'23*) VIDM: Video Implicit Diffusion Models [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26094) [[code]](https://github.com/MKFMIKU/VIDM)
- (*CVPR'23*) Video Probabilistic Diffusion Models in Projected Latent Space [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Video_Probabilistic_Diffusion_Models_in_Projected_Latent_Space_CVPR_2023_paper.html) [[code]](https://github.com/sihyun-yu/PVDM)
- (*CVPR'23*) Physics-Driven Diffusion Models for Impact Sound Synthesis from Videos [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Su_Physics-Driven_Diffusion_Models_for_Impact_Sound_Synthesis_From_Videos_CVPR_2023_paper.html) [[code]](https://github.com/sukun1045/video-physics-sound-diffusion)
- (*arXiv 2024.01*) Dance-to-Music Generation with Encoder-Based Textual Inversion of Diffusion Models [[paper]](https://arxiv.org/abs/2401.17800)
#### Healthcare
- (*bioRxiv 2023.11*) Video Diffusion Models for the Apoptosis Forecasting [[paper]](https://www.biorxiv.org/content/10.1101/2023.11.16.567461v1)
- (*PRIME'23*) DermoSegDiff: A Boundary-Aware Segmentation Diffusion Model for Skin Lesion Delineation [[paper]](https://arxiv.org/abs/2308.02959) [[code]](https://github.com/xmindflow/DermoSegDiff)
- (*ICCV'23*) Multimodal Motion Conditioned Diffusion Model for Skeleton-Based Video Anomaly Detection [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Flaborea_Multimodal_Motion_Conditioned_Diffusion_Model_for_Skeleton-based_Video_Anomaly_Detection_ICCV_2023_paper.html) [[code]](https://github.com/aleflabo/MoCoDAD)
- (*arXiv 2023.01*) MedSegDiff-V2: Diffusion Based Medical Image Segmentation with Transformer [[paper]](https://arxiv.org/abs/2301.11798) [[code]](https://github.com/kidswithtokens/medsegdiff)
- (*MICCAI'23*) Diffusion Transformer U-Net for Medical Image Segmentation [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_59)
#### Robotics
- (*IEEE RA-L'23*) DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics [[paper]](https://arxiv.org/abs/2210.02438)
- (*CoRL'22*) StructDiffusion: Object-Centric Diffusion for Semantic Rearrangement of Novel Objects [[paper]](https://arxiv.org/abs/2211.04604)[[code]](https://github.com/StructDiffusion/StructDiffusion)
- (*arXiv 2022.05*) Planning with Diffusion for Flexible Behavior Synthesis [[paper]](https://arxiv.org/abs/2205.09991)[[code]](https://github.com/jannerm/diffuser)
- (*arXiv 2022.11*) Is Conditional Generative Modeling All You Need for Decision-Making? [[paper]](https://arxiv.org/abs/2211.15657)[[code]](https://github.com/zbzhu99/decision-diffuser-jax?tab=readme-ov-file)
- (*IROS'23*) Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models [[paper]](https://arxiv.org/abs/2308.01557)[[code]](https://github.com/jacarvalho/mpd-public)
- (*ICLR'24*) Seer: Language Instructed Video Prediction with Latent Diffusion Models [[paper]](https://arxiv.org/abs/2303.14897)[[code]](https://github.com/seervideodiffusion/SeerVideoLDM)
- (*arXiv 2023.02*) GenAug: Retargeting Behaviors to Unseen Situations via Generative Augmentation [[paper]](https://arxiv.org/abs/2302.06671)[[code]](https://github.com/genaug/genaug)
- (*arXiv 2022.12*) CACTI: A Framework for Scalable Multi-Task Multi-Scene Visual Imitation Learning [[paper]](https://arxiv.org/abs/2212.05711)[[code]](https://github.com/Cacti/cacti)

## Citation

```
@misc{2024SoraReview,
      title={Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models}, 
      author={Yixin Liu and Kai Zhang and Yuan Li and Zhiling Yan and Chujie Gao and Ruoxi Chen and Zhengqing Yuan and Yue Huang and Hanchi Sun and Jianfeng Gao and Lifang He and Lichao Sun},
      year={2024},
      eprint={2402.17177},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

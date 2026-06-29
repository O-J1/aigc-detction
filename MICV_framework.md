# MICV - AIGC Detection Framework using DINOv3 ensemble
Overview In this work, we propose a robust ensemble based framework for AI-generated image detection, specifi-
cally designed to address the challenges of domain generalization and cross-platform detection. 

Our approach centers on three core pillars:
• Hierarchical Data Strategy: We curate a large-scale, multi-source training corpus that integrates open-source academic benchmarks, synthetic images from cutting edge generative models, and high-fidelity samples from closed-source commercial APIs, ensuring comprehensive coverage of diverse generative artifacts.
• Ensemble-based Architecture: We utilize the powerful representation capabilities of multiple DINOv3 backbones, organized into two distinct model committees. By employing a late-fusion strategy that averages the detection probabilities derived from these ensembles, we achieve a more holistic and discriminative signature of AI-generated content.

• Robust Augmentation: To enhance robustness against “in-the-wild” degradations, we implement a hierarchical, difficulty-aware data augmentation pipeline. Combined with a Focal Loss-driven optimization and Stochastic Weight Averaging (SWA), our approach effectively bridges the distribution gap between laboratory benchmarks and real-world unconstrained imagery.


## Data Collection 
High-quality, large-scale, and diverse datasets are fundamental to the robustness and generalizability of AI-Generated Image detection. To effectively emulate the complex, “in-the-wild” distribution of generative artifacts, we curate a comprehensive training corpus comprising millions of samples. Our data acquisition strategy is hierarchical, categorized into four primary tiers: 

• Open-Source Datasets: To establish a solid foundation for cross-domain generalization, we integrate a diverse array of open-source resources, ranging from established academic benchmarks to large-scale repositories hosted on platforms like HuggingFace. This collection includes, but is not limited to, GenImage, WildFake, AIGIBench, CommunityForensics, and So-Fake-Set, along with representative datasets sourced from open-source communities. By aggregating these heterogeneous data sources, we ensure that our model is exposed to a wide spectrum of generation paradigms and realworld artifacts, significantly enhancing the scalability and effectiveness of our model.

• Open-Source Generative Models: To align with the rapid evolution of generative architectures, we construct a substantial synthetic dataset using state-of-the-art open source models. Our pipeline encompasses a multi-faceted range of generation tasks, including Text-to-Image (T2I), Image-to-Image (I2I), image editing and in-painting, leveraging representative models such as Qwen-Image, Z-Image, and the FLUX series to capture the distinct artifacts inherent in contemporary architectures.

• Closed-Source Commercial Models: Given the prevalence of closedsource platforms in real-world applications, we supplement our corpus with high-fidelity samples obtained via official APIs. By integrating outputs from industry-leading engines such as Seedream, Kling, GPT-Image, and Nano-banana-pro, we effectively mitigate the distribution shift between open-source research models and proprietary commercial systems, thereby enhancing the detector’s practical deployment efficacy.

• Challenge-Specific Datasets: To further expand our set of
training samples, we utilized the image samples provided
by the competition organizers.

## Methodology 

As illustrated in Figure 1, we propose a feature fusion architecture for AI generated image detection. Our approach leverages the powerful feature representation capabilities of DINOv3 backbones by employing two distinct subnetworks, each designed to process image features through an ensemble of pretrained backbones. Specifically, the architecture comprises two independent streams: the first stream aggregates feature maps from a committee of four DINOv3 backbones, while the second stream integrates features from a separate committee of two DINOv3 backbones. Within each sub-network, the aggregated backbone features are processed through a dedicated projection layer to map them into a latent space, followed by a multi-layer perceptron (MLP) head that produces the detection probability. To derive the final prediction, we average output probabilities from both streams.

To bridge the distribution shift between controlled benchmarks and challenging “in-the-wild” imagery, we design a hierarchical, stochastic data augmentation pipeline structured by difficulty levels. Our pipeline progresses from simple, individual transformations to complex combina-
torial perturbations. While the former applies individual degradations such as blur, noise, geometric shifts or compression, the latter employs multi-stage pipelines that sim-
ulate complex degradations. This hierarchical and multi-faceted augmentation strategy effectively narrows the domain gap, ensuring superior detection performance even in highly unconstrained and degraded environments. 

## Implementation Details

We initialize our framework using pretrained DINOv3 backbones and perform end-to-end
fine-tuning. During training, images are randomly cropped and resized to 512 × 512 pixels, supplemented by our hierarchical augmentation pipeline. During inference, images are directly resized to 512 × 512 to preserve the global spatial context. The model is trained on 32 NVIDIA A100 GPUs for 10 epochs, with the entire procedure completing in approximately 8 hours.

• Objective Function: We employ Focal Loss as our primary objective function to address the potential imbalance in sample difficulty and to mitigate the dominance of easy-negative samples. The focus parameter γ is set to 2.0, and the balance parameter α is empirically set to 0.5, to ensure that the model focuses on hard-to-classify generative artifacts.

• Optimization Strategy: We utilize the AdamW optimizer with a weight decay of 0.02. The learning rate is initialized at 1×10-5. To ensure training stability, we implement a linear warmup strategy over the first epoch, followed by a Cosine Annealing schedule to gradually decay the learning rate for the remainder of the training process. We employ Stochastic Weight Averaging (SWA) over the final epochs to aggregate model weights. This refinement yields a more stable and generalized weight configuration, which serves as our final inference model. 

• Evaluation and Refinement: We evaluate model performance using the Area Under the Receiver Operating Characteristic curve (ROC AUC) on a dedicated validation set, which is curated by sampling 10,000 label balanced images from the official training corpus. To ensure a robust assessment under challenging conditions, we apply a static version of the hierarchical data augmentation pipeline to this validation set to assess robustness.
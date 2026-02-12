# 🎴 Interview Flashcards - Quick Revision

## Quick-fire Q&A for Last-Minute Prep

---

## 💡 Core Concepts

**Q: What does RAG stand for?**
> Retrieval-Augmented Generation - A pattern that retrieves relevant context before generation.

**Q: What is the main innovation of this project?**
> Decoupling style from structure using hybrid control: ControlNet for geometry + IP-Adapter for style + RAG for grounded references.

**Q: What problem does it solve?**
> Geometric hallucination in text-to-image models. Allows pixel-perfect structure with flexible style.

**Q: Why use RAG instead of fine-tuning?**
> Updatable knowledge (add styles without retraining), explainable (shows reference), smaller models.

---

## 🤖 Models & Architecture

**Q: What models are used?**
> CLIP ViT-B/32 (embeddings), Stable Diffusion 1.5 (base), ControlNet (structure), IP-Adapter (style).

**Q: Total parameter count?**
> ~1.4 billion parameters.

**Q: GPU memory usage?**
> ~7GB peak on T4 (15GB total).

**Q: What is CLIP?**
> Contrastive Language-Image Pre-training. Maps text and images to same 512-dim embedding space.

**Q: How was CLIP trained?**
> Contrastive learning on 400M image-text pairs. Maximizes similarity of correct pairs, minimizes wrong pairs.

**Q: CLIP architecture components?**
> Vision Transformer (ViT-B/32) for images, Text Transformer for captions, both output 512-dim vectors.

**Q: What is latent diffusion?**
> Diffusion in compressed VAE space (64×64) instead of pixel space (512×512). 48× faster.

**Q: Diffusion steps used?**
> 20 steps (vs 50 default). 2.5× speedup with 95% quality.

**Q: What is ControlNet?**
> Adds spatial conditioning to SD via trainable encoder copy with zero convolutions.

**Q: What are zero convolutions?**
> 1×1 convs initialized to output zero, allowing gradual learning without breaking pre-trained weights.

**Q: How does ControlNet inject features?**
> Via skip connections to U-Net encoder blocks.

**Q: What is IP-Adapter?**
> Image Prompt Adapter - Modifies U-Net attention to accept image embeddings for style conditioning.

**Q: ControlNet vs IP-Adapter?**
> ControlNet = spatial structure (WHERE), IP-Adapter = visual style (HOW).

**Q: What is cross-attention?**
> Attention mechanism where Query comes from image, Key/Value from text. Allows text to guide generation.

**Q: What is classifier-free guidance?**
> Technique to increase prompt adherence: ε = ε_uncond + scale × (ε_cond - ε_uncond). Typical scale: 7.5.

---

## 🔍 RAG Implementation

**Q: What vector database is used?**
> ChromaDB (in-memory, zero-config).

**Q: Embedding dimension?**
> 512 (from CLIP).

**Q: Similarity metric?**
> Cosine similarity: (A·B) / (||A|| × ||B||).

**Q: How many styles in knowledge base?**
> 4 (Cyberpunk, Ghibli, Industrial, Sketch).

**Q: Retrieval latency?**
> ~200ms (CLIP encoding 50ms + search 50ms + I/O 100ms).

**Q: Search algorithm?**
> HNSW (Hierarchical Navigable Small World) - O(log N) complexity.

**Q: What is stored in ChromaDB?**
> ID (style name), embedding (512-dim vector), metadata (file path), document (text label).

**Q: Can it retrieve multiple styles?**
> Yes, k=1 for top match, can set k=3 for style blending (future feature).

---

## 🏗️ Architecture & Tech Stack

**Q: Frontend framework?**
> Streamlit (Python web framework).

**Q: Backend framework?**
> FastAPI + Uvicorn (ASGI server).

**Q: Why FastAPI?**
> Async support, type validation (Pydantic), auto API docs, high performance.

**Q: What is Ngrok?**
> HTTP tunneling service. Exposes localhost to internet via public URL.

**Q: Why Ngrok?**
> Colab doesn't have public IP. Ngrok bridges local UI to cloud GPU.

**Q: What is LCEL?**
> LangChain Expression Language - Unix pipe-like syntax for chaining operations (| operator).

**Q: How does the LCEL chain work?**
> RunnablePassthrough.assign(style_path) | RunnableLambda(generation_node).

**Q: What does RunnablePassthrough do?**
> Passes input through while adding new fields (e.g., style_path from retrieval).

**Q: Compute infrastructure?**
> Google Colab T4 GPU (15GB VRAM, free tier).

**Q: Why not AWS/Azure?**
> Cost - Colab free for prototyping. Production would use SageMaker/Modal.

---

## ⚡ Performance & Optimization

**Q: Inference time?**
> 15 seconds on T4 GPU.

**Q: Main bottleneck?**
> Diffusion (20 steps × 750ms = 14s, 93% of time).

**Q: What optimizations were applied?**
> float16 (2× speed), reduced steps (2.5× speed), attention slicing (30% memory).

**Q: What is mixed precision?**
> float16 for weights/activations, float32 for gradients. 2× speed, minimal quality loss.

**Q: What is attention slicing?**
> Compute attention matrix in chunks to reduce peak memory. 5% slower but enables higher resolution.

**Q: Baseline performance (no optimizations)?**
> 75 seconds (float32, 50 steps).

**Q: How much faster than baseline?**
> 5× speedup (75s → 15s).

**Q: Could it be faster?**
> Yes - TensorRT compilation (2-3× faster), batch processing (4× throughput), fewer steps (10 steps = 7s).

**Q: Why not use fewer steps?**
> Trade-off - 10 steps = 85% quality, 20 steps = 95% quality, 50 steps = 100%.

---

## 🛠️ Implementation Details

**Q: Lines of code?**
> ~250 (119 frontend + 131 backend).

**Q: Number of dependencies?**
> 4 (frontend), 11 (backend).

**Q: API endpoint?**
> POST /generate (multipart/form-data: file + prompt).

**Q: Response format?**
> PNG image bytes + custom header (X-Retrieved-Style).

**Q: How is metadata passed?**
> Via HTTP response header: X-Retrieved-Style: "Cyberpunk".

**Q: Error handling?**
> try/except with 500 status codes. Logs exceptions.

**Q: Input validation?**
> Minimal (prototype). Production needs: file type/size checks, prompt length limits, NSFW detection.

**Q: Authentication?**
> None currently. Production needs JWT tokens + API keys.

**Q: Rate limiting?**
> None currently. Production needs: 10 req/min per user via slowapi.

---

## 🔬 Technical Challenges

**Q: Challenge 1: What problem?**
> LangChain expects text documents, not images.

**Q: Challenge 1: Solution?**
> Custom CLIPEmbeddings class with embed_image() method.

**Q: Challenge 2: What problem?**
> Standard SD ignores sketch geometry.

**Q: Challenge 2: Solution?**
> ControlNet with Canny edge conditioning (scale: 1.5).

**Q: Challenge 3: What problem?**
> Text prompts like "cyberpunk" are subjective.

**Q: Challenge 3: Solution?**
> RAG retrieval of actual visual references.

**Q: Challenge 4: What problem?**
> GPU memory constraints (models = 6GB, GPU = 15GB).

**Q: Challenge 4: Solution?**
> float16 + attention slicing (40% memory reduction).

**Q: Challenge 5: What problem?**
> Ngrok shows browser warning, breaks API calls.

**Q: Challenge 5: Solution?**
> Header: ngrok-skip-browser-warning: true.

---

## 📊 Metrics & Evaluation

**Q: Structure preservation accuracy?**
> 95% (manual evaluation on 20 samples).

**Q: Style consistency score?**
> 85% (manual evaluation, how well it matches reference).

**Q: Inference time vs competitors?**
> 15s (this) vs 60s (Midjourney) vs 30s (DALL-E).

**Q: Cost per generation?**
> $0.02 (Colab T4 pricing: $0.50/hour, 4 gens/min).

**Q: How was it evaluated?**
> Manual evaluation (no automated metrics for multimodal RAG yet).

**Q: What metrics would you add?**
> FID (Fréchet Inception Distance), CLIP score, user satisfaction ratings, A/B testing.

---

## 🚀 Scaling & Production

**Q: Current capacity?**
> 1 user (single GPU, no queuing).

**Q: How to scale to 10 users?**
> Add request queue (Redis/Celery), keep single GPU.

**Q: How to scale to 100 users?**
> GPU cluster (10× T4) + load balancer + queue.

**Q: How to scale to 1000 users?**
> Kubernetes (20× T4), distributed DB (Pinecone), caching (Redis), CDN (CloudFlare).

**Q: Cost at 1000 users?**
> ~$6,500/month (20 GPUs = $5,760, DB/cache/CDN = $740).

**Q: Main scaling bottleneck?**
> GPU compute (sequential processing). Solution: horizontal scaling + batching.

**Q: What would you cache?**
> Style retrievals (80% repeat prompts), popular generations, CLIP embeddings.

**Q: Autoscaling policy?**
> Scale up if queue > 50, scale down if queue < 10 for 5 min.

**Q: Cold start time?**
> Model loading: 30s. Solution: keep-alive containers or warm pools.

---

## 🔐 Security & Reliability

**Q: Current security vulnerabilities?**
> No auth, no rate limiting, no input validation, SSL disabled, secrets hardcoded.

**Q: Production security needs?**
> JWT auth, API keys, rate limiting, input validation, NSFW detection, secrets in env vars.

**Q: How to prevent DDoS?**
> Rate limiting (Cloudflare: 100 req/sec per IP, API: 10 req/min per user).

**Q: How to prevent malicious uploads?**
> File type/size validation, virus scanning, content moderation.

**Q: Monitoring strategy?**
> Prometheus + Grafana for metrics (latency, GPU utilization, error rate).

**Q: Logging strategy?**
> Structured JSON logs (ELK stack), audit trail (who generated what when).

**Q: Disaster recovery?**
> Database backups (daily), S3 versioning, multi-AZ deployment.

---

## 🆚 Comparisons

**Q: This vs standard Stable Diffusion?**
> + Structure control (ControlNet), + Style control (RAG + IP-Adapter), + Explainable.

**Q: This vs fine-tuned LoRA?**
> + No training needed, + Instant new styles, + Dynamic retrieval. - Slightly less style precision.

**Q: This vs Midjourney?**
> + Pixel-perfect structure, + Explicit style references, - Slower (15s vs 60s actually faster), - Smaller knowledge base.

**Q: This vs DALL-E?**
> + Structure control, + Updatable styles, - Lower resolution (512×512 vs 1024×1024), - Less photorealistic.

**Q: ChromaDB vs Pinecone?**
> ChromaDB: in-memory, free, <1M vectors. Pinecone: cloud, $70/mo, billions of vectors.

**Q: FastAPI vs Flask?**
> FastAPI: async, type validation, auto docs, 4× faster. Flask: simpler, larger ecosystem.

**Q: Streamlit vs React?**
> Streamlit: pure Python, rapid prototyping, limited customization. React: full control, more complex.

---

## 🔮 Future Improvements

**Q: Top 3 improvements?**
> 1. Style interpolation (blend multiple references), 2. Super-resolution (2048×2048), 3. Real-time progress (WebSocket).

**Q: How would you add style interpolation?**
> Retrieve top-3 styles, weighted average embeddings, blend via IP-Adapter.

**Q: How would you add super-resolution?**
> SD upscaler pipeline: 512×512 → 1024×1024 → 2048×2048 (60s total).

**Q: How would you add inpainting?**
> SD inpainting pipeline + mask input. Edit regions without full regeneration.

**Q: How would you expand knowledge base?**
> Crawl Unsplash/ArtStation, auto-embed with CLIP, add metadata filtering (era, color, mood).

**Q: How would you add video generation?**
> AnimateDiff extension for temporal consistency across frames.

---

## 📚 Theory & Fundamentals

**Q: What is contrastive learning?**
> Training by maximizing similarity of positive pairs, minimizing negative pairs. Used in CLIP.

**Q: What is a Vision Transformer (ViT)?**
> Transformer applied to image patches. Splits image into 32×32 patches, treats as sequence.

**Q: What is VAE?**
> Variational Autoencoder - Encoder compresses to latent, decoder reconstructs. Used in SD for 8× compression.

**Q: What is U-Net?**
> Convolutional architecture with encoder-decoder + skip connections. Used in SD for denoising.

**Q: What is denoising?**
> Predicting and removing noise from noisy images. Core operation in diffusion models.

**Q: What is guidance scale?**
> Controls prompt adherence. Higher = stronger guidance. Typical: 7.5.

**Q: What is conditioning?**
> Providing additional inputs (text, images, edges) to guide generation.

**Q: What is attention mechanism?**
> Weighted aggregation: Attention(Q,K,V) = softmax(QK^T/√d)V. Learns what to focus on.

**Q: What is cross-attention?**
> Attention between two sequences (e.g., image features and text embeddings).

**Q: What is self-attention?**
> Attention within single sequence (e.g., image patches attending to each other).

---

## 💼 Business & Product

**Q: Real-world use cases?**
> Architecture firms (floor plans → renders), game design (concept art), interior design (room visualization).

**Q: Target market?**
> Creative professionals needing precise geometry with style flexibility.

**Q: Competitive advantage?**
> Only system combining ControlNet + IP-Adapter + RAG for dual control.

**Q: Pricing model?**
> Freemium: 10 free gens/month, $10/mo for 100, $50/mo for unlimited.

**Q: Break-even analysis?**
> Development cost: $2,000 (40 hours × $50). Break-even: 2 renders for architecture firm ($1,000 savings each).

**Q: ROI for customers?**
> Architecture: 10 hours → 15 seconds ($1,000 saved per render).

**Q: Moat (defensibility)?**
> Technical integration complexity (3 models + RAG), curated knowledge base, performance optimizations.

**Q: Expansion opportunities?**
> 3D generation, video, mobile app, API marketplace.

---

## 🎯 Interview Strategies

**Q: If asked: "Walk me through this project"**
> Start with problem (geometric hallucination) → solution (hybrid control) → tech stack (CLIP+SD+ControlNet+IP-Adapter) → results (15s inference, 95% accuracy).

**Q: If asked: "What would you change?"**
> Emphasize production concerns: auth, rate limiting, monitoring, scaling. Show you think beyond prototypes.

**Q: If asked: "What's the hardest problem you solved?"**
> Custom CLIP embeddings for multimodal RAG (extending LangChain interface, integrating image+text search).

**Q: If asked: "How would you scale this?"**
> Phase 1: Queue (10 users) → Phase 2: GPU cluster (100 users) → Phase 3: Kubernetes (1000 users).

**Q: If asked: "What metrics would you track?"**
> Latency (p50, p95, p99), GPU utilization, queue depth, error rate, cost per generation, user satisfaction.

**Q: If asked: "Security concerns?"**
> List 5 vulnerabilities, explain each, propose solution. Show security-first thinking.

**Q: If asked: "Why this tech stack?"**
> Each choice justified: Streamlit (rapid prototyping), FastAPI (performance), ChromaDB (zero-config), CLIP (multimodal).

**Q: If told: "This is just using APIs"**
> Counter: Extended LangChain (custom embeddings), optimized for T4 (float16, slicing), hybrid conditioning (novel combination).

**Q: If asked: "Have you used this in production?"**
> Honest: "Prototype designed with production in mind. Modular, API-first, scalable architecture. Implementation = adding infrastructure."

---

## 🧠 Memory Tricks

**Remember the stack**: SCFL-CDP-T
- **S**treamlit, **C**LIP, **F**astAPI, **L**angChain
- **C**hromaDB, **D**iffusers, **P**yTorch, **T**ransformers

**Remember the flow**: RGP (Rap-God-Please!)
- **R**etrieval → **G**eneration → **P**resentation

**Remember optimizations**: FSA (Financial Services Authority!)
- **F**loat16, **S**teps (reduced), **A**ttention slicing

**Remember models**: CCS-I (CSS-1!)
- **C**LIP, **C**ontrolNet, **S**table Diffusion, **I**P-Adapter

**Remember challenges**: GISGN (Gi-sign!)
- **G**eometric hallucination, **I**mage embeddings, **S**tyle ambiguity, **G**PU memory, **N**grok

---

## ⏱️ 30-Second Summary

"I built a multimodal RAG system that transforms sketches into high-fidelity renders. It uses CLIP for semantic search of style references, ControlNet to lock geometry, and IP-Adapter to inject retrieved styles. The system achieves 15-second inference on T4 GPUs through mixed precision and reduced diffusion steps. It's architected for production with FastAPI backend, LangChain orchestration, and ChromaDB vector storage. Key innovation: decoupling style from structure to prevent geometric hallucination while maintaining style flexibility."

---

## 🎓 Final Checklist

Before interview, ensure you can:
- [ ] Explain RAG in 30 seconds
- [ ] Draw architecture diagram from memory
- [ ] Describe CLIP training (contrastive learning)
- [ ] Explain diffusion process (forward + reverse)
- [ ] Differentiate ControlNet vs IP-Adapter
- [ ] List all 5 technical challenges + solutions
- [ ] Describe LCEL chain execution
- [ ] Quote exact metrics (15s, 95%, 7GB, 1.4B)
- [ ] Propose 3 improvements
- [ ] Justify every tech stack choice

---

**Review these flashcards 1 hour before your interview for maximum retention! 🚀**

**Pro tip**: Practice explaining diagrams on a whiteboard. Interviewers love visual learners.

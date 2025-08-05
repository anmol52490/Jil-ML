# Machine Learning Portfolio

### **Profile Summary**
Anmol is a final-year B.Tech student in Computer Science (Data Science) whose work stretches from classical models Linear/Logistic Regression, SVM, XGBoost to CNNs, RNN/LSTMs and transformer-based Document-AI. As one of the first ML hires at HiDevs he shipped search, content and workflow automations that helped the product land Product Hunt #7 Product of the Day. Recent projects include re-balancing the DocLayNet corpus and fine-tuning HuggingFace LayoutLMv3 to extract document hierarchies, plus LLM-driven multi-agent pipelines in production. Everything runs on a stack that fuses PyTorch and TensorFlow under the hood, orchestrated by LangChain agents and served through lean FastAPI endpoints. 

-----


### **I. Experience**

**Gen AI Developer Intern** | HiDevs, Bengaluru, India
*December 2024 – June 2025*

* **Dave (formerly EchoDeepak): Personalised EdTech Assistant**
  Utilised Gemini 1.5 Flash with LangChain to deliver context-aware tutoring, then routed every reply through an LLM-judge scoring layer for relevance, hallucination and bias. Added a diagram generator that lets the model write vetted Python, runs it in a sandbox and streams the image back over FastAPI, so students get visuals as well as text. Session memory lives in MongoDB and median latency stays near 150 ms on standard cloud instances.

* **Lead-Generation Multi-Agent Pipeline**
  Orchestrated four CrewAI agents—activity scorer, profile summariser, activity summariser and vision-alignment checker—to analyse LinkedIn data and output a ready-to-contact CSV. Automation removed roughly 80 percent of manual prospecting and lifted the share of qualified leads by **45 percent**, while a strict token budget keeps per-prospect runtime under five seconds.

* **LinkedIn Post Generator**
  Built an agentic workflow that turns user achievements inside HiDevs into brand-consistent LinkedIn posts. Introduced an evaluation-and-rewrite feedback loop that raised human approval from 62 percent to **90 percent** and cut generation time from about two minutes to roughly 30 seconds, all served through the same FastAPI backbone for unified logging and monitoring.

* **Inbox Assistant: Email Operations Automation**
  Assembled an n8n-driven chatbot that links Gmail APIs with Gemini and SimpleMemory to summarise incoming mail, prioritise threads and draft context-aware replies from a single chat pane. The tool tames a high-volume inbox by trimming daily triage from roughly 40 minutes to under five and keeps turnaround under a minute for priority clients.

-----

### **II. Machine-Learning Projects**

* **Emotion Detector** *(real-time facial sentiment analysis)* – a webcam pipeline that crops faces with OpenCV, feeds them to a CNN trained from scratch on the FER-2013 benchmark (28 709 × 48 px greyscale images across seven emotions), applies flip/zoom/normalise augmentation to curb over-fitting and streams predictions live on CPU-only hardware. [GitHub Repo](https://github.com/Med-Time/Emotion-Detector)

* **LayoutLMv3 Heading Detector** *(document-AI hierarchy extraction)* – cleaned and re-labelled the 80 k-page DocLayNet corpus, converted it to Hugging Face `datasets`, then fine-tuned `layoutlmv3-base` to tag H1–H3 tokens in noisy PDFs, outperforming baseline layout taggers while keeping the model exportable for downstream parsing tasks. [GitHub Repo](https://github.com/anmol52490/Finetune-Layoutlmv3-base)

* **Bank Churn Prediction** *(classical ML for customer retention)* – ingested the Kaggle bank-customer dataset of \~10 000 rows, engineered categorical one-hots and balance logs, compared Logistic Regression, Random Forest and XGBoost with cross-validation, and wrapped the best estimator behind a lightweight REST endpoint so analysts can flag at-risk accounts early. [GitHub Repo](https://github.com/anmol52490/Bank_Churn_Prediction)

* **BDA with PySpark** *(distributed NLP workflow)* – partitioned the 20 Newsgroups collection across Spark DataFrames, tokenised and TF-IDF-vectorised posts, ran topic-frequency analysis and generated word clouds and length histograms to show how the same code scales from a laptop to a multi-node cluster with only configuration tweaks. [GitHub Repo](https://github.com/Med-Time/BDA-with-PySpark)

* **SpamHam Responsible-AI Classifier** *(interpretable email filtering)* – trained a Random Forest on the UCI SMS Spam Collection, then applied LIME explanations and Responsible-AI diagnostics to surface token importance, error slices and bias metrics, exporting both a PDF audit and an interactive HTML dashboard for governance review. [GitHub Repo](https://github.com/Med-Time/SpamHam-Classifier-ResponsibleAI)

* **CodeSense** *(multi-agent GitHub pull-request reviewer)* – orchestrates specialised LLM agents that scan diffs for security smells, style violations and potential bugs, posts structured markdown feedback inside the PR thread and caches embeddings to keep response times predictable even on large repositories; the modular design makes it easy to bolt on new review dimensions. [GitHub Repo](https://github.com/Med-Time/CodeSense)


-----


### Research & Publications — In Progress
**A LangGraph-Driven Knowledge-Assessment System for Lesson-Plan Generation** (manuscript in preparation) presents an agent-orchestrated tutor that begins with a diagnostic interview, scores each learner response through Gemini 1.5 Flash, infers a persona, and then iteratively composes and validates a personalised lesson plan; the end-to-end stack—FastAPI micro-services steering a LangGraph finite-state workflow over Qdrant and MongoDB—is open-sourced at [GitHub](https://github.com/Med-Time/Viveka), with the current paper draft available at [Drive](https://drive.google.com/drive/folders/1pn8arEtyn8ZvX-vbeM5i7VhQlePiAb_K?usp=sharing), and early evaluations across 50 + simulated sessions report 92 % precision on multiple-choice scoring, 85 % accuracy on open-ended answers, 95 % prerequisite-correct lesson sequencing, and a median end-to-end latency of 2.3 s, with submission to a peer-reviewed AI-in-education venue targeted for Q3 2025 .


-----


### Hackathon and Participations
* **Google Cloud Agentic AI Day Hackathon 2025 — City Pulse (Top 850 of 9 100+ teams, Guinness-certified as the world’s largest generative-AI hackathon)[GitHub](https://github.com/anmol52490/Google-Cloud-Agentic-AI-Day.git)**
City Pulse addressed real-time “city-data overload” by unifying social-media posts, citizen reports and IoT telemetry into a self-evolving urban-intelligence dashboard. Gemini multimodal agents enrich each raw signal with geotags, sentiment and timestamps; a Qdrant-backed vector matcher (cosine similarity > 0.5) deduplicates near-identical items before a LangGraph feedback loop forecasts cascading events such as grid instabilities or traffic jams. The end-to-end stack—Gemini APIs, LangChain / LangGraph orchestration, FastAPI micro-services and a Firebase-hosted Google Maps UI—streams live, predictive alerts while continually retraining on user corrections. Judges cited the architecture’s adaptive learning loop and city-scale applicability, ranking City Pulse in the top 850 teams (≈ 9 percentile) out of more than 9 100 nationwide submissions.


* **Adobe India Hackathon 2025  (Top 4 000 teams, about 4.6 percent acceptance)**
Roughly 262 159 developers signed up and, with a three-member cap, could form about 87 000 teams. The challenge asked Round 1 entrants to (a) generate an offline, CPU-only JSON outline of a PDF’s Title and H1–H3 hierarchy and (b) rank the most relevant sections across 3 to 10 PDFs for a given persona and job-to-be-done.
The proposed approach fine-tunes **LayoutLMv3-base** [Huggingface](https://huggingface.co/microsoft/layoutlmv3-base) on a heuristically relabelled slice of **DocLayNet** [Doclaynet](https://huggingface.co/datasets/pierreguillou/DocLayNet-base). Pre-processing scripts preserve heading order and create supervised labels, then the model is fine-tuned in the repository **Finetune-Layoutlmv3-base** [Finetuned Model](https://github.com/anmol52490/Finetune-Layoutlmv3-base). A companion repository, **Adobe-Hackathon** [1b](https://github.com/aaryanrn/Adobe-Hackathon), packages a Docker image that chunks each document, embeds sections, scores cosine similarity against persona queries and outputs an importance-ranked JSON. This concept secured a place in the top 4 000 teams invited to present.


### **III. Technical Skills**

  * **AI & ML:** PyTorch, TensorFlow, Transformers, Huggingface, Generative AI, RAG, LangChain, LangGraph,  CrewAI, Qdrant
  * **Programming:** Python
  * **Web & Databases:** FastAPI, Streamlit, MongoDB, MySQL
  * **Tools:** Git, Docker, Jenkins

-----

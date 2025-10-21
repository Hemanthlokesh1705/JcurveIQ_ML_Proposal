# JcurveIQ_ML_Proposal
# Financial Document QA & Risk Detection Pipeline

## 1) Problem Statement
Build a system that answers analyst queries about public companies by reading SEC filings, earnings calls, and PDFs — returning accurate, citation-anchored answers and flagging suspicious (“fishy”) accounting.  
**Top risks:** hallucination, wrong citations, missed evidence, and noisy flagging.

---

## 2) Data Sources & Collection

### Primary (authoritative)
- **EDGAR (SEC) filings:** 10-K (annual), 10-Q (quarterly), 8-K (events).  
  *Why:* Canonical financial disclosures with tables and discussion.
- **Earnings-call transcripts:** Management commentary and Q&A.  
  *Why:* Forward-looking guidance and tone.
- **Investor presentations / slides / PDFs:**  
  *Why:* Useful summaries and sometimes non-SEC disclosures.

### Supplemental
- Price & market data (e.g., Yahoo / AlphaVantage) for time-series anomaly detection.
- Expert-labeled QA pairs and “fishiness” labels (analyst annotations).
- Public datasets: FinQA, TAT-QA for span extraction & numerical reasoning.

### Collection Approach
- Build EDGAR crawler or use EDGAR API (bulk FTP) to download filings by CIK and date.
- Use licensed or public transcripts; run ASR (Whisper or commercial) for audio.
- Store raw docs and metadata in object storage (S3 / MinIO).
- Emit ingestion events (Kafka) to an indexing pipeline.

**Rationale:** EDGAR is authoritative. Combining transcripts and filings enables cross-checking (numbers vs commentary) for fishiness detection.

---

## 3) Data Cleaning & Preprocessing

### Goals
- Canonicalize text
- Preserve tables/figures
- Keep metadata for audit

### Steps
1. **PDF → structured text:** PyMuPDF (fitz) or pdfplumber; preserve page numbers, headers/footers, table bounding boxes.
2. **OCR fallback:** Tesseract / commercial OCR; correct numeric errors (e.g., “O” → “0”).
3. **Boilerplate removal:** Remove repeated headers/footers and SEC page stamps.
4. **Section tagging:** Regex/heuristics for “Risk Factors”, “MD&A”, “Notes to Financial Statements”, “Consolidated Balance Sheet”.
5. **Table extraction:** Tabula / pandas; store as structured objects and text.
6. **Normalization:** Canonicalize dates, names, numeric formats, currencies (keep raw values too).
7. **Indexable chunking:** 300–600 tokens with 20–30% overlap; include doc_id, page, start_char, end_char, section.
8. **Logging & audit:** Save raw + cleaned versions; maintain traceability.

**Why:** Preserves provenance, ensures numeric fields for anomaly detection, reduces noisy text that confuses retrieval.

---

## 4) Baseline Architecture & Models

**Chunking:** Simple overlapping splitter (300–500 tokens)  
**Retriever:**  
- Sparse: BM25 via Elasticsearch  
- Dense: sentence-transformers/all-MiniLM-L6-v2 → FAISS  
- Fusion: union top results from BM25 + dense retriever (deduplicate)

**Generator:** GPT-4 / GPT-4o API, strict prompt, cite chunk IDs, temperature=0.0  

**Audit Trail:** save retrieved chunk IDs, prompt text, model response, tokens used  

**Fishiness Detector (baseline):** rule-based heuristics: YoY thresholds, large goodwill changes, related-party keywords  

**Why Baseline:** Fast, explainable, cheap; BM25 catches numeric queries, MiniLM adds semantic coverage  

**Scalability:** Elasticsearch horizontal, FAISS IVF/PQ scales to millions of vectors, LLM via API scales by concurrency quotas  

---

## 5) Candidate Models

### A. Retrieval Improvements
- **Domain-specific embeddings:** FinBERT / FinDomain sentence-transformer; better semantic precision.
- **Hybrid retrieval:** BM25 + Dense + keyword expansions; merge top results for max recall.
- **Cross-encoder reranker:** Re-rank top-N candidates; improves precision@k; run on GPUs; distill later.

### B. Evidence Extraction & Citation
- Span-extractor / token-level QA (RoBERTa/FinBERT QA)  
- Extract exact supporting sentences → higher citation fidelity, smaller LLM context.  

### C. Generation Improvements
- Instruction-tuned / LoRA-fine-tuned generator (7B–13B) for financial QA + citations  
- Citation-aware decoding: enforce patterns like “Claim — [DOC_ID|CHUNK_ID]”

### D. Verification & Safety
- NLI model / entailment classifier (claim vs evidence)  
- Automates hallucination detection; triggers human review  

### E. Fishiness / Accounting Risk Detector
- **Hybrid:** Rule-based + supervised classifier + numeric anomaly detection  
- Signals: “related party”, restatement, one-time gains, ambiguous language, time-series anomalies  

### F. Operational Acceleration
- Distill cross-encoder → faster bi-encoder / lightweight reranker  
- Maintains near cross-encoder accuracy with lower latency  

---

## 6) Offline Metrics

**Retrieval:** Recall@k, MRR, Precision@k  
**Reranker:** nDCG@k  
**Span extraction:** EM, F1  
**QA generation:** Answer F1 / EM; ROUGE-L / BLEU (human eval preferred)  
**Citation & faithfulness:** Citation Precision, Span IOU, Hallucination rate  
**Fishiness detection:** Precision@k, Recall, AUROC  
**Cost/latency:** tokens/query, latency (ms), cost per 1k queries  

---

## 7) Online Validation / A/B Plan

- **A/B buckets:** Baseline (A) vs Candidate (B)  
- **Traffic routing:** Random assignment ensures fair comparison  
- **Primary KPI:** Human Analyst Trust Score (1–5)  
- **Secondary KPIs:** Correction rate, time-to-decision, citation precision, fishiness FP  
- Collect N queries/day, blinded human evaluation  
- Roll out candidate gradually if trust & corrections improve  
- Safety: adjust fishiness thresholds if FP spikes  

---

## 8) Handling Special Data Issues

**Class imbalance:** Up-sample positives, class weights, synthetic augmentation, active learning  
**Data leakage:** Strict time-based splits; avoid prompt leakage  
**Drift:** Monitor retrieval recall, embedding distributions, hallucinations; retrain/incremental re-index if thresholds crossed  

---

## 9) MLOps Plan

- **Experiment tracking:** W&B / MLflow  
- **Data & model versioning:** DVC / LakeFS, MLflow model registry  
- **Reproducibility:** Docker containers, Airflow / Prefect pipelines, seeded runs  
- **Deployment:** Microservices (retriever, reranker, extractor, generator, fishiness scoring), Kubernetes, API gateway (FastAPI + OAuth2)  
- **CI/CD:** Unit/integration tests, nightly E2E tests, canary rollout  
- **Observability:** Metrics (latency, tokens, recall, hallucination), logs, secure audit  

---

## 10) Security & Compliance

- Data privacy: maintain TOS records, access controls  
- Secrets: Vault / AWS Secrets Manager  
- Encryption: at rest (S3 SSE) + in transit (TLS)  
- RBAC & audit: role-based access, immutable logs  
- Redaction: auto-PII removal (regex/NER)  
- Model safety: content filters, human-in-loop for risky outputs  
- Compliance: full provenance, human disclaimers for financial advisory  

---

## 11) Preferred Tools & Stack

- Storage/Infra: S3 / MinIO, EC2 / EKS, Docker  
- Ingestion/Orchestration: Kafka, Prefect / Airflow  
- Search/Vector DB: Elasticsearch, FAISS / Milvus / Pinecone  
- Embeddings & LLMs: OpenAI, sentence-transformers, FinBERT, LLaMA/Mistral + vLLM  
- Reranker & NLI: Hugging Face Transformers  
- MLOps: W&B / MLflow, DVC, Git, CI (GitHub Actions)  
- API: FastAPI  
- Monitoring: Prometheus + Grafana; Sentry  
- Security: Vault, IAM, KMS  

---

## 12) Two-Week Starter Timeline

### Week 1 — Core ingestion + baseline retrieval
- Project scaffolding, repo, CI  
- EDGAR ingestion skeleton (5–10 filings)  
- Text extraction + chunking  
- Baseline retrieval: BM25 + MiniLM embeddings + FAISS  
- Basic API: /ingest + /ask  
- **Deliverables:** Repo, ingestion script/notebook, demo endpoint, retrieval evaluation notebook, small helper artifact  

### Week 2 — Reranker, span extraction, fishiness prototype
- Integrate cross-encoder reranker, top-N → top-6  
- Extractive QA model (HF or fine-tuned)  
- Baseline fishiness detector: rules + small supervised classifier  
- Strict prompt templates + pipeline orchestration  
- Human evaluation harness: collect analyst labels (50–200 QA pairs)  
- **Deliverables:** Working API (retrieval → reranker → extractor → generator), fishiness flags, evaluation report (offline metrics + human eval), A/B test plan skeleton, containerized deployment manifests  

---

## 13) Offline Evaluation / Quick Checks

- Compute Recall@10 for retriever  
- Compute EM/F1 for extractive QA  
- Manual review: Citation Precision (50 answers)  
- Fishiness detector recall on historical restatement cases

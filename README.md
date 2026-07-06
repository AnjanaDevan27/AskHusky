# AskHusky — AI Voice Help Desk for Northeastern International Students

> This project is actively being built and is not yet complete.

AskHusky is a 24/7 AI voice help desk for Northeastern University F-1 international students. It provides instant, accurate guidance on visa topics including CPT, OPT, travel signatures, and co-op rules — available outside OGS office hours (8:30am–4:30pm ET).

---

## The Problem

International students on F-1 visas face high-stakes questions at all hours:

- "Can I do CPT while taking one class?"
- "My travel signature expired — can I still travel?"
- "What happens if I exceed 364 days of full-time CPT?"

Wrong answers have real consequences — SEVIS violations, loss of status, deportation risk. OGS is only open 8:30am–4:30pm. AskHusky is available 24/7.

---

## Architecture

```
Student Voice Input
        |
   Whisper (STT)
        |
 Orchestrator Agent
        |
 Specialized Agents (LangGraph)
 Visa Status | CPT | OPT | Travel | Co-op | Urgency | Appointment
        |
   RAG Retriever (Pinecone)
        |
   Safety Layer (OGS Disclaimer + GSOC Routing)
        |
   ElevenLabs (TTS)
        |
   Student Voice Output
```

---

## Tech Stack

### Voice & AI
| Tool | Purpose |
|---|---|
| openai-whisper | Speech to text |
| Claude API | LLM backbone |
| ElevenLabs | Text to speech |
| LangGraph | Multi-agent orchestration |
| LangSmith | Agent tracing and observability |
| Pinecone | Vector database |
| RAGAS | RAG evaluation |

### Data Engineering
| Tool | Purpose |
|---|---|
| BeautifulSoup | OGS web scraper |
| Prefect | Pipeline orchestration |
| Confluent Cloud (Kafka) | Event streaming |
| Databricks | Spark processing |
| Delta Lake | Data lakehouse |
| dbt Core | Data transformations |
| Snowflake | Data warehouse |

### MLOps & Monitoring
| Tool | Purpose |
|---|---|
| MLflow | Experiment tracking |
| Evidently AI | Data drift detection |
| Tableau Public | Analytics dashboard |

### Deployment
| Tool | Purpose |
|---|---|
| FastAPI + WebSocket | API layer |
| Docker | Containerization |
| AWS ECR | Container registry |
| AWS ECS Fargate | Serverless deployment |
| Terraform | Infrastructure as code |
| GitHub Actions | CI/CD |

### Security
| Tool | Purpose |
|---|---|
| Presidio | PII scrubbing |
| slowapi | Rate limiting |
| JWT | Authentication |

---

## Project Structure

```
askhusky/
├── .github/workflows/ci.yml
├── infra/terraform/
├── data/
│   ├── scraper/
│   │   ├── ogs_scraper.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   └── embed_and_store.py
│   └── pii/
│       └── scrubber.py
├── agents/
│   ├── orchestrator.py
│   ├── visa_status.py
│   ├── cpt.py
│   ├── opt.py
│   ├── travel.py
│   ├── coop.py
│   ├── urgency.py
│   └── appointment.py
├── rag/
│   ├── retriever.py
│   └── safety.py
├── voice/
│   ├── whisper_input.py
│   └── elevenlabs_output.py
├── api/
│   ├── main.py
│   ├── auth.py
│   └── rate_limit.py
├── pipelines/
│   └── prefect_refresh.py
├── data_engineering/
│   ├── dbt/models/
│   ├── delta_lake/schema.py
│   └── snowflake/loader.py
├── evaluation/
│   ├── ragas_eval.py
│   ├── golden_test_set.json
│   └── mlflow_logger.py
├── monitoring/
│   ├── evidently_drift.py
│   └── langsmith_config.py
├── tests/
│   ├── test_scraper.py
│   ├── test_agents.py
│   ├── test_safety.py
│   └── load/locustfile.py
├── frontend/app.py
├── docker/Dockerfile
├── .env.example
├── requirements.txt
└── pyproject.toml
```

---

## The 7 Agents

| Agent | Responsibility |
|---|---|
| Orchestrator | Intent classification and routing |
| Visa Status | F-1 maintenance, SEVIS, enrollment rules |
| CPT | Eligibility, 364-day limit, remote co-op |
| OPT | Pre/post OPT, STEM OPT, USCIS timelines |
| Travel | Travel signatures, visa renewal, I-94 |
| Co-op | International co-op rules, GPA requirements |
| Urgency | High-stakes detection, GSOC emergency routing |
| Appointment | OGS advisor booking by query type |

---

## Data Pipeline

The knowledge base is built from all public pages on `international.northeastern.edu/ogs`:

1. BFS crawler scrapes 268 pages, respecting `robots.txt` and a 1.5-second crawl delay
2. Chunker filters 19 junk pages (events, login, venue) and splits remaining content into 3,106 chunks at 500 characters with 100-character overlap
3. Each chunk is embedded using `all-MiniLM-L6-v2` and upserted to Pinecone with source metadata
4. A Prefect pipeline re-runs this process weekly to keep the knowledge base current
5. Every pipeline run is logged to MLflow for tracking and comparison over time

---

## Safety Design

Every immigration response includes:

- OGS disclaimer — "This is not legal advice. Verify with your advisor before taking action."
- GSOC 24/7 emergency routing — for urgent situations including SEVIS termination, deportation risk, and port of entry denial
- PII scrubbing — student names, SEVIS numbers, passport numbers, and emails are anonymized via Presidio before any logging or storage

AskHusky is designed to augment OGS, not replace it.

---

## RAG Pipeline

Student questions are answered through a retrieval-augmented generation pipeline:

1. Student question is embedded using the same `all-MiniLM-L6-v2` model used at index time
2. Pinecone returns the top 5 most semantically similar OGS chunks (cosine similarity 0.60–0.66)
3. Retrieved chunks are formatted as context and passed to the relevant LangGraph agent
4. The agent generates an answer grounded in the retrieved OGS content
5. The safety layer appends the OGS disclaimer before the response is returned

---

## Testing

The project uses a multi-layer testing strategy:

- pytest for unit tests covering scraper URL filtering, text cleaning, chunker logic, junk detection, and PII scrubbing — 45 tests, both good and bad paths covered
- RAGAS for RAG quality evaluation (faithfulness, answer relevance, context recall)
- locust for load testing the FastAPI endpoints
- GitHub Actions for CI/CD — tests run on every push

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/askhusky.git
cd askhusky
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
cp .env.example .env
# Add your API keys to .env
```

### Required API Keys

```bash
ANTHROPIC_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=askhusky
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ELEVENLABS_API_KEY=
LANGCHAIN_API_KEY=
```

### Run the pipeline

```bash
# Scrape OGS
python data/scraper/ogs_scraper.py

# Chunk content
python data/scraper/chunker.py

# Embed and store in Pinecone
python data/embeddings/embed_and_store.py

# Run tests
pytest tests/ -v

# View MLflow dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Estimated Cost

| Service | Cost |
|---|---|
| Claude API | ~$15–30 |
| AWS ECS Fargate | ~$5–10/month |
| Pinecone | Free tier |
| Snowflake | $400 free credit |
| Confluent Cloud | $400 free credit |
| Everything else | Free / open source |
| Total (6 months hosted) | ~$150–200 |

ECS is configured to scale to zero at midnight ET and scale up at 8am ET to minimize cost.

---

## Scoping Validation

Project scoped using Dr. Ramin Mohammadi's MLOps framework at mlwithramin.com across seven dimensions: need for AI, data assessment, impact assessment, timeline, model evaluation, infrastructure, and risk assessment. All seven dimensions pass.

---

## Author

Built by Anjana — targeting Data Scientist, ML Engineer, Data Engineer, and LLM Engineer roles.
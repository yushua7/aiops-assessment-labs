# Natural Language Processing II: Transformer-Based Incident Intelligence for AIOps

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
The operations desk receives hundreds of free-text incident tickets a day, and humans read each one to route it, pull out the affected host, IP, service and error code, and check whether it has happened before. The job was to build a CPU-only "ticket intelligence" microservice that does all three, and to evaluate it against honest baselines before trusting it.

## What I did
1. **Built the corpus:** generated 1,500 synthetic tickets in 5 resolver categories with character-offset entity labels, then profiled them (class balance, duplicates, span integrity).
2. **Analysed tokenization:** measured how WordPiece splits operational identifiers and chose `max_length` from the token-length distribution.
3. **Built a baseline:** TF-IDF with logistic regression, with top features per class.
4. **Fine-tuned a transformer:** BERT-Mini with the Hugging Face Trainer, best-checkpoint selection on macro F1, error analysis and an out-of-distribution test.
5. **Semantic search and deduplication:** all-MiniLM-L6-v2 embeddings against TF-IDF, scored with Recall@k, plus duplicate thresholds.
6. **Clustering:** KMeans on embeddings with and without log-style masking, with c-TF-IDF topic labels.
7. **Custom NER:** a spaCy statistical NER plus an `entity_ruler` for IPs and error codes.
8. **Served it:** one FastAPI endpoint combining the classifier, NER and KB search, packaged as an offline, non-root Docker image with a healthcheck.

## Results

### Data
| Item | Value |
|------|-------|
| Tickets | 1,500 (train 1,050 / val 225 / test 225) |
| Noisy labels injected (train only) | 29 |
| Entity mentions | ERROR_CODE 1,040, HOST 1,140, IP 542, SERVICE 733 |
| Bad entity spans | 0 |

### Tokenization
| Item | Value |
|------|-------|
| Tokens per word | 2.33 |
| p95 / p99 / max length | 38 / 41 / 47 tokens |
| Unknown tokens | 0 |
| Chosen `max_length` | 48 (0% truncated) |

An IP such as `10.42.7.19` becomes 7 tokens, and `k8s-node-07.prod.local` becomes 12. WordPiece falls back to characters, so nothing becomes `[UNK]`.

### Classifier: baseline versus transformer
| Model | Test accuracy | Test macro-F1 | OOD accuracy (10 paraphrases) |
|-------|---------------|---------------|-------------------------------|
| TF-IDF + logistic regression | 0.9956 | 0.9955 | 0.50 |
| BERT-Mini fine-tuned | 1.0000 | 1.0000 | 0.30 |

Both models are near-perfect on templated text, and BERT-Mini took about 110 seconds to train on 2 vCPUs. On hand-written paraphrases both fell apart, and the transformer did worse than the baseline (0.30 against 0.50). With only 10 probes this is a rough signal rather than a firm ranking, but it shows that perfect test scores on templated data say little about real tickets. The remedy is real labelled tickets.

### Semantic search
| Retriever | Recall@1 | Recall@3 |
|-----------|----------|----------|
| TF-IDF | 0.500 | 0.583 |
| MiniLM | 1.000 | 1.000 |

TF-IDF failed on paraphrases that share no words with the KB title. Duplicate detection separated cleanly: the three real duplicates scored 0.62 to 0.67 and the two unrelated tickets scored 0.22 and 0.06. Changing the threshold to 0.75 moved all three duplicates to RELATED, while 0.45 gave the same verdicts as the default.

### Clustering
| Text | Silhouette (k=5) | ARI | Homogeneity |
|------|------------------|-----|-------------|
| Raw | 0.155 | 0.395 | 0.530 |
| Masked | 0.195 | 0.514 | 0.597 |

Masking hosts, IPs and numbers improved every metric. Silhouette preferred k=8 over the 5 resolver groups, so clusters follow incident families more than resolver teams. The c-TF-IDF topics were readable (for example `waf, ids, triggered, malware` for security).

### Entity extraction
- The statistical model reached 100% F1 on the test split for HOST, IP, ERROR_CODE and SERVICE.
- On an unseen host name (`cache-west-a`), the model truncated it to a SERVICE entity `cache-west`, because the name did not follow the `name-NN` pattern seen in training.
- The `entity_ruler` guarantees IPs and error codes such as `ECONNREFUSED`, which rules handle better than statistics for closed formats.

### Service
- The image `ticket-nlp:1.0` was 551 MB of content (2.22 GB on disk), ran as a non-root user with offline model loading and reported healthy.
- It used about 496 MiB of its 2 GiB limit, with a mean latency of 46.8 ms per request.
- Empty input returned 422, and an input of about 1,400 tokens returned a normal 200 because the tokenizer carries `model_max_length=512`.
- A realistic ticket returned the category, four entities and similar incidents. A KB paraphrase was flagged as a duplicate of KB-001.

### Validation
The lab's validation script passed all 15 checks.

## Problems I ran into
**1. spaCy CLI crashed (`TyperArgument.make_metavar() takes 1 positional argument but 2 were given`).** spaCy 3.7.4 uses typer 0.9.4, which is incompatible with click 8.5. Pinning `click==8.1.7` fixed it, and I added the pin to `requirements.txt`.

**2. The hybrid NER pipeline would not reload.** The trained model shared a `tok2vec` component with `ner`. Removing the component left config references such as `${components.tok2vec.model.encode.width}` pointing at nothing, and `spacy.load` failed with a config validation error. The same bug broke the container. It took three attempts to find the right fix: retrain with the standard config, merge `tok2vec` into `ner`, patch the 7 leftover config references, then confirm the pipeline reloads as exactly `['entity_ruler', 'ner']`. My second attempt (training a standalone config) failed because the training run did not produce a model, which I diagnosed by checking the exit code and log before building anything on top of it.

**3. The container could not read the model.** `model.safetensors` was saved with `-rw-------` permissions, so the non-root container user could not open it. I made the model files world-readable before rebuilding the image.

**4. Environment differed from the lab text.** The server had Python 3.12 and Docker 29 rather than 3.10 and 24, so I used `python:3.12-slim` as the base image to match the host.

## What I learned
- Perfect scores on templated data are not evidence of quality. The out-of-distribution probe is the more honest test, and here the simple baseline beat the transformer on it.
- Tokenization sets cost. Operational identifiers expand to many subwords, so choosing `max_length` from measured percentiles is better than guessing.
- Embeddings beat lexical search on paraphrases, and the duplicate threshold is a business trade-off between a wrong fix being applied and a known fix being rediscovered.
- Masking high-variance tokens (hosts, IPs, numbers) makes unsupervised clusters follow meaning rather than noise.
- Rules suit closed formats such as IPs and error codes, while statistical models suit open vocabularies. Combining them gives auditable behaviour.
- A saved model has to be loadable in a fresh environment. Reloading the pipeline and checking file permissions caught two bugs that training metrics could not.
- Pin the whole dependency tree, including packages you never import directly.

## Repo contents
- `reports/`: token statistics, baseline, transformer, OOD, search, clustering and NER metrics
- `scripts/`: data generation, analysis, training, search, clustering and NER scripts
- `service/`: the FastAPI application and serving requirements
- `Dockerfile`, `data/kb.json`, `data/kb_queries.json`, `data/ood_tickets.csv`

Course-provided starter files are not included. Trained models (`models/`) are not included because of their size.

## Tools
PyTorch (CPU), Hugging Face Transformers, sentence-transformers, spaCy, scikit-learn, FastAPI, Docker, jq, Bash

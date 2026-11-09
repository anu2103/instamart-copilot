\# Instamart Copilot

\### Customer Intelligence · Retention Modelling · Policy-Grounded Support Automation



An offline-first analytics and support-automation project built with Python, pandas, NumPy, scikit-learn, SQLite, and the supplied deterministic `mock\_llm.py`.



This project follows the customer journey from raw order data to actionable retention insights, then extends the workflow into privacy-aware ticket triage, policy retrieval, and a tool-using refund agent. All language-model calls use the supplied offline mock—no API key, paid service, or network connection is required.



> \*\*Project principle:\*\* make each decision inspectable. Data transformations are validated, model results are saved as outputs, ticket handling includes guardrails, policy answers are grounded in retrieved documents, and refund actions are constrained by explicit business rules.



\## Contents



\- \[Overview](#overview)

\- \[Reproduce](#reproduce)

\- \[Part A — Data Quality and Analytics](#part-a--data-quality-and-analytics)

\- \[Part B — Customer Segmentation and Lapse Prediction](#part-b--customer-segmentation-and-lapse-prediction)

\- \[Part C — Ticket Triage and Policy RAG](#part-c--ticket-triage-and-policy-rag)

\- \[Part D — Tool-Using Refund Agent](#part-d--tool-using-refund-agent)

\- \[Limitations and Next Steps](#limitations-and-next-steps)



\---



\## Overview



\### Problem statement



An online grocery business needs more than a dashboard. It needs reliable customer and order data, a way to identify customers who may stop purchasing, consistent support-ticket routing, policy answers traceable to source documents, and safe handling of refund requests.



\### What this project delivers



| Workstream | Implementation | Evidence produced |

|---|---|---|

| Data foundation | Raw-data audit, cleaning, SQLite database, SQL queries and validation | Clean datasets, database and analysis outputs |

| Customer intelligence | RFM features, K-means segmentation and lapse prediction | Segment profiles, model evaluation, feature importance and retention list |

| Support triage | PII redaction, zero-shot/few-shot prompts, retries, output validation, token budget and routing | Ticket results, prompt comparison, confusion summary and budget demonstration |

| Policy retrieval | Chunked policy documents, TF-IDF retrieval and grounded mock answers | Chunk-overlap evaluation outputs |

| Refund automation | Order lookup, policy search, refund calculation, simulated issue/escalation and checkpointing | Scenario summary, run log, refund ledger and state checkpoints |



\### Technology



\- \*\*Language:\*\* Python 3.11 or newer

\- \*\*Data and modelling:\*\* pandas, NumPy, scikit-learn and joblib

\- \*\*Storage and queries:\*\* SQLite and SQL

\- \*\*Language-model interface:\*\* supplied `mock\_llm.py`, used offline

\- \*\*Reproducibility:\*\* fixed random seeds where specified and saved CSV/JSONL outputs



\### Repository structure



```text

.

├── generate\_data.py

├── mock\_llm.py

├── requirements.txt

├── .gitignore

├── README.md

├── data/

│   └── policies/

│       ├── delivery\_policy.txt

│       ├── escalation\_policy.txt

│       ├── payment\_policy.txt

│       ├── privacy\_policy.txt

│       └── refund\_policy.txt

├── scripts/

├── sql/

│   └── queries.sql

├── outputs/

└── state/

```



The supplied `generate\_data.py` and `mock\_llm.py` must remain unmodified. Generated data and intermediate files should be reproducible from the scripts. Commit the required evaluation outputs, source code, SQL, policy text files and agent evidence. Do not commit secrets, virtual environments, screenshots or prohibited media.



\---



\## Reproduce



Run the following commands from the repository root. These instructions use Windows PowerShell.



\### 1. Create the environment and install dependencies



```powershell

py -3 -m venv .venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\\.venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```



Python 3.11 or newer is required. If the `py -3` launcher is unavailable, use `python -m venv .venv` with a compatible Python installation.



\### 2. Generate the data and run Part A



```powershell

python .\\generate\_data.py

python .\\scripts\\audit\_raw.py

python .\\scripts\\clean\_data.py

python .\\scripts\\build\_database.py

python .\\scripts\\validate\_data.py

python .\\scripts\\run\_sql\_queries.py

python .\\scripts\\analyze\_data.py

```



The previously shared validation run finished with `ALL CHECKPOINTS PASSED`.



\### 3. Run Part B



```powershell

python .\\scripts\\build\_rfm.py

python .\\scripts\\cluster\_customers.py

python .\\scripts\\profile\_segments.py

python .\\scripts\\build\_lapse\_dataset.py

python .\\scripts\\prepare\_lapse\_data.py

python .\\scripts\\train\_lapse\_models.py

python .\\scripts\\oversample\_lapse\_model.py

python .\\scripts\\finalize\_lapse\_model.py

```



\### 4. Run Part C



```powershell

python .\\scripts\\triage\_tickets.py

python .\\scripts\\policy\_rag.py

```



\### 5. Run Part D



```powershell

python .\\scripts\\refund\_agent.py

```



Run the scripts in order because later stages depend on data and model artifacts created earlier. Before submission, verify that every command matches a real, committed filename in the repository.



\### 6. Verify files and branch



```powershell

Get-ChildItem .\\scripts

Get-ChildItem .\\data\\policies\\\*.txt

Get-ChildItem .\\outputs

Get-ChildItem .\\state

git branch --show-current

git status --short

```



The required branch name is `main`. The final public repository link must point directly to that branch, for example:



`https://github.com/<your-account>/<your-repository>/tree/main`



\---



\## Part A — Data Quality and Analytics



\### Approach



Part A establishes a dependable data foundation before analysis. The workflow audits generated raw files, handles duplicate and invalid records, produces cleaned datasets, loads the data into SQLite, executes SQL queries, and validates the resulting tables and business totals.



\### Data-quality checkpoints



| Checkpoint | Result |

|---|---:|

| Raw customers | 1,200 |

| Raw products | 40 |

| Raw orders | 13,084 |

| Duplicate order rows identified | 119 |

| Cleaned orders | 12,309 |

| Completed orders | 11,110 |

| Cancelled orders | 1,199 |

| Completed-order revenue | INR 2,998,668.80 |

| Cleaned customer ages outside 18–60 | 0 |

| Unknown customer gender values retained explicitly | 30 |



The cleaned customer data contains 610 female, 560 male and 30 unknown gender values. The raw customer data contained 61 missing ages and 6 invalid ages, which were handled during cleaning rather than treated as valid ages.



\### Findings



Completed-order revenue is \*\*INR 2,998,668.80\*\*. The order data decreases from 13,084 raw rows to 12,309 cleaned rows after the documented cleaning and duplicate-handling process.



Keeping unknown gender values explicit avoids silently assigning a gender to customers whose data does not provide one. Validating the cleaned age range also helps prevent invalid values from distorting later customer profiles.



The validation script reported `ALL CHECKPOINTS PASSED` in the previously shared run. Re-run it from a clean clone to confirm the same checkpoints in the submission environment.



\### Key artifacts



\- `sql/queries.sql` — SQL analysis queries

\- `scripts/audit\_raw.py` — raw-data audit

\- `scripts/clean\_data.py` — data cleaning

\- `scripts/build\_database.py` — SQLite database creation

\- `scripts/validate\_data.py` — data-quality checkpoints

\- `scripts/run\_sql\_queries.py` and `scripts/analyze\_data.py` — query execution and analysis



\---



\## Part B — Customer Segmentation and Lapse Prediction



\### RFM features and segmentation



Recency, frequency and monetary (RFM) features summarise customer purchasing behaviour. The reported RFM snapshot date is \*\*1 October 2025\*\*.



| Measure | Result |

|---|---:|

| Customers with completed orders in the RFM snapshot | 1,175 |

| Median recency | 19 days |

| Mean frequency | 9.4553 |

| Maximum frequency | 33 |

| Mean monetary value | INR 2,552.06 |

| Maximum monetary value | INR 14,766.55 |

| Selected K-means clusters | 4 |

| K=4 silhouette score | 0.526 |



Four clusters were selected using the recorded elbow and silhouette evaluation.



| Segment | Customers | Potential business action |

|---|---:|---|

| Growth Opportunity | 436 | Encourage a second or more frequent purchase with relevant offers |

| Frequent Loyal | 419 | Reinforce repeat behaviour and avoid unnecessary discounting |

| At-Risk | 182 | Prioritise relevant win-back outreach |

| Premium | 138 | Protect the experience of high-value customers with appropriate service |



These names translate the clusters into practical customer groups. They are descriptive profiles of model-generated clusters, not proof that every customer in a group behaves identically.



\### Lapse prediction



The lapse dataset contains \*\*1,156 customers\*\*. A customer is labelled lapsed if they had no completed order during \*\*1–30 September 2025\*\*, using a \*\*1 September 2025\*\* cutoff.



| Target | Customers | Share |

|---|---:|---:|

| Lapsed | 443 | 38.32% |

| Not lapsed | 713 | 61.68% |

| Total | 1,156 | 100% |



The reported best Random Forest configuration used `max\_depth=8` and `min\_samples\_leaf=10`.



| Evaluation measure | Reported result |

|---|---:|

| F1 score | 0.655367 |

| Precision | 0.659091 |

| Recall | 0.651685 |



The reported confusion matrix is:



| Actual / Predicted | Not lapsed | Lapsed |

|---|---:|---:|

| Not lapsed | 113 | 30 |

| Lapsed | 31 | 58 |



An oversampled Random Forest run reported F1 \*\*0.6575\*\*, recall \*\*0.8090\*\*, precision \*\*0.5538\*\* and accuracy \*\*0.6767\*\*. The higher recall comes with lower precision, so more customers may be flagged for outreach, including some who would not lapse.



The saved feature-importance output ranks frequency (\*\*0.279718\*\*) and monetary value (\*\*0.271441\*\*) highest among the reported features, followed by recency (\*\*0.115184\*\*), average delivery time (\*\*0.061242\*\*) and cancellation rate (\*\*0.052334\*\*).



\### Business interpretation



The model can support prioritisation, not automatic decisions about customers. A retention list is most useful when paired with a controlled outreach experiment and measurement of incremental repeat purchases. Model quality should be judged against the cost of missed lapses and unnecessary outreach, not by one metric alone.



\### Key artifacts



\- `outputs/rfm.csv`

\- `outputs/kmeans\_metrics.csv`

\- `outputs/customer\_segments.csv`

\- `outputs/segment\_profiles.csv`

\- `outputs/lapse\_dataset.csv`

\- `outputs/model\_evaluation.csv`

\- `outputs/logistic\_cv\_results.csv`

\- `outputs/forest\_cv\_results.csv`

\- `outputs/logistic\_regression\_coefficients.csv`

\- `outputs/random\_forest\_feature\_importance.csv`

\- `outputs/retention\_list.csv`

\- `outputs/lapse\_model.joblib`



\---



\## Part C — Ticket Triage and Policy RAG



\### C1. Support-ticket triage



The triage pipeline uses the supplied offline mock model to classify tickets into \*\*billing\*\*, \*\*delivery\*\*, \*\*technical\*\* or \*\*fraud\*\*. It compares zero-shot and few-shot prompting and adds operational guardrails around the model response.



The implementation includes:



\- Redaction of email addresses, phone numbers and order IDs before classification.

\- Explicit handling for empty tickets and tickets estimated above 400 tokens.

\- A maximum of three retries for transient failures, using exponential backoff with jitter.

\- JSON parsing and category validation, with a keyword fallback for non-JSON responses.

\- A separate invalid-output status when a response cannot safely be accepted.

\- A token-budget demonstration and routing rules for fraud, premium customers, and human review.



The previously shared ticket dataset contains \*\*164 rows\*\*, including \*\*4 empty\*\* and \*\*4 long\*\* tickets. Reported redaction counts are \*\*55 email addresses\*\*, \*\*91 phone numbers\*\* and \*\*132 order IDs\*\*—278 redaction matches in total. A single ticket can contain more than one match.



The earlier prompt evaluation reported \*\*147 classified tickets\*\* for each prompt comparison. Zero-shot accuracy was \*\*85.7% (126/147)\*\*, while the reported few-shot run reached \*\*100%\*\* on that evaluation set. The zero-shot errors included 11 fraud cases classified as billing and 10 technical cases classified as billing.



These results describe a deterministic mock environment and a finite evaluation set. They should not be interpreted as evidence of equivalent accuracy with real customer traffic or a production language model.



\### C2. Policy retrieval-augmented generation (RAG)



The policy assistant retrieves policy text before asking the offline mock model to answer. Five policy files cover delivery, escalation, payment, privacy and refunds. The reported chunk configuration uses \*\*40-word chunks\*\* with \*\*10-word overlap\*\*. Retrieval uses TF-IDF-style lexical weighting and returns the top three chunks for a question.



Overlap matters because policy facts can straddle a chunk boundary. In the reported example, a question about the delivery-time threshold for a full refund retrieved the relevant refund policy, but a zero-overlap chunk omitted the threshold phrase from the answer context.



| Configuration | Chunks | Answerable questions | Retrieval hits | Expected facts found | Unanswerable questions | Fallback answers |

|---|---:|---:|---:|---:|---:|---:|

| 10-word overlap | 15 | 12 | 12 | 12 | 2 | 2 |

| No overlap | 13 | 12 | 12 | 10 | 2 | 2 |



With overlap, the evaluation found all \*\*12/12\*\* expected facts and used the fallback for both \*\*2/2\*\* unanswerable questions. Without overlap, retrieval still hit the expected document for all 12 answerable questions, but only \*\*10/12\*\* expected facts appeared in the answers. This shows that retrieving the right document is not sufficient if the required sentence is split across chunks.



\### Key artifacts



\- `scripts/triage\_tickets.py`

\- `scripts/policy\_rag.py`

\- `data/policies/\*.txt`

\- `outputs/triage\_results.csv`

\- `outputs/triage\_zero\_shot.csv`

\- `outputs/triage\_few\_shot.csv`

\- `outputs/triage\_prompt\_comparison.csv`

\- `outputs/triage\_zero\_shot\_confusion.csv`

\- `outputs/triage\_budget\_demo.csv`

\- Policy-evaluation CSV outputs generated by `scripts/policy\_rag.py`



\---



\## Part D — Tool-Using Refund Agent



\### Design



The refund agent uses a tool-driven reasoning loop to check an order and apply written refund rules. Its tools cover order lookup, policy search, refund calculation, simulated refund issuance and escalation to a human lead.



The agent applies these controls:



1\. An unknown order returns `NOT\_FOUND`.

2\. An order that is not completed is denied.

3\. A completed order delivered within the policy threshold is not eligible for the late-delivery refund.

4\. A calculated refund up to INR 1,000 is issued by the simulated tool.

5\. A refund above INR 1,000 is escalated to a human lead.

6\. Token, time and iteration limits trigger explicit fallback outcomes.

7\. Run logs and state checkpoints support inspection and crash/resume testing.



\*\*Safety boundary:\*\* `issue\_refund` and the refund ledger are part of a local simulation. They do not move real money or connect to a payment provider.



\### Scenario evaluation



The previously shared run reported these outcomes:



| Scenario | Expected outcome | Reported outcome | Refund amount | Tool calls | Tokens |

|---|---|---|---:|---:|---:|

| S1 | REFUND | REFUND | INR 128.25 | 5 | 1,152 |

| S2 | REFUND | REFUND | INR 135.00 | 5 | 1,152 |

| S3 | ESCALATE | ESCALATE | INR 1,999.00 | 5 | 1,139 |

| S4 | DENIED | DENIED | INR 0.00 | 4 | 809 |

| S5 | DENIED | DENIED | INR 0.00 | 3 | 490 |

| S6 | NOT\_FOUND | NOT\_FOUND | INR 0.00 | 2 | 174 |

| S7 | FALLBACK | FALLBACK\_MAX\_ITERATIONS | INR 0.00 | 6 | 3,306 |



The amount for S3 is the \*\*INR 1,999.00\*\* escalation amount, not the INR 1,000 policy threshold. The earlier summary-extraction issue was corrected to read the amount from the relevant tool observation.



\### Safety and recovery tests



| Test | Reported result |

|---|---|

| S1 with a token limit of 200 | `FALLBACK\_TOKEN\_LIMIT` |

| S1 with `time\_limit\_seconds=0` | `FALLBACK\_TIMEOUT` |

| S7 with a maximum of 6 iterations | `FALLBACK\_MAX\_ITERATIONS` |

| S1 crash/resume demonstration | Resumed to `REFUND` for INR 128.25 |



In the crash/resume demonstration, the tool sequence was `get\_order → search\_policy → calculate\_refund → issue\_refund`. The recorded run ID was `S1\_RESUME\_DEMO\_bb65e316`. Run logs, the refund ledger, the scenario summary and checkpoint files make the simulated behaviour inspectable.



\### ReAct and plan-and-execute



\*\*ReAct-style execution\*\* alternates between a model decision and a tool result. It is useful when the next action depends on the current order status or calculated refund. Its disadvantage is that extra iterations increase token use and the risk of loops, so iteration limits are important.



\*\*Plan-and-execute\*\* creates a sequence of steps before carrying them out. It can make a predictable workflow easier to audit, but a fixed plan may need adjustment when a tool returns an unexpected result. Either design still needs order validation, policy checks, monetary limits, escalation and explicit failure handling. This implementation demonstrates a bounded tool loop with checkpoints and simulated side effects; it is not a production payment system.



\### Key artifacts



\- `scripts/refund\_agent.py`

\- `outputs/agent\_scenario\_summary.csv`

\- `outputs/agent\_runs.jsonl`

\- `outputs/refund\_ledger.jsonl`

\- `state/` — agent checkpoint JSON files



Confirm these Part D files exist after the final run and include the required ones in the repository.



\---



\## Limitations and Next Steps



\### Limitations



\- \*\*Synthetic data:\*\* results are based on generated data and cannot establish how the system would perform in a real grocery business.

\- \*\*Offline mock model:\*\* `mock\_llm.py` is deterministic and intentionally simulates selected failure modes. Prompt accuracy is not a benchmark of a hosted model.

\- \*\*Lexical retrieval:\*\* TF-IDF retrieval may miss semantically equivalent wording. Chunk size and overlap affect whether the complete policy fact reaches the answerer.

\- \*\*Finite evaluation samples:\*\* ticket and policy results are based on small, predefined evaluation sets. A perfect few-shot score on this set does not guarantee generalisation.

\- \*\*Lapse modelling:\*\* customer behaviour can change over time. Precision/recall trade-offs, calibration, temporal validation and business impact need further testing.

\- \*\*Simulated refunds:\*\* the agent ledger is local evidence only; it does not execute real payments, authenticate a customer or provide production-grade authorization.

\- \*\*Privacy and operations:\*\* redaction rules can miss unexpected formats. A real deployment would need broader privacy testing, secure logs, monitoring, access controls and human escalation.



\### Next steps



1\. Evaluate segmentation and lapse prediction on a later time window to reduce temporal leakage risk.

2\. Run controlled retention experiments and measure incremental repeat purchases, margin and customer experience.

3\. Expand ticket evaluation with independently labelled examples, category-level precision/recall and adversarial PII cases.

4\. Compare lexical retrieval with a stronger retrieval method while preserving source attribution and unanswerable-question fallbacks.

5\. Add structured traces for every agent decision and test duplicate tool calls, interrupted runs and idempotency.

6\. Before any real refund integration, require authenticated order ownership, authorization checks, durable idempotency keys, audit trails, rate limits and human approval for exceptions.

7\. Keep the repository reproducible: retain the supplied files unchanged, document every reported metric, pin compatible dependencies and rerun the full sequence from a clean clone before submission.



\---



\## Submission checklist



\- \[ ] `generate\_data.py` and `mock\_llm.py` are present and unchanged.

\- \[ ] `requirements.txt` and `.gitignore` are committed.

\- \[ ] All source code for Parts A–D and `sql/queries.sql` are committed.

\- \[ ] All five policy `.txt` files are committed.

\- \[ ] Required CSV/JSONL/model outputs and required agent checkpoint evidence are present.

\- \[ ] README section names and reproduction commands match the actual repository.

\- \[ ] No screenshots, images, charts, PDFs, slides, audio, video, secrets or virtual environment are committed.

\- \[ ] `git branch --show-current` returns `main`.

\- \[ ] All changes are committed and pushed to `origin/main`.

\- \[ ] Repository visibility is \*\*Public\*\*.

\- \[ ] The submitted link opens the `main` branch directly.




\# Instamart Customer Intelligence \& Support Copilot



From raw commerce data to customer retention signals, policy-grounded support, and controlled refund decisions.



A reproducible, offline-first analytics and support project that combines data quality checks, SQL analysis, customer segmentation, lapse-risk modelling, ticket triage, policy retrieval, and a tool-using refund agent.



Repository: \[anu2103/instamart-copilot](https://github.com/anu2103/instamart-copilot)  

Submission branch: `main`  

Runtime: Python 3.11+  

Model execution: Supplied offline `mock\_llm.py` — no API key or paid service required.



\## Overview



Quick-commerce teams need more than order totals. They need trustworthy data, useful customer segments, early retention signals, consistent support routing, and safeguards around customer refunds.



This project builds an end-to-end workflow around those needs. It starts with synthetic customer, product, order, and support-ticket data, then progresses from analytical foundations to machine-learning experiments and policy-aware automation.



\### What the project delivers


| Workstream | Outcome |
|---|---|
| Part A — Data foundation | Raw-data audit, cleaning, SQLite database, SQL analysis, and validation checkpoints |
| Part B — Customer intelligence | RFM features, K-means customer segments, lapse-risk modelling, and a retention-priority list |
| Part C — Support copilot | PII redaction, ticket guardrails, zero-shot/few-shot triage comparison, retry handling, token-budget handling, and policy retrieval evaluation |
| Part D — Controlled refund agent | Policy-aware tool use, refund calculations, escalation rules, execution limits, checkpointing, and resume testing |

\### Design principles



\- Reproducibility: generated data and fixed seeds support repeatable experiments.

\- Evidence over claims: reported results are numerical and tied to saved outputs.

\- Privacy before classification: email addresses, phone numbers, and order IDs are redacted before ticket processing.

\- Grounded support: policy answers are retrieved from local policy documents; unsupported questions receive a fallback.

\- Controlled actions: refunds are simulated locally, and high-value refunds require human approval.

\- Offline by design: the supplied mock language model avoids external accounts, API keys, and paid inference.



\## Reproduce



Run the project from the repository root. Use Python 3.11 or newer and SQLite 3.25 or newer. All dependencies are free and open source.



\### 1. Clone and create the environment



```bash

git clone https://github.com/anu2103/instamart-copilot.git

cd instamart-copilot



python -m venv .venv

```



Activate the environment:



Windows PowerShell

```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Linux or macOS

```bash

source .venv/bin/activate

```



Install the project dependencies:



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\### 2. Generate and validate the data



Run the commands in this order:



```bash

python generate\_data.py

python scripts/audit\_raw.py

python scripts/clean\_data.py

python scripts/build\_database.py

python scripts/run\_sql\_queries.py

python scripts/analyze\_data.py

python scripts/validate\_data.py

```



The validation script should finish with:



```text

ALL CHECKPOINTS PASSED

```



\### 3. Build customer intelligence models



```bash

python scripts/build\_rfm.py

python scripts/cluster\_customers.py

python scripts/profile\_segments.py

python scripts/build\_lapse\_dataset.py

python scripts/prepare\_lapse\_data.py

python scripts/train\_lapse\_models.py

python scripts/oversample\_lapse\_model.py

python scripts/finalize\_lapse\_model.py

```



\### 4. Run support triage and policy retrieval



```bash

python scripts/triage\_tickets.py

python scripts/policy\_rag.py

```



The ticket-triage script includes its own acceptance checks. Its final run should report `FINAL ACCEPTANCE CHECKS PASSED`.



\### 5. Run the refund-agent scenarios



```bash

python scripts/refund\_agent.py

```



This runs the seven scenarios and the safety demonstrations, and writes agent-run logs, a simulated refund ledger, scenario summaries, and checkpoint state.



Reproducibility note: run the commands from the repository root and in the order shown. The scripts create intermediate data, SQLite files, model artefacts, and evaluation outputs. The committed outputs are reference evidence; rerunning a script may regenerate them. The final submission should be tested from a fresh clone, not only from an existing working folder.



\## Part A — Data Foundation and SQL Analysis



\### Objective



Create a dependable analytical foundation from synthetic quick-commerce data. The workflow audits raw inputs, handles duplicate and invalid records, produces clean datasets, loads SQLite, runs analytical SQL, and checks important data-quality and business-metric checkpoints.



\### Data quality and validation


| Checkpoint | Result |
|---|---:|
| Raw customers | 1,200 |
| Products | 40 |
| Raw orders | 13,084 |
| Duplicate order rows identified | 119 |
| Cleaned order rows | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |

The customer dataset contains 1,200 records. The cleaning workflow handles missing or invalid age values, normalises gender values, and standardises city data. The cleaned customer age range is 18–60; gender counts are 610 female, 560 male, and 30 unknown.



The raw order count and cleaned order count differ because duplicate and invalid records are addressed during data preparation. The cleaned order dataset is the basis for downstream analysis.



\### SQL and analytical workflow



The project stores the analytical data in `data/instamart.db` and keeps reusable queries in `sql/queries.sql`. The Python SQL runner executes the queries against SQLite. The analysis script produces additional numerical summaries.



The workflow is designed to make the path from raw input to business result inspectable: audit first, clean next, load the database, run SQL, and validate the results.



\### Key finding



The cleaned data contains 11,110 completed orders and INR 2,998,668.80 in completed-order revenue. This becomes the consistent starting point for customer-level features and later retention analysis.



Evidence: `scripts/audit\_raw.py`, `scripts/clean\_data.py`, `scripts/build\_database.py`, `scripts/run\_sql\_queries.py`, `scripts/analyze\_data.py`, `scripts/validate\_data.py`, and `sql/queries.sql`.



\## Part B — Customer Segmentation and Lapse-Risk Modelling



\### Objective



Move from order-level reporting to customer-level decisions. The workflow creates Recency, Frequency, and Monetary (RFM) features, groups customers into interpretable segments, builds a lapse-labelled dataset, evaluates classification models, and produces a retention-priority list.



\### RFM snapshot



The RFM snapshot date is 1 October 2025. The analysis includes 1,175 customers with completed orders.


| Measure | Result |
|---|---:|
| Customers with completed orders | 1,175 |
| Median recency | 19 days |
| Mean frequency | 9.4553 orders |
| Maximum frequency | 33 orders |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

Recency measures how recently a customer ordered, frequency measures the number of completed orders, and monetary value measures completed-order spend. Together, these features provide a compact behavioural view of customer activity.



\### Customer segments



K-means clustering was evaluated using cluster-quality metrics, with K = 4 selected for the reported segmentation. The four-cluster silhouette score was 0.526.


| Segment | Customers | Interpretation |
|---|---:|---|
| Growth Opportunity | 436 | Customers to nurture through relevant engagement and repeat-purchase opportunities |
| Premium | 138 | High-value customers whose experience should be protected |
| Frequent Loyal | 419 | Regular customers who may respond to loyalty and convenience improvements |
| At-Risk | 182 | Customers who warrant closer retention attention |

These labels are business-facing interpretations of the clusters, not guarantees about an individual customer's future behaviour.



\### Lapse prediction



The lapse dataset contains 1,156 customers. The label identifies whether a customer had no completed order from 1 September through 30 September 2025, using the 1 September 2025 cutoff.


| Measure | Result |
|---|---:|
| Customers in lapse dataset | 1,156 |
| Lapsed customers | 443 |
| Non-lapsed customers | 713 |
| Lapse share | 38.32% |

The reported Random Forest configuration uses `max\_depth=8` and `min\_samples\_leaf=10`.


| Evaluation measure | Reported result |
|---|---:|
| Precision | 0.6591 |
| Recall | 0.6517 |
| F1 score | 0.6554 |
| Confusion matrix — true negatives | 113 |
| Confusion matrix — false positives | 30 |
| Confusion matrix — false negatives | 31 |
| Confusion matrix — true positives | 58 |

The confusion matrix is shown in the order `\[\[TN, FP], \[FN, TP]]`. The F1 score balances precision and recall; neither a good F1 score nor a predicted risk guarantees that a particular customer will lapse.



An oversampled Random Forest experiment increased recall to approximately 0.8090, with precision approximately 0.5538, F1 approximately 0.6575, and accuracy approximately 0.6767. This illustrates the trade-off: finding more potential lapses can also create more false positives.



\### Feature importance



The reported Random Forest feature-importance ranking is:


| Feature | Importance |
|---|---:|
| Frequency | 0.2797 |
| Monetary value | 0.2714 |
| Recency | 0.1152 |
| Average delivery minutes | 0.0612 |
| Cancellation rate | 0.0523 |

Frequency and monetary value have the highest importance among the listed features. These values describe the fitted model's relative feature importance; they do not establish that a feature causes customer behaviour.



\### Business application



The retention list in `outputs/retention\_list.csv` prioritises Premium customers from the test set by predicted lapse probability. A practical next step would be to test targeted retention messages or service improvements and measure their impact rather than assuming that a prediction itself prevents churn.



Evidence: `outputs/rfm.csv`, `outputs/kmeans\_metrics.csv`, `outputs/customer\_segments.csv`, `outputs/segment\_profiles.csv`, `outputs/lapse\_dataset.csv`, `outputs/model\_evaluation.csv`, `outputs/logistic\_cv\_results.csv`, `outputs/forest\_cv\_results.csv`, `outputs/logistic\_regression\_coefficients.csv`, `outputs/random\_forest\_feature\_importance.csv`, `outputs/retention\_list.csv`, and `outputs/lapse\_model.joblib`.



\## Part C — Privacy-Aware Ticket Triage and Policy RAG



\### Objective



Build a local support workflow that redacts sensitive patterns, applies input guardrails, classifies eligible tickets, handles transient model failures, compares prompt strategies, and retrieves answers from policy documents.



\### Ticket privacy and guardrails



The supplied ticket dataset contains 164 tickets. The final triage run recorded the following redactions:


| Pattern | Substitutions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| Total substitutions | 278 |

The redaction verification passed with no remaining matches for the three configured patterns. Four empty tickets were skipped and four long tickets were flagged rather than sent for normal classification.



The final few-shot run recorded:


| Final ticket status | Tickets |
|---|---:|
| Classified using JSON output | 135 |
| Classified using keyword fallback | 12 |
| Flagged for invalid model output | 9 |
| Flagged as long | 4 |
| Skipped as empty | 4 |
| Total | 164 |

Transient failures were recovered through retry handling. Invalid outputs, empty inputs, long inputs, and budget-exhausted tickets are handled explicitly rather than silently treated as successful classifications.



\### Zero-shot versus few-shot



The evaluation compares classification performance and token use on the 147 tickets that received a classification.


| Measure | Zero-shot | Few-shot |
|---|---:|---:|
| Correct classifications | 126 / 147 | 147 / 147 |
| Accuracy | 85.71% | 100.00% |
| Total response tokens | 18,292 | 31,396 |
| Tokens per classified ticket | 124.44 | 213.58 |

In this mock-model evaluation, few-shot prompting improved observed accuracy by 14.29 percentage points while using 13,104 additional response tokens, or about 89.14 extra tokens per classified ticket. The result is specific to the supplied synthetic evaluation and mock model; it should not be interpreted as a guarantee of real-world production accuracy.



The zero-shot confusion table recorded 10 technical tickets predicted as billing and 11 fraud tickets predicted as billing. These errors matter operationally because misrouting fraud-related tickets could delay the required security response.



\### Token-budget demonstration



A zero-shot run with a token budget of 6,000 recorded 104 budget-exhausted tickets, 49 classified tickets, and 6,101 total response tokens. The run demonstrates that budget handling must account for the cost of the response that crosses the limit; the observed total is slightly above the configured budget. The final output explicitly records budget-exhausted cases rather than presenting them as successful classifications.



\### Routing policy



The final output recorded these routing totals:


| Route | Tickets |
|---|---:|
| Human review | 17 |
| Premium desk | 11 |
| Security team | 34 |
| Support queue | 102 |
| Total | 164 |

Fraud tickets are routed to the security team at high priority; Premium customers can be routed to the premium desk; and invalid, empty, long, or budget-exhausted cases are routed for human review. Routing is a rule-based decision in this project, separate from the model's predicted category.



\### Policy retrieval and overlap experiment



The local policy retriever chunks policy text and ranks matching passages using TF-IDF similarity. The experiment compares chunks with 10-word overlap against chunks without overlap.


| Configuration | Chunks | Answerable questions | Retrieval hits | Expected facts found | Unanswerable questions | Fallback answers |
|---|---:|---:|---:|---:|---:|---:|
| 10-word overlap | 15 | 12 | 12 | 12 | 2 | 2 |
| No overlap | 13 | 12 | 12 | 10 | 2 | 2 |

In this evaluation, overlap preserved key facts that could otherwise be split across chunk boundaries. The no-overlap configuration retrieved relevant material for all 12 answerable questions, but only 10 answers contained the expected fact. Both configurations returned fallback answers for the two unanswerable questions.



The policy documents cover refunds, delivery, payment, privacy, and escalation. Retrieval quality and answer quality are not identical: finding a relevant passage does not by itself guarantee that the final answer contains the required threshold or rule.



Evidence: `outputs/triage\_zero\_shot.csv`, `outputs/triage\_few\_shot.csv`, `outputs/triage\_results.csv`, `outputs/triage\_prompt\_comparison.csv`, `outputs/triage\_zero\_shot\_confusion.csv`, `outputs/triage\_budget\_demo.csv`, `outputs/policy\_eval\_overlap10.csv`, `outputs/policy\_eval\_overlap0.csv`, and `outputs/policy\_eval\_summary.csv`.



\## Part D — Policy-Grounded Refund Agent



\### Objective



Implement a tool-using agent that looks up orders, retrieves policy, calculates eligible refunds, and either issues a simulated refund, denies an ineligible request, reports a missing order, or escalates to a human lead.



\### Tool workflow and safeguards



The agent uses these local tools:



\- `get\_order` — retrieves order details.

\- `search\_policy` — retrieves relevant policy content.

\- `calculate\_refund` — calculates the eligible refund amount.

\- `issue\_refund` — records a simulated refund.

\- `escalate\_to\_human` — routes cases requiring approval to a human lead.



The workflow applies these rules:



\- Only completed orders are eligible for late-delivery refunds.

\- A completed order delivered in more than 45 minutes is eligible under the configured late-delivery policy.

\- Refunds up to INR 1,000 can be approved automatically.

\- Refunds above INR 1,000 require human approval.

\- Missing orders, ineligible orders, execution limits, and unresolved cases receive explicit outcomes rather than being reported as successful refunds.



The refund ledger is a local simulation. No real payment, bank transfer, or customer account is affected.



\### Seven scenario results


| Scenario | Outcome | Refund amount (INR) | Expected amount (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|---:|
| S1 | REFUND | 128.25 | 128.25 | 5 | 1,152 |
| S2 | REFUND | 135.00 | 135.00 | 5 | 1,152 |
| S3 | ESCALATE | 1,999.00 | 1,999.00 | 5 | 1,139 |
| S4 | DENIED | 0.00 | 0.00 | 4 | 809 |
| S5 | DENIED | 0.00 | 0.00 | 3 | 490 |
| S6 | NOT\_FOUND | 0.00 | 0.00 | 2 | 174 |
| S7 | FALLBACK\_MAX\_ITERATIONS | 0.00 | 0.00 | 6 | 3,306 |

All seven scenario amounts matched their expected amounts. S3 correctly escalated a refund above INR 1,000; its INR 1,999 amount is not an automatically issued refund.



\### Failure handling and resume behaviour


| Demonstration | Observed outcome |
|---|---|
| Token limit reached | `FALLBACK\_TOKEN\_LIMIT` |
| Time limit reached | `FALLBACK\_TIMEOUT` |
| Maximum of six model calls reached | `FALLBACK\_MAX\_ITERATIONS` |
| Checkpoint/resume demonstration | Resumed with final status `REFUND` |

In the resume demonstration, the tools executed across the simulated interruption and continuation were `get\_order`, `search\_policy`, `calculate\_refund`, and `issue\_refund`. The resumed answer reported the simulated INR 128.25 refund. Checkpointing helps preserve progress across an interruption; it does not replace production-grade transaction guarantees or idempotency controls.



\### Why this design matters



The agent separates reasoning from actions. A model response alone does not establish that a refund is valid: the order record, policy rules, computed amount, approval threshold, tool result, and final status all matter. Execution limits and human escalation provide explicit failure paths instead of allowing an agent to continue indefinitely or claim an unperformed action.



Evidence: `scripts/refund\_agent.py`, `outputs/agent\_scenario\_summary.csv`, `outputs/agent\_runs.jsonl`, `outputs/refund\_ledger.jsonl`, and the checkpoint files under `state/`.



\## Limitations and Next Steps



\### Limitations



1\. Synthetic data: the project uses generated data and does not establish how the workflow will perform on live Instamart operations.

2\. Offline mock model: classification accuracy and agent behaviour are evaluated against the supplied `mock\_llm.py`; they do not establish the performance of a hosted or production language model.

3\. Lapse prediction is not causation: model feature importance and predicted risk do not prove why a customer lapses or that a retention action will work.

4\. Model evaluation scope: the reported scores apply to the chosen dataset, label window, features, and evaluation split. Performance may change with new customers or different time periods.

5\. Lexical retrieval: TF-IDF can miss semantic matches and can return passages that do not contain the exact fact needed. Chunk size and overlap affect whether policy rules remain together.

6\. Pattern-based PII redaction: the configured regular expressions cover the tested email, phone, and order-ID formats, not every possible form of personal or confidential information.

7\. Simulated refund execution: the ledger is not connected to a payment provider. Production use would require authenticated access, durable audit logs, idempotency, transaction controls, monitoring, and human approval workflows.

8\. Reproducibility environment: dependency compatibility, random seeds, file paths, and generated state should be verified by running the documented sequence in a fresh environment.



\### Next steps



\- Add automated tests for data contracts, SQL results, model outputs, redaction, routing, policy thresholds, and agent failure paths.

\- Evaluate lapse prediction on a later time window and monitor precision-recall trade-offs before using predictions for customer outreach.

\- Test retention interventions with controlled experiments and track incremental repeat orders.

\- Improve policy retrieval with stronger document boundaries, metadata filters, and more varied paraphrased evaluation questions.

\- Extend privacy testing with additional identifiers and adversarial text patterns.

\- Add structured observability for retries, token usage, budget exhaustion, tool errors, escalation reasons, and checkpoint recovery.

\- If integrated with real systems, require authenticated tools, explicit approval for consequential actions, durable idempotency keys, and reconciliation against the payment provider.



\### Final submission checklist



\- \[ ] `generate\_data.py` and `mock\_llm.py` are included and unchanged.

\- \[ ] `requirements.txt`, `.gitignore`, all Part A–D source files, and `sql/queries.sql` are committed.

\- \[ ] All five policy documents under `data/policies/` are committed.

\- \[ ] Required output files are committed; charts and screenshots are not committed.

\- \[ ] The README contains all seven required sections and reports results in text.

\- \[ ] Reproduction commands succeed in a fresh environment.

\- \[ ] All work is on the `main` branch.

\- \[ ] The repository is public and the submission link resolves to `https://github.com/anu2103/instamart-copilot/tree/main`.

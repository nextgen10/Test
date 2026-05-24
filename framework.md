# Enterprise Intelligent QA + Compliance + Document Validation Framework

**Version:** 1.0  
**Status:** Planning  
**Last updated:** 2026-05-24

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Business Context](#2-business-context)
3. [Design Principles](#3-design-principles)
4. [Architecture Overview](#4-architecture-overview)
5. [Technology Stack and Responsibilities](#5-technology-stack-and-responsibilities)
6. [Service Boundaries](#6-service-boundaries)
7. [Multi-Agent Model](#7-multi-agent-model)
8. [Core Data Models](#8-core-data-models)
9. [End-to-End Workflow](#9-end-to-end-workflow)
10. [Human-in-the-Loop (HITL)](#10-human-in-the-loop-hitl)
11. [Rule Engine and Policy-as-Code](#11-rule-engine-and-policy-as-code)
12. [Document Intelligence Pipeline](#12-document-intelligence-pipeline)
13. [Durable Execution with Temporal](#13-durable-execution-with-temporal)
14. [Workflow Governance with Flowable](#14-workflow-governance-with-flowable)
15. [Agent Orchestration with LangGraph](#15-agent-orchestration-with-langgraph)
16. [RAG and Memory with LlamaIndex](#16-rag-and-memory-with-llamaindex)
17. [Eventing, Integration, and APIs](#17-eventing-integration-and-apis)
18. [Security, Identity, and Compliance](#18-security-identity-and-compliance)
19. [Observability and Evaluation](#19-observability-and-evaluation)
20. [Deployment Topology](#20-deployment-topology)
21. [Implementation Phases](#21-implementation-phases)
22. [Reference: Consent-Letter Journey](#22-reference-consent-letter-journey)
23. [Appendices](#23-appendices)

---

## 1. Purpose and Scope

### 1.1 What This Framework Defines

This document is the **canonical plan** for building an enterprise platform that:

- Ingests and validates regulatory and customer documents (consent letters, compliance forms, multilingual communications).
- Applies versioned business and regulatory rules at scale (e.g. 178+ rule sets).
- Orchestrates multiple AI agents with clear ownership, retries, and escalation.
- Pauses for human review when confidence or risk thresholds are not met.
- Generates QA test cases and executes validation workflows.
- Maintains a **complete, replayable audit trail** suitable for banking, insurance, and regulated QA organizations.

### 1.2 What This Framework Is Not

- Not a single-tool replacement for BPM, agents, or durable execution.
- Not a prompt-only compliance system (rules and evidence must be externalized).
- Not a batch OCR pipeline without governance, HITL, or audit linkage.

### 1.3 In Scope

| Area | Included |
|------|----------|
| Multi-agent collaboration | Yes |
| Human approvals and SLAs | Yes |
| Long-running workflows (days/weeks) | Yes |
| Auditability and regulator replay | Yes |
| Multilingual document handling | Yes |
| RAG over policy and historical decisions | Yes |
| Rule-based + AI hybrid validation | Yes |
| Test generation and execution | Yes |

### 1.4 Out of Scope (Initial Release)

- Model training/fine-tuning pipelines (capture feedback; train in Phase 3+).
- Full legal e-signature platform (integrate via adapters).
- Customer-facing self-service portal (API-ready; UI optional).

---

## 2. Business Context

### 2.1 Typical Inputs

- Consent letters and amendments
- Compliance and regulatory forms
- Product disclosure documents
- Multilingual customer communication (DE, FR, IT, EN, extensible)

### 2.2 Required Outcomes

| Outcome | Description |
|---------|-------------|
| **Validation** | Document structure, signatures, required clauses present |
| **Rule compliance** | Product × country × language × channel rules satisfied |
| **Risk detection** | Policy violations, translation drift, missing disclosures |
| **QA artifacts** | Generated test cases linked to rules and document versions |
| **Escalation** | Low-confidence or high-risk items routed to humans |
| **Audit** | Every decision tied to inputs, models, rules, and human actions |

### 2.3 Example Business Rule

```text
IF Product = Wealth Management
AND Country = Switzerland
AND Language = German
THEN Include Consent Block A
AND Signature Block Required = true
```

Rules are **identified**, **versioned**, and **evaluated** outside LLM prompts; agents explain outcomes against rule IDs.

---

## 3. Design Principles

### 3.1 Separation of Concerns

| Concern | Owner |
|---------|--------|
| Business process lifecycle, approvals, SLAs | **Flowable** |
| Durable technical execution, waits, retries | **Temporal** |
| Agent routing, reasoning graphs, tool use | **LangGraph** |
| Document parsing, indexing, retrieval | **LlamaIndex** (+ OCR adapters) |

**Critical rule:** Do not duplicate the same long-running state machine in both Flowable and Temporal. Flowable owns the **case**; Temporal owns **durable activities** invoked from that case.

### 3.2 Deterministic Before Probabilistic

1. Schema and format validation  
2. Hard rule engine (AND/OR, nested conditions)  
3. Retrieval-augmented context  
4. LLM reasoning and compliance scoring  
5. Human review when thresholds fail  

### 3.3 Evidence-First Decisions

Every automated or human decision produces a **Decision Record** (see [§8](#8-core-data-models)) with hashes, versions, and reasoning traces—not only a boolean pass/fail.

### 3.4 Fail Safe on Regulated Paths

- Low confidence → HITL, not auto-approve.
- Missing rule version → block, do not guess.
- Audit write failure → fail the step (no silent drops).

### 3.5 Version Everything

| Artifact | Versioned |
|----------|-----------|
| Regulatory rules | Yes (effective dates) |
| Prompts / agent graphs | Yes |
| Models | Yes |
| Document snapshots | Content hash |
| Retrieval corpora | Index version |

---

## 4. Architecture Overview

### 4.1 Logical Layers

```text
┌─────────────────────────────────────────────────────────────────┐
│  Experience Layer (Reviewer UI, Ops Dashboard, API Clients)      │
├─────────────────────────────────────────────────────────────────┤
│  API Gateway + Auth (Keycloak)                                   │
├─────────────────────────────────────────────────────────────────┤
│  Workflow Governance (Flowable) — cases, tasks, SLAs, escalate   │
├─────────────────────────────────────────────────────────────────┤
│  Agent Orchestration (LangGraph) — supervisor + specialist agents │
├─────────────────────────────────────────────────────────────────┤
│  Durable Runtime (Temporal) — activities, timers, compensation   │
├─────────────────────────────────────────────────────────────────┤
│  Domain Services — rules, documents, compliance, QA, audit       │
├─────────────────────────────────────────────────────────────────┤
│  Data — operational DB, object store, vector DB, audit log       │
├─────────────────────────────────────────────────────────────────┤
│  Event Bus (Kafka) — domain events, integration, analytics       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 End-to-End Control Flow

```text
Customer Uploads Document
            |
            v
    API / Ingestion Service
            |
            v
    Flowable: Start Case (e.g. DocumentReviewProcess)
            |
            v
    Temporal: IngestionActivity (store, hash, virus scan)
            |
            v
    LangGraph: SupervisorAgent
            |
    +-------+--------+-------------+
    |                |             |
    v                v             v
DocumentAgent   RuleAgent    TranslationAgent
    |                |             |
    +-------+--------+-------------+
            |
            v
    ComplianceAgent (confidence + risk)
            |
    confidence < threshold OR risk = HIGH?
            |
      yes --+--> Flowable User Task (HITL)
            |         |
            |    Approve / Reject / Edit / Escalate
            |
            v
    QAAgent → Test case artifacts
            |
            v
    ExecutionAgent → run tests / store results
            |
            v
    Audit + Decision Record persisted
            |
            v
    Flowable: Complete Case
```

### 4.3 Capability Matrix

| Capability | Supported | Primary technology |
|------------|-----------|-------------------|
| Multi-agent collaboration | Yes | LangGraph |
| Human approvals | Yes | Flowable |
| Long-running workflows | Yes | Flowable + Temporal |
| Auditability | Yes | Audit service + immutable log |
| Document intelligence | Yes | LlamaIndex + OCR adapters |
| RAG | Yes | LlamaIndex + Weaviate |
| Rule-based AI | Yes | Rule service + agents |
| Stateful workflows | Yes | Flowable case state |
| Retry/recovery | Yes | Temporal |
| Multilingual support | Yes | Translation agent + normalized store |

---

## 5. Technology Stack and Responsibilities

| Layer | Technology | Solves |
|-------|------------|--------|
| Workflow governance | **Flowable** | BPMN cases, user tasks, SLAs, escalations, org roles |
| Agent orchestration | **LangGraph** | Supervisor routing, agent graphs, tool calls, checkpoints |
| Durable execution | **Temporal** | Retries, timers, human-wait (days), sagas, idempotency |
| Document intelligence | **LlamaIndex** | Parsing pipelines, indexes, structured extraction, RAG |
| Vector memory | **Weaviate** | Policy chunks, prior decisions, feedback embeddings |
| Event streaming | **Apache Kafka** | Domain events, async integration, analytics |
| Observability | **Langfuse** | Traces, prompts, agent spans, eval hooks |
| Authentication | **Keycloak** | OIDC, roles, service accounts, tenant boundaries |
| OCR / layout (adapter) | Azure Doc Intelligence, Unstructured.io | Tables, signatures, layout-aware text |

### 5.1 Why All Four (Flowable + LangGraph + Temporal + LlamaIndex)

| Tool alone | Gap |
|------------|-----|
| Flowable alone | Weak at LLM agent graphs and semantic document pipelines |
| LangGraph alone | No enterprise BPM, org-wide task inbox, or regulator-friendly case model |
| Temporal alone | No business analyst–friendly process design or human task UI |
| LlamaIndex alone | No end-to-end compliance workflow or HITL |

**Combined:** governed process + intelligent agents + reliable execution + document/RAG depth.

---

## 6. Service Boundaries

### 6.1 Recommended Microservices (Logical)

| Service | Responsibility |
|---------|----------------|
| **ingestion-api** | Upload, metadata, case creation trigger |
| **case-orchestrator** | Flowable delegate listeners, case ↔ Temporal bridge |
| **agent-runtime** | Hosts LangGraph graphs, exposes `runAgentStep(caseId, step)` |
| **document-service** | Storage, OCR jobs, LlamaIndex pipelines, extraction schemas |
| **rule-service** | Load rule packs, evaluate, return violations with rule IDs |
| **compliance-service** | Policy checks, risk score, calibrated confidence |
| **qa-service** | Test case generation, execution scheduling, result aggregation |
| **audit-service** | Append-only Decision Records, export for regulators |
| **notification-service** | Email/Teams for tasks, escalations, timeouts |
| **integration-adapter** | Core banking, CRM, document management (optional) |

### 6.2 Ownership Rules

- **Flowable** stores: process instance ID, current activity, assignee, due dates, business variables (case summary).
- **Temporal** stores: workflow run ID, activity heartbeats, retry state, timer handles—not business rule definitions.
- **LangGraph** stores: per-step agent state (checkpointed); references case ID and document hashes.
- **LlamaIndex** stores: index metadata; vectors in Weaviate; raw files in object storage.

---

## 7. Multi-Agent Model

### 7.1 Agent Roster

| Agent | Role | Primary tools / deps |
|-------|------|----------------------|
| **Supervisor** | Route, retry, escalate, aggregate | LangGraph router, case API |
| **Document** | OCR, parse, classify, extract clauses | LlamaIndex, OCR adapters |
| **Rule validation** | Execute rule packs, explain violations | rule-service |
| **Translation** | Normalize DE/FR/IT/EN, semantic alignment | translation API, glossary index |
| **Compliance** | Policy + governance + risk + confidence | RAG, compliance policies |
| **QA** | Generate test cases from rules + doc state | qa-service, templates |
| **Execution** | Run tests, collect evidence | test runners, Temporal activities |
| **Reporting** | Summaries for ops and audit export | audit-service |

### 7.2 LangGraph Topology (Simplified)

```text
SupervisorAgent
    -> DocumentAgent
    -> TranslationAgent (if multilingual or detected)
    -> RuleValidationAgent
    -> ComplianceAgent
    -> [conditional] HumanApprovalNode (external wait via Flowable/Temporal)
    -> QAAgent
    -> ExecutionAgent
    -> ReportingAgent
    -> END
```

### 7.3 Supervisor Responsibilities

- Read case context and document classification.
- Enforce step order and skip rules (e.g. skip translation if already EN-only).
- On agent failure: retry with backoff (Temporal) or escalate (Flowable).
- Emit `AgentStepCompleted` events for audit.
- Never override hard rule **FAIL** with LLM **PASS**.

### 7.4 Agent Contracts (Input / Output)

Each agent receives and returns a **partial Decision Record** fragment:

```json
{
  "agent": "document",
  "status": "completed",
  "artifacts": {
    "document_type": "consent_letter",
    "extracted_clauses": ["..."],
    "signature_detected": true
  },
  "confidence": 0.94,
  "issues": [],
  "trace_id": "..."
}
```

---

## 8. Core Data Models

### 8.1 Case (Flowable Business Key)

```json
{
  "case_id": "uuid",
  "business_key": "CONSENT-2026-0001842",
  "tenant_id": "ubs-ch",
  "product": "wealth_management",
  "country": "CH",
  "language": "de",
  "document_ids": ["doc-uuid-1"],
  "process_definition_key": "document_review_v1",
  "status": "in_review",
  "created_at": "ISO-8601",
  "sla_due_at": "ISO-8601"
}
```

### 8.2 Document

```json
{
  "document_id": "uuid",
  "content_hash": "sha256:...",
  "mime_type": "application/pdf",
  "storage_uri": "s3://...",
  "source_language": "de",
  "normalized_text_uri": "s3://...",
  "ocr_provider": "azure_di",
  "index_version": "llama-idx-v3",
  "classification": "consent_letter"
}
```

### 8.3 Rule Evaluation Result

```json
{
  "rule_pack_id": "consent_ch_wm_v12",
  "rule_pack_version": "2026-04-01",
  "evaluated_at": "ISO-8601",
  "results": [
    {
      "rule_id": "CONSENT_102",
      "status": "fail",
      "severity": "high",
      "message": "Consent Block A missing for DE/CH/WM",
      "evidence_spans": [{"page": 2, "start": 120, "end": 340}]
    }
  ],
  "summary": { "pass": 176, "fail": 2, "skip": 0 }
}
```

### 8.4 Decision Record (Canonical Audit Unit)

**Every** terminal or human-overridden outcome MUST persist one Decision Record:

```json
{
  "decision_id": "uuid",
  "case_id": "uuid",
  "decision_type": "automated | human_override",
  "outcome": "approved | rejected | needs_edit",
  "confidence": {
    "raw_model": 0.71,
    "calibrated": 0.68,
    "factors": {
      "rule_criticality": "high",
      "retrieval_quality": 0.82,
      "translation_variance": 0.15
    }
  },
  "rule_references": [
    { "rule_id": "CONSENT_102", "version": "2026-04-01", "result": "fail" }
  ],
  "reasoning_trace": "Structured narrative or span-linked explanation",
  "evidence": {
    "document_hashes": ["sha256:..."],
    "retrieval_snapshot_id": "snap-uuid",
    "agent_traces": ["langfuse-trace-id"]
  },
  "versions": {
    "model": "gpt-4.1-2026-03",
    "prompt": "compliance-agent-v7",
    "agent_graph": "supervisor-v4"
  },
  "human_review": {
    "reviewer_id": "user@bank.com",
    "action": "edit",
    "notes": "Corrected clause translation",
    "completed_at": "ISO-8601"
  },
  "immutable_audit_ref": "audit-log-sequence-88421"
}
```

### 8.5 Human Task Payload (Flowable)

```json
{
  "task_type": "compliance_review",
  "title": "Review consent clause mismatch",
  "case_id": "uuid",
  "ai_summary": {
    "rule_id": "CONSENT_102",
    "confidence": 0.71,
    "issue": "Potential mismatch in translated clause"
  },
  "allowed_actions": ["approve", "reject", "edit", "escalate"],
  "due_in_hours": 48
}
```

---

## 9. End-to-End Workflow

### 9.1 Process Definition (BPMN Conceptual)

| Stage | Flowable | Temporal | LangGraph |
|-------|----------|----------|-----------|
| Intake | Start event | Ingest activity | — |
| Extract | Service task | OCR activity | DocumentAgent |
| Translate | Gateway (lang) | Translate activity | TranslationAgent |
| Rules | Service task | Rule eval activity | RuleValidationAgent |
| Compliance | Service task | — | ComplianceAgent |
| HITL | User task | Wait signal (up to N days) | Pause graph |
| QA | Service task | Test gen activity | QAAgent |
| Execute | Service task | Test run activity | ExecutionAgent |
| Close | End event | — | ReportingAgent |

### 9.2 State Transitions (Case)

```text
RECEIVED → INGESTED → EXTRACTING → EXTRACTED →
RULE_EVALUATING → RULE_COMPLETE →
COMPLIANCE_EVALUATING →
  ├─ (pass) → QA_GENERATING → EXECUTING → COMPLETED
  └─ (fail/low confidence) → PENDING_HUMAN →
        ├─ APPROVED → QA_GENERATING → ...
        ├─ REJECTED → COMPLETED_REJECTED
        └─ ESCALATED → PENDING_SUPERVISOR → ...
```

### 9.3 Idempotency

- All Temporal activities keyed by `(case_id, step_name, input_hash)`.
- Replayed Kafka consumers use `decision_id` deduplication.
- LangGraph checkpoints keyed by `case_id + graph_version`.

---

## 10. Human-in-the-Loop (HITL)

### 10.1 Trigger Conditions

| Condition | Default action |
|-----------|----------------|
| Calibrated confidence &lt; 85% | Create Flowable user task |
| Any `severity: critical` rule fail | Mandatory human review |
| Translation variance above threshold | Human + bilingual reviewer queue |
| Regulatory risk score = HIGH | Escalation queue |
| Agent disagreement (rule FAIL vs LLM PASS) | Block auto-approve; HITL |

Thresholds are **configurable per tenant/product**, not hardcoded in agents.

### 10.2 Flowable Pause Pattern

```text
AI: Confidence = 72%, issue = "Potential compliance issue"
        |
        v
Flowable: User Task "Review consent clause mismatch"
        |
   [Approve] [Reject] [Edit] [Escalate]
        |
        v
Temporal: signalHumanTaskCompleted(caseId, outcome)
        |
        v
LangGraph: resume from HumanApprovalNode
        |
        v
Feedback → vector memory + audit + optional training queue
```

### 10.3 Reviewer UI Requirements

- Side-by-side: source PDF, extracted clauses, rule violations, AI reasoning.
- Highlight evidence spans (page/offset).
- Show rule text and version used.
- Capture structured override reason codes.
- Display SLA countdown and escalation path.

### 10.4 Feedback Loop

| Destination | Content |
|-------------|---------|
| Vector memory (Weaviate) | Approved corrections, glossary pairs |
| Audit log | Full human edit diff |
| Evaluation dataset | `(input, expected, model_output, human_label)` |
| Rule backlog | Suggested new rules (human triage, not auto-deploy) |

---

## 11. Rule Engine and Policy-as-Code

### 11.1 Rule Pack Structure

```yaml
rule_pack:
  id: consent_ch_wm
  version: "2026-04-01"
  effective_from: "2026-04-01"
  effective_to: null
  rules:
    - id: CONSENT_102
      description: Wealth Management CH DE Consent Block A
      when:
        all:
          - field: product
            equals: wealth_management
          - field: country
            equals: CH
          - field: language
            equals: de
      then:
        require_clause: CONSENT_BLOCK_A
        require_signature: true
      severity: high
```

### 11.2 Evaluation Pipeline

1. Load rule pack by `(tenant, product, country, effective_date)`.
2. Build evaluation context from Document Agent extraction.
3. Run deterministic evaluator (no LLM for pass/fail).
4. Rule Agent adds natural-language explanations and groups violations for HITL.

### 11.3 Rule vs LLM Boundary

| Task | Owner |
|------|--------|
| Pass/fail against rule ID | Rule engine |
| Missing clause detection (structured) | Rule engine + extraction schema |
| Ambiguous legal interpretation | Compliance Agent + HITL |
| New rule proposal | Human governance workflow |

---

## 12. Document Intelligence Pipeline

### 12.1 Stages

```text
Raw upload → Virus scan → Hash → OCR/layout →
Chunk + embed → Classify doc type →
Extract structured fields/clauses →
Index (LlamaIndex) → Publish DocumentExtracted event
```

### 12.2 Extraction Schema (Example)

| Field | Type | Used by |
|-------|------|---------|
| `document_type` | enum | Routing |
| `clauses[]` | list | Rules, compliance |
| `signatures[]` | list | Rules |
| `parties` | object | KYC linkage (optional) |
| `effective_date` | date | Rule effective dating |

### 12.3 Adapters

| Adapter | Use case |
|---------|----------|
| Azure Document Intelligence | Forms, tables, signatures |
| Unstructured.io | Heterogeneous PDFs, email attachments |
| Native PDF text | Born-digital PDFs (cheaper path) |

### 12.4 Quality Gates

- OCR confidence below threshold → flag for human prep.
- Page count / size limits → reject or async batch queue.
- PII detection → redact in logs; full content in secured store only.

---

## 13. Durable Execution with Temporal

### 13.1 What Temporal Owns

- Retries with exponential backoff (OCR, external APIs).
- **Wait for human approval** up to N days (e.g. 5 business days).
- Timeout → escalate activity (notify supervisor, reassign task).
- Saga compensation (e.g. delete partial index on fatal failure).
- Scheduled re-validation (periodic compliance re-check).

### 13.2 Example: Human Wait

```text
Activity: WaitForHumanApproval
  timeout: 5 days
  on_timeout:
    - Flowable: escalate task to supervisor group
    - emit HumanApprovalTimeout event
  on_signal:
    - resume LangGraph checkpoint
    - continue pipeline
```

### 13.3 Workflow vs Activity Guidelines

| Use workflow for | Use activity for |
|------------------|------------------|
| Orchestration sequence | OCR call |
| Timers and signals | Rule engine HTTP call |
| Child workflows (per document) | Kafka publish |
| | LlamaIndex index build |

### 13.4 Anti-Patterns

- Storing full PDF bytes in Temporal workflow history.
- Embedding LLM prompts in workflow code (use versioned config service).
- Duplicating Flowable user task state inside Temporal (reference `taskId` only).

---

## 14. Workflow Governance with Flowable

### 14.1 Process Catalog (Starter)

| Process key | Purpose |
|-------------|---------|
| `document_review_v1` | Standard intake → compliance → QA |
| `consent_letter_fast_track` | Pre-classified consent, shortened path |
| `periodic_revalidation` | Scheduled re-run of rules on corpus |
| `rule_pack_promotion` | Human approval for new rule versions |

### 14.2 Variables (Case-Level)

- `caseId`, `tenantId`, `product`, `country`, `language`
- `documentIds[]`, `lastDecisionId`, `hitlRequired`
- `temporalWorkflowId`, `langGraphThreadId`

### 14.3 SLAs and Escalation

| SLA | Action |
|-----|--------|
| HITL not claimed in 4h | Notify pool lead |
| HITL not completed in 48h | Escalate to supervisor |
| Case not closed in 7d | Ops dashboard alert |

### 14.4 Integration with Temporal

- Flowable service task: `startTemporalWorkflow(caseId, definition)`.
- Flowable receive task or message: completed when Temporal signals.
- Human task completion → delegate calls `signalHumanTaskCompleted`.

---

## 15. Agent Orchestration with LangGraph

### 15.1 Graph Design

- **State:** `CaseAgentState` (document refs, rule results, compliance score, hitl_flag).
- **Nodes:** one per specialist agent + `human_gate` + `supervisor_route`.
- **Edges:** conditional on classification, language, rule failures, confidence.
- **Checkpointing:** Postgres or Redis; thread ID = `case_id`.

### 15.2 Supervisor Routing Logic (Pseudocode)

```python
def supervisor_route(state):
    if not state.extracted:
        return "document"
    if state.needs_translation:
        return "translation"
    if not state.rules_evaluated:
        return "rules"
    if not state.compliance_done:
        return "compliance"
    if state.hitl_required and not state.human_resolved:
        return "human_gate"  # external wait
    if not state.qa_generated:
        return "qa"
    if not state.tests_executed:
        return "execution"
    return "reporting"
```

### 15.3 Tool Policies

- Agents may call: rule-service, document-service, retrieval, calculator—not arbitrary network.
- PII fields masked in Langfuse traces per tenant policy.
- Max tokens / cost budget per case (supervisor enforces).

### 15.4 Failure Handling

| Failure | Behavior |
|---------|----------|
| Transient LLM | Retry 3x, then Temporal activity retry |
| Tool timeout | Supervisor marks degraded; optional HITL |
| Contradiction with rules | Force HITL, never auto-pass |

---

## 16. RAG and Memory with LlamaIndex

### 16.1 Indexes

| Index | Contents | Consumers |
|-------|----------|-----------|
| `policy-corpus` | Internal policies, regulator guidance | Compliance Agent |
| `rule-narratives` | Human-readable rule descriptions | Rule Agent explanations |
| `glossary` | Approved DE/FR/IT/EN terms | Translation Agent |
| `decision-history` | Past Decision Records (approved) | Similar-case retrieval |
| `hitl-feedback` | Human corrections | Reduces repeat errors |

### 16.2 Retrieval Snapshot

For audit, persist `retrieval_snapshot_id` listing chunk IDs and scores at decision time—do not rely on live index state for replay.

### 16.3 LlamaIndex Pipeline Components

- `SimpleDirectoryReader` / custom PDF reader → OCR output
- `NodeParser` (semantic chunks, clause-aware splitting where possible)
- `VectorStoreIndex` → Weaviate
- `QueryEngine` with metadata filters (`tenant_id`, `product`, `country`)

---

## 17. Eventing, Integration, and APIs

### 17.1 Domain Events (Kafka Topics)

| Topic | Event |
|-------|--------|
| `case.lifecycle` | CaseCreated, CaseCompleted, CaseEscalated |
| `document.lifecycle` | DocumentUploaded, DocumentExtracted |
| `compliance.decisions` | DecisionRecorded, HitlRequired |
| `qa.results` | TestCasesGenerated, TestRunCompleted |
| `rules.governance` | RulePackPublished |

### 17.2 External API (REST, Conceptual)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/cases` | Create case + upload metadata |
| POST | `/v1/cases/{id}/documents` | Attach document |
| GET | `/v1/cases/{id}` | Case status + decisions |
| GET | `/v1/cases/{id}/audit` | Export audit package |
| POST | `/v1/tasks/{id}/complete` | HITL completion (also via Flowable UI) |

### 17.3 Webhooks

- `onDecisionFinalized` → downstream QA tools / GRC systems.
- `onHitlRequired` → ticketing (ServiceNow, Jira).

---

## 18. Security, Identity, and Compliance

### 18.1 Authentication and Authorization

- Keycloak realms per environment; OIDC for UI and service accounts.
- Roles: `reviewer`, `supervisor`, `rule_admin`, `ops_readonly`, `api_integration`.
- Row-level security by `tenant_id` on all case and document queries.

### 18.2 Data Protection

- Encryption at rest (object store, DB, vectors).
- TLS everywhere; mTLS service-to-service in production.
- Field-level encryption for highly sensitive attributes.
- Data residency flags per tenant (CH/EU).

### 18.3 Audit and Retention

- Immutable audit log (WORM bucket or dedicated audit DB).
- Retention policies per document class (e.g. 10 years consent).
- Right-to-erasure only where legally permitted; otherwise legal hold.

### 18.4 Model Governance

- Approved model allowlist per environment.
- Prompt injection guards on document text.
- Output filters for PII leakage in logs.

---

## 19. Observability and Evaluation

### 19.1 Langfuse Instrumentation

- Trace per agent step with parent `case_id`.
- Log prompts, completions (redacted), tool calls, latency, token cost.
- Link `decision_id` ↔ `trace_id` in Decision Record.

### 19.2 Metrics (Prometheus / Dashboards)

| Metric | Purpose |
|--------|---------|
| `case_duration_seconds` | SLA tracking |
| `hitl_rate` | Automation health |
| `rule_failures_by_id` | Rule drift detection |
| `ocr_confidence_histogram` | Document quality |
| `calibrated_confidence_vs_outcome` | Threshold tuning |

### 19.3 Evaluation Sets

- Golden documents per product/country/language.
- Regression run on every rule pack and model version change.
- Multilingual parity tests (same semantic content, four languages).

### 19.4 Confidence Calibration

Combine:

- Raw model confidence
- Rule severity weights
- Retrieval score distribution
- Translation alignment score
- Historical false-positive rate for similar cases

Publish **calibrated** score to HITL gates—not raw LLM probability alone.

---

## 20. Deployment Topology

### 20.1 Environments

| Env | Purpose |
|-----|---------|
| `dev` | Local/docker-compose, mocked OCR |
| `test` | Integration + golden eval sets |
| `staging` | Production-like, anonymized data |
| `prod` | HA, multi-AZ, strict change control |

### 20.2 Reference Production Layout

```text
[K8s Cluster]
  - ingestion-api (HPA)
  - agent-runtime (GPU optional)
  - domain services (stateless)
  - Flowable (HA DB)
  - Temporal cluster (separate namespace)
  - Kafka cluster
  - Weaviate
  - Postgres (operational)
  - Object store (S3-compatible)
  - Keycloak
  - Langfuse
```

### 20.3 Non-Functional Targets (Initial)

| NFR | Target |
|-----|--------|
| Availability | 99.5% (business hours critical) |
| P95 case intake API | &lt; 500ms (excl. upload size) |
| OCR + extract (50 pg) | &lt; 10 min async |
| Audit write | Synchronous, &lt; 200ms |

---

## 21. Implementation Phases

### Phase 0 — Foundation (4–6 weeks)

- [ ] Decision Record schema + audit-service
- [ ] Keycloak, API gateway, case/create/upload APIs
- [ ] Object storage + Postgres
- [ ] Docker-compose dev stack (Flowable, Temporal, Kafka, Weaviate)

**Exit criteria:** Create case, store document, write audit entry.

### Phase 1 — Document + Rules (6–8 weeks)

- [ ] Document pipeline (OCR adapter, LlamaIndex index)
- [ ] Rule pack YAML format + rule-service evaluator
- [ ] Temporal: ingest + OCR + rule activities
- [ ] LangGraph: Document + Rule agents only

**Exit criteria:** End-to-end extract + rule report for one product/country.

### Phase 2 — Compliance + HITL (6–8 weeks)

- [ ] Compliance Agent + calibrated confidence
- [ ] Flowable user tasks + Temporal human wait
- [ ] Reviewer UI MVP
- [ ] Langfuse tracing

**Exit criteria:** Low-confidence case pauses, human completes, audit shows override.

### Phase 3 — QA + Execution (4–6 weeks)

- [ ] QA Agent test generation
- [ ] Execution Agent + result storage
- [ ] Kafka events + ops dashboard

**Exit criteria:** Generated tests run and link to Decision Record.

### Phase 4 — Enterprise Hardening (ongoing)

- [ ] Multi-tenant isolation, DR, performance testing
- [ ] Rule pack promotion workflow
- [ ] Eval regression gates in CI
- [ ] Optional integrations (GRC, DMS)

---

## 22. Reference: Consent-Letter Journey

### 22.1 Scenario

Wealth Management customer in Switzerland submits a **German** consent letter PDF.

### 22.2 Step-by-Step

| Step | System | Result |
|------|--------|--------|
| 1 | User uploads via API | `case_id`, `document_id`, hash stored |
| 2 | Flowable starts `document_review_v1` | Case `RECEIVED` |
| 3 | Temporal `IngestActivity` | Virus scan OK |
| 4 | Document Agent | Type=`consent_letter`, clauses extracted |
| 5 | Rule Agent | `CONSENT_102` FAIL (Block A missing) |
| 6 | Compliance Agent | confidence=0.71, risk=MEDIUM |
| 7 | Gate | HITL required (&lt; 0.85) |
| 8 | Flowable task | Reviewer sees mismatch + DE clause |
| 9 | Reviewer **Edit** | Adds corrected clause text |
| 10 | Re-run rules | PASS |
| 11 | QA Agent | 12 test cases for CONSENT_* rules |
| 12 | Execution Agent | Tests pass |
| 13 | Audit | Decision Record with human_override |
| 14 | Flowable end | Case `COMPLETED` |

### 22.3 Sample AI → Human Handoff JSON

```json
{
  "rule_id": "CONSENT_102",
  "confidence": 0.71,
  "issue": "Potential mismatch in translated clause",
  "recommended_action": "human_review"
}
```

---

## 23. Appendices

### A. Repository References

| Component | Repository |
|-----------|------------|
| LangGraph | https://github.com/langchain-ai/langgraph |
| LlamaIndex | https://github.com/run-llama/llama_index |
| Flowable | https://github.com/flowable/flowable-engine |
| Temporal | https://github.com/temporalio/temporal |

### B. Glossary

| Term | Definition |
|------|------------|
| **Case** | Business container for one or more documents through a governed process |
| **Decision Record** | Immutable unit of audit for one decision point |
| **Rule pack** | Versioned set of regulatory/business rules |
| **Calibrated confidence** | Score adjusted for risk, rules, and retrieval quality |
| **HITL** | Human-in-the-loop review step |

### C. Document Map (Suggested Repo Layout)

```text
framework/
├── FRAMEWORK.md                 # This plan
├── docs/
│   ├── architecture/
│   ├── api/
│   └── runbooks/
├── schemas/
│   ├── decision-record.json
│   ├── case.json
│   └── rule-pack.schema.json
├── processes/
│   └── flowable/                # BPMN exports
├── agents/
│   └── langgraph/               # Graph definitions
├── temporal/
│   └── workflows/               # Workflow + activities
├── rules/
│   └── packs/                   # YAML rule packs
└── eval/
    └── golden/                  # Test documents
```

### D. Open Decisions (Track in ADRs)

| ID | Question | Options |
|----|----------|---------|
| ADR-001 | Single vs multi Temporal namespace per tenant | Multi-tenant isolation vs ops complexity |
| ADR-002 | Flowable vs custom task UI for HITL | Flowable Task UI vs React reviewer app |
| ADR-003 | Primary OCR vendor | Azure DI vs Unstructured vs both |
| ADR-004 | Confidence threshold defaults | Global vs per product matrix |

---

## Summary

The enterprise-winning pattern is **intentional layering**: Flowable governs the case and people, Temporal guarantees durable execution and long waits, LangGraph coordinates specialist agents, and LlamaIndex powers document understanding and RAG—all unified by a **versioned Decision Record** and deterministic rule evaluation before probabilistic AI.

**Next recommended artifact:** JSON Schema files under `schemas/` for `decision-record`, `case`, and `rule-pack`, then a Phase 0 docker-compose for local integration testing.

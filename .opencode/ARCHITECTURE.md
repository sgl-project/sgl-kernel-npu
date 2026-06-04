# SGLang Ascend Agents Architecture Design

## 1. Top-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                    USER                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR AGENT                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐│
│  │ User Input  │  │ Task         │  │ Execution      │  │ State        ││
│  │ Receiver    │  │ Dispatcher   │  │ Tracker        │  │ Manager      ││
│  └─────────────┘  └──────────────┘  └────────────────┘  └───────────────┘│
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐                   │
│  │ Workflow    │  │ Human Review │  │ Global Context │                   │
│  │ Scheduler   │  │ Handler      │  │ Manager        │                   │
│  └─────────────┘  └──────────────┘  └────────────────┘                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EXECUTION LOG (Markdown)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  NOTE: Orchestrator serves as the MAIN AGENT (Coordinator):                │
│  - Receives user input directly                                           │
│  - Calls Task Analyzer to analyze and plan                                │
│  - Calls question tool for Human Review approval                          │
│  - Manages Global Context                                                 │
│  - Launches Executors and Verifier after approval                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │        [Invokes Task Analyzer as subagent]    │
              ▼                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TASK ANALYZER AGENT                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐│
│  │ Intent      │  │ Codebase     │  │ Task Type      │  │ Workflow     ││
│  │ Parser      │  │ Knowledge    │  │ Classifier     │  │ Designer     ││
│  └─────────────┘  └──────────────┘  └────────────────┘  └───────────────┘│
│  Output: Task Plan Document (Markdown) + Workflow Definition               │
│  (subagent, invoked by Orchestrator)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ writes task_plan.md
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR (continued)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Human Review Handler                                                 │   │
│  │ - Reads task_plan.md                                                │   │
│  │ - Calls question tool for approval                                  │   │
│  │ - On approval: launches Executors                                    │   │
│  │ - On rejection: returns to Task Analyzer or terminates              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                ▼                ▼
         ┌───────────────────┐ ┌───────────────┐ ┌───────────────┐
         │   TASK EXECUTOR   │ │  TASK EXECUTOR│ │  TASK EXECUTOR│
         │   (Agent A)       │ │  (Agent B)    │ │  (Agent C)    │
         │   [Parallel]      │ │  [Serial]     │ │  [Parallel]   │
         └───────────────────┘ └───────────────┘ └───────────────┘
                     │                │                │
                     └────────────────┴────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TASK VERIFIER AGENT                                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐│
│  │ Result      │  │ Quality      │  │ Failure        │  │ Report       ││
│  │ Comparator  │  │ Checker      │  │ Classifier     │  │ Generator    ││
│  └─────────────┘  └──────────────┘  └────────────────┘  └───────────────┘│
│  NOTE: Independent Agent, invoked by Orchestrator for verification       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PERSISTENT MEMORY LAYER                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐│
│  │ Working     │  │ Short-term   │  │ Long-term      │  │ Permanent     ││
│  │ Memory      │  │ Memory       │  │ Memory         │  │ Memory        ││
│  │ (RAM)       │  │ (Session+File│  │ (File+DB)     │  │ (Git+Docs)   ││
│  └─────────────┘  └──────────────┘  └────────────────┘  └───────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Execution Flow (Complete Lifecycle)

```
1. User → Orchestrator
   └── User submits task request

2. Orchestrator → Task Analyzer (subagent)
   └── Sends user input, requests analysis and planning

3. Task Analyzer → Orchestrator
   └── Generates task_plan.md, returns

4. Orchestrator (Human Review)
   └── Reads task_plan.md
   └── Calls question tool, awaits user approval

5. User Response
   └── Approved → Orchestrator launches Executors
   └── Rejected → Returns to Task Analyzer or terminates

6. Orchestrator → Executors
   └── Dispatches sub-tasks according to DAG schedule

7. Executors → Orchestrator
   └── Sub-tasks complete, report results

8. Orchestrator → Verifier
   └── Invokes verifier to validate results

9. Verifier → Orchestrator
   └── Returns verification report

10. Orchestrator → User
    └── Returns final result
```

## 2. Four-Layer Agent Detailed Design

### 2.0 Orchestrator Agent (Main Agent)

**Role**: Acts as the Main Agent, coordinates all other agents, serves as the primary user interface.

**Core Modules**:
| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| UserInputReceiver | User input | Structured request | Receives and parses user task requests |
| HumanReviewHandler | task_plan.md | Approval result | Invokes question tool for human review |
| GlobalContextManager | System state | Context update | Manages system-level context (capabilities, etc.) |
| WorkflowScheduler | Task Plan | Execution Schedule | Generates execution order and parallel plan |
| TaskDispatcher | Schedule | Task Assignments | Assigns tasks to appropriate Executors |
| ExecutionTracker | Task Status | State Updates | Tracks status of each sub-task |
| StateManager | Execution Events | Persistent State | Maintains global execution state |
| FailureCoordinator | Failure Reports | Retry/Escalation | Handles failure decisions |

**Execution Model** (Hybrid):
- **Parallel Execution**: Sub-tasks without dependencies execute simultaneously
- **Serial Execution**: Sub-tasks with dependencies or requiring sequential results
- **Dependency Resolution**: Automatic analysis of DAG dependencies

**Failure Handling**:
- Executor auto-retries N times (configurable, default 3)
- Each retry must be logged to Execution Log
- After N retries still failing → Report to Orchestrator
- Orchestrator decision: Skip / Replace Executor / Human Intervention

### 2.1 Task Analyzer Agent

**Role**: A **subagent** invoked by Orchestrator, responsible for parsing user input and designing workflows.

**Responsibilities**: Parse user input, design workflow, determine task type.

**Core Modules**:
| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| IntentParser | User intent (from Orchestrator) | Structured Intent | Parses task goals, constraints, context |
| CodebaseKnowledge | Task-related paths | Required codebase context | Loads relevant code on demand |
| TaskTypeClassifier | Intent + Context | Simple/Complex | Task classification |
| WorkflowDesigner | Task decomposition | Execution plan | DAG workflow definition |

**Output Format** (Task Plan Document):
```markdown
# Task Plan: [Task Name]

## Task Type
[Simple | Complex]

## Task Decomposition
### Step 1: [Sub-task Name]
- Skill Required: [skill_name]
- Executor: [agent_type]
- Parallel: [true/false]
- Retry Policy: [auto/manual]

### Step 2: ...

## Workflow
[DAG graphical description or YAML definition]

## Expected Output
[Task success criteria]

## Verification Criteria
[Verification checklist]
```

**Review Mechanism**:
- After Task Analyzer output, waits for Orchestrator to process
- Orchestrator reads task_plan.md and calls question tool for human review
- After approval, Orchestrator starts execution

### 2.2 Task Executor Agents

**Responsibilities**: Execute specific sub-tasks, invoke Skills/tools.

**Executor Type Breakdown**:

| Executor Type | Sub-Agents | Scope |
|---------------|------------|-------|
| **ModelOpsExecutor** | ModelLoader, ConfigValidator, EnvChecker | Model out-of-box, loading, configuration |
| **PerfOptimizerExecutor** | Profiler, KernelOptimizer, BottleneckAnalyzer | Performance profiling, kernel optimization |
| **AnalyzerExecutor** | ArchitectureParser, OperatorAnalyzer, MemoryAnalyzer | Model architecture, operators, memory analysis |
| **CodeDevExecutor** | CodeSearcher, PatchApplicator, TestRunner | Code development, bug fixes, testing |
| **ReviewExecutor** | CodeReviewer, CoverageAnalyzer, DocChecker | PR review, test coverage checking |

**Common Capabilities**:
- Unified Tool Calling interface
- Standardized Error/Failure reporting format
- Auto-retry mechanism (with logging)
- Intermediate result persistence

**Skill Integration**:
```
Executor
├── Skill: model_load
├── Skill: profiling
├── Skill: analyze_architecture
├── Skill: fix_bug
├── Skill: run_tests
└── ...
```

### 2.3 Task Verifier Agent

**Responsibilities**: Verify execution results, determine task success.

**Core Modules**:
| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| ResultComparator | Expected vs Actual | Match/Mismatch | Result comparison |
| QualityChecker | Execution artifacts | Quality Score | Quality scoring |
| FailureClassifier | Failure information | Failure Type | Failure classification |
| ReportGenerator | Verification Results | Markdown Report | Generate verification report |

**Verification Timing**:
- Immediate verification after each sub-task completes
- Final overall verification
- Configurable verification strictness

## 3. Context Architecture

### 3.1 Four-Layer Context Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        GLOBAL CONTEXT                            │
│  - Agent System Capabilities (capability descriptions)          │
│  - Available Skills Registry (skill list and interfaces)      │
│  - NPU/Ascend Hardware Properties (hardware knowledge)        │
│  - Supported Model List (validated models)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SESSION CONTEXT                           │
│  - Current Task (task description)                              │
│  - Conversation History                                        │
│  - Session State (running/paused/completed)                   │
│  - User Preferences                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TASK CONTEXT                            │
│  - Task Plan Document                                          │
│  - Relevant Codebase (loaded on demand)                        │
│  - Artifacts (code, logs, reports)                             │
│  - Sub-task States                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION CONTEXT                         │
│  - Current Tool Calls                                           │
│  - Intermediate Results                                        │
│  - Error/Retry Log                                            │
│  - Verifier Feedback                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Context Lifecycle Management

| Layer | Created | Destroyed | Storage |
|-------|---------|-----------|---------|
| Global Context | System Init | System Shutdown | Memory + Config Files |
| Session Context | Session Start | Session End | Memory + Session File |
| Task Context | Task Analyzer Complete | Task End | Memory + Task File |
| Execution Context | Executor Start | Executor End | Memory (temporary) |

## 4. Persistent Memory System

### 4.1 Memory Layer Architecture

```
                    ┌─────────────────────────────────────┐
                    │         MEMORY ACCESS LAYER         │
                    │  (Unified interface, abstracts      │
                    │   underlying storage differences)    │
                    └─────────────────────────────────────┘
          ┌─────────────┬─────────────┬─────────────┬─────────────┐
          ▼             ▼             ▼             ▼             ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Working  │ │Short-term│ │ Long-term│ │Permanent │ │Knowledge │
    │ Memory   │ │ Memory   │ │ Memory   │ │ Memory   │ │  Base    │
    │ (RAM)    │ │(Session) │ │ (File)   │ │ (Git)    │ │ (Search) │
    └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 4.2 Layer Detailed Design

**Working Memory (RAM)**:
- Content: Current context, active reasoning, immediate state
- Format: Python Objects (in-memory)
- Persistence: Not persisted, lost when session ends

**Short-term Memory (Session File)**:
- Content: Task plan, findings, progress, key decisions
- Format: Markdown files (task_plan.md, findings.md, progress.md)
- Persistence: Real-time file writes, recoverable across sessions

**Long-term Memory (File System)**:
- Content: Historical task summaries, successful optimization strategies, failure patterns
- Format: Markdown + JSON (structured index)
- Persistence: Retained across sessions, retrieved on demand

**Permanent Memory (Git + Docs)**:
- Content: Architecture decisions, Skill definition templates, code standards
- Format: Markdown documents, config files
- Persistence: Version controlled

**Knowledge Base (Searchable Store)**:
- Content: Code patterns, successful cases, technical documentation
- Format: Vector Index + Full-text Index
- Persistence: Long-term storage, supports semantic retrieval

### 4.3 File Structure Design

```
.opencode/                              # Framework code (aligned with planning-with-files skill)
├── agents/
│   ├── orchestrator.md                 # Main Agent definition
│   ├── analyzer.md                     # Task Analyzer definition
│   ├── executor.md                    # Executor system definition
│   └── verifier.md                    # Verifier definition
├── skills/
│   ├── model_load/
│   ├── profiling/
│   ├── analyze_architecture/
│   └── ...
├── templates/
│   ├── task_plan.md
│   ├── findings.md
│   └── progress.md
└── docs/                              # Runtime data (session memory)
    ├── sessions/
    │   └── [session_id]/
    │       ├── task_plan.md
    │       ├── findings.md
    │       ├── progress.md
    │       └── execution_logs/
    │           └── [step_id]_retry_log.md
    ├── longterm/
    │   ├── task_history/
    │   ├── strategies/
    │   └── patterns/
    └── knowledge_base/
        ├── code_patterns/
        └── successful_cases/
```

## 5. Skill Abstraction Design

### 5.1 Skill Definition

Each Skill is an independent functional unit containing:

| Component | Description |
|-----------|-------------|
| SKILL.md | Skill description, interface definition, usage |
| Templates | Input/output templates |
| Scripts | Executable scripts (if any) |
| Tests | Skill's own tests |

### 5.2 Skill Categories

| Category | Skills | Executor Binding |
|----------|--------|------------------|
| **Model Ops** | model_load, env_check, config_validate | ModelOpsExecutor |
| **Performance** | profiling, benchmark, kernel_optimize | PerfOptimizerExecutor |
| **Analysis** | analyze_architecture, analyze_memory, analyze_operator | AnalyzerExecutor |
| **Code Dev** | fix_bug, write_test, apply_patch | CodeDevExecutor |
| **Review** | code_review, test_coverage | ReviewExecutor |

## 6. Workflow Orchestration Mechanism

### 6.1 Task Classification and Execution Strategy

| Task Type | Criteria | Execution Strategy |
|-----------|----------|-------------------|
| **Simple Task** | Single Skill completes, no complex dependencies | Direct Skill call, Executor executes, Verifier validates |
| **Complex Task** | Multiple Skills required, DAG dependencies | Task Analyzer decomposes, Orchestrator orchestrates, hybrid execution |

### 6.2 DAG Workflow Definition

```yaml
workflow:
  name: "New Model Out-of-Box"
  type: "complex"

  steps:
    - id: step_1
      name: "Environment Check"
      skill: "env_check"
      executor: "ModelOpsExecutor"
      parallel_group: "init"

    - id: step_2
      name: "Model Loading"
      skill: "model_load"
      executor: "ModelOpsExecutor"
      parallel_group: "init"
      depends_on: ["step_1"]

    - id: step_3
      name: "Config Validation"
      skill: "config_validate"
      executor: "ModelOpsExecutor"
      depends_on: ["step_2"]

    - id: step_4
      name: "Basic Function Test"
      skill: "run_tests"
      executor: "CodeDevExecutor"
      verifier: "TaskVerifier"
      depends_on: ["step_3"]

    - id: step_5
      name: "Performance Benchmark"
      skill: "profiling"
      executor: "PerfOptimizerExecutor"
      parallel_group: "perf"
      depends_on: ["step_4"]

    - id: step_6
      name: "Architecture Analysis"
      skill: "analyze_architecture"
      executor: "AnalyzerExecutor"
      parallel_group: "perf"
      depends_on: ["step_4"]

  execution:
    parallel_groups:
      init: ["step_1", "step_2"]
      perf: ["step_5", "step_6"]
    retry_policy:
      max_attempts: 3
      backoff: "exponential"
```

### 6.3 Execution State Machine

```
                    ┌─────────────────────────────────────────────────┐
                    │                      IDLE                       │
                    │           (Awaiting new task)                  │
                    └─────────────────────────────────────────────────┘
                                        │
                                        ▼ Task Assigned
                    ┌─────────────────────────────────────────────────┐
                    │                    ANALYZING                   │
                    │         (Task Analyzer in progress)             │
                    └─────────────────────────────────────────────────┘
                                        │
                                        ▼ Plan Ready
                    ┌─────────────────────────────────────────────────┐
                    │              AWAITING_APPROVAL                 │
                    │         (Human review - Sync Block)            │
                    └─────────────────────────────────────────────────┘
                                        │
                                        ▼ Approved
                    ┌─────────────────────────────────────────────────┐
                    │                  ORCHESTRATING                 │
                    │           (Orchestrator scheduling)              │
                    └─────────────────────────────────────────────────┘
                                        │
                                        ▼ Schedule Created
                    ┌─────────────────────────────────────────────────┐
                    │                    RUNNING                      │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │ SUB-TASK STATES:                          │   │
                    │  │ - step_1: COMPLETED                      │   │
                    │  │ - step_2: RUNNING                         │   │
                    │  │ - step_3: PENDING (waiting for step_2)    │   │
                    │  │ - step_4: PENDING                         │   │
                    │  │ - step_5: PENDING                         │   │
                    │  │ - step_6: PENDING                         │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌─────────────────────┐               ┌─────────────────────┐
        │       COMPLETED      │               │        FAILED        │
        │  (All sub-tasks done) │               │  (Failed, unrecoverable) │
        └─────────────────────┘               └─────────────────────┘
                    │                                       │
                    ▼                                       ▼
        ┌─────────────────────┐               ┌─────────────────────┐
        │       VERIFYING      │               │     ESCALATING      │
        │    (Task Verifier)   │               │  (Human intervention) │
        └─────────────────────┘               └─────────────────────┘
```

## 7. Core Data Structures

### 7.1 Task Plan Document

```markdown
# Task Plan: [Task Name]

## Task Metadata
- Task ID: [uuid]
- Task Type: [Simple | Complex]
- Created: [timestamp]
- Session: [session_id]

## Task Classification
- Business Activity: [New Model OOB | Performance Optimization | ...]
- Complexity Score: [1-10]
- Estimated Duration: [time estimate]

## Sub-tasks
| Step | Name | Skill | Executor | Parallel Group | Depends On | Retry Policy |
|------|------|-------|----------|----------------|------------|--------------|
| 1    | ...  | ...   | ...      | init           | -          | auto:3       |
| 2    | ...  | ...   | ...      | init           | step_1     | auto:3       |

## Verification Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| ...      | ...       |

## Errors Encountered
| Error | Step | Attempt | Resolution |
|-------|------|---------|------------|
| ...   | ...  | ...     | ...        |
```

### 7.2 Execution Log Entry

```markdown
## Execution Log: [Task ID]

### Session: [timestamp]

#### Step [N]: [Step Name]
- **Status**: [pending | running | completed | failed]
- **Started**: [timestamp]
- **Finished**: [timestamp]
- **Executor**: [agent_id]
- **Retry Log**:
  - Attempt 1: [action] → [result]
  - Attempt 2: [action] → [result]
- **Artifacts**:
  - [file_path_1]
  - [file_path_2]
- **Errors**:
  - [Error Description]
```

## 8. Skill Development Framework

### 8.1 Skill Definition Specification

Each Skill should contain:

```
skill_name/
├── SKILL.md              # Skill definition file
├── templates/
│   ├── input_template.md
│   └── output_template.md
├── scripts/
│   └── [helper_scripts]
└── tests/
    └── test_skill.py
```

## 9. Key Technical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent Abstraction | 4-layer (Analyzer/Orchestrator/Executor/Verifier) | Separation of concerns, supports independent evolution |
| Execution Model | Hybrid mode | Flexible adaptation based on task type |
| Failure Handling | Auto-retry + Report + Log | Balances automation and control |
| Human Review | Sync blocking | Ensures planning correctness, avoids resource waste |
| Storage Format | Plain text Markdown | Aligns with planning-with-files philosophy, human readable |
| Skill Reuse | Agent-Skill separation | Skills evolve independently, specialized accumulation |
| Persistence Strategy | Layered Memory (Working/Short-term/Long-term/Permanent) | Distinguishes volatile/permanent, optimizes retrieval |

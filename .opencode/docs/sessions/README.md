# Sessions Directory

Contains session-specific data including task plans, findings, and execution logs.

## Directory Structure
```
sessions/
└── [session_id]/
    ├── task_plan.md
    ├── findings.md
    ├── progress.md
    └── execution_logs/
        └── [step_id]_retry_log.md
```

## Usage
- Each user session gets a unique session_id
- All session data is stored here for persistence
- Supports session recovery via planning files

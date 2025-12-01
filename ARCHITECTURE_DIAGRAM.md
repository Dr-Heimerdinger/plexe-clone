# Agent Thinking Feature - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser (Frontend)                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Chat Component                               │ │
│  │  - Displays user messages                                       │ │
│  │  - Displays assistant responses                                 │ │
│  │  - Displays thinking messages (NEW!)                            │ │
│  │    ┌──────────────────────────────────────────────┐            │ │
│  │    │ Thinking Message Display:                     │            │ │
│  │    │ ┌────────────────────────────────────────┐   │            │ │
│  │    │ │ Agent Name        Step Number          │   │            │ │
│  │    │ ├────────────────────────────────────────┤   │            │ │
│  │    │ │ Thinking content...                    │   │            │ │
│  │    │ └────────────────────────────────────────┘   │            │ │
│  │    └──────────────────────────────────────────────┘            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▲                                       │
│                              │ WebSocket Messages                   │
│                              │ {type: "thinking", agent_name, ...}  │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                         FastAPI Server                               │
│  ┌────────────────────────────┴──────────────────────────────────┐  │
│  │                  WebSocket Endpoint (/ws)                      │  │
│  │                                                                 │  │
│  │  1. Accept connection                                          │  │
│  │  2. Create WebSocketEmitter ──┐                                │  │
│  │  3. Create ConsoleEmitter     ├─→ MultiEmitter ──┐             │  │
│  │                               │                   │             │  │
│  │  4. Create ChainOfThoughtCallable ←───────────────┘             │  │
│  │                 │                                               │  │
│  │                 ▼                                               │  │
│  │  5. Initialize ConversationalAgent(chain_of_thought_callable)  │  │
│  │                 │                                               │  │
│  │                 ▼                                               │  │
│  │  6. agent.run(user_message) ──────────────────┐                │  │
│  │                                                │                │  │
│  └────────────────────────────────────────────────┼────────────────┘  │
└─────────────────────────────────────────────────┼────────────────────┘
                                                  │
┌─────────────────────────────────────────────────┼────────────────────┐
│                    Agent Execution Layer         │                    │
│  ┌───────────────────────────────────────────────▼─────────────────┐ │
│  │              ConversationalAgent                                 │ │
│  │  (ToolCallingAgent with step_callbacks)                         │ │
│  │                                                                  │ │
│  │  For each step in reasoning:                                    │ │
│  │    1. Generate thought                                          │ │
│  │    2. Call tool (optional)                                      │ │
│  │    3. Observe result                                            │ │
│  │    4. ────→ Trigger step_callbacks ────┐                        │ │
│  │                                         │                        │ │
│  └─────────────────────────────────────────┼────────────────────────┘ │
└──────────────────────────────────────────┼─────────────────────────────┘
                                           │
┌──────────────────────────────────────────┼─────────────────────────────┐
│           Chain of Thought System        │                             │
│  ┌────────────────────────────────────────▼──────────────────────┐    │
│  │         ChainOfThoughtCallable.__call__(step, agent)          │    │
│  │                                                                │    │
│  │  1. Extract step summary (StepSummary)                        │    │
│  │     - agent_name                                              │    │
│  │     - step_number                                             │    │
│  │     - model_output (thought)                                  │    │
│  │     - tool_calls                                              │    │
│  │     - observations                                            │    │
│  │                                                                │    │
│  │  2. Generate friendly summary                                 │    │
│  │                                                                │    │
│  │  3. Emit to MultiEmitter ───────────────┐                     │    │
│  │                                          │                     │    │
│  └──────────────────────────────────────────┼─────────────────────┘    │
│                                             │                          │
│  ┌──────────────────────────────────────────▼─────────────────────┐   │
│  │                      MultiEmitter                              │   │
│  │                                                                 │   │
│  │  ┌──────────────────────┐    ┌──────────────────────┐         │   │
│  │  │  WebSocketEmitter    │    │  ConsoleEmitter      │         │   │
│  │  │                      │    │                      │         │   │
│  │  │  emit_thought():     │    │  emit_thought():     │         │   │
│  │  │  - Create payload    │    │  - Format with Rich  │         │   │
│  │  │  - Send via WS ──────┼────┼─→ Print to console   │         │   │
│  │  │    asyncio.create_   │    │                      │         │   │
│  │  │    task()            │    │                      │         │   │
│  │  └──────────────────────┘    └──────────────────────┘         │   │
│  │           │                                                    │   │
│  └───────────┼────────────────────────────────────────────────────┘   │
└──────────────┼─────────────────────────────────────────────────────────┘
               │
               ▼
     WebSocket.send_json({
       "type": "thinking",
       "role": "thinking", 
       "agent_name": "...",
       "message": "...",
       "step_number": N
     })
               │
               ▼
     Frontend receives and displays
     in thinking-bubble component
```

## Data Flow Summary

1. **User sends message** → WebSocket → Server
2. **Server initializes** → ConversationalAgent with chain_of_thought_callable
3. **Agent processes** → Each step triggers callback
4. **ChainOfThoughtCallable** → Extracts step info → Emits to MultiEmitter
5. **MultiEmitter** → 
   - WebSocketEmitter → Send to browser (async)
   - ConsoleEmitter → Print to server console
6. **Frontend receives** → Displays thinking message in UI
7. **Agent completes** → Final response sent to user

## Key Components

### Backend
- `WebSocketEmitter`: Sends thinking messages via WebSocket
- `ChainOfThoughtCallable`: Intercepts agent steps
- `MultiEmitter`: Broadcasts to multiple outputs
- `ConversationalAgent`: Main agent with callback support

### Frontend  
- `Message` component: Renders different message types
- `thinking-bubble` CSS: Styles for thinking messages
- WebSocket listener: Receives and displays messages

## Message Types

1. **User Message**: `{role: "user", content: "..."}`
2. **Thinking Message**: `{role: "thinking", agent_name: "...", message: "...", step_number: N}`
3. **Assistant Response**: `{role: "assistant", content: "..."}`

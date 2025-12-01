# Agent Thinking Display Feature

## Tổng quan (Overview)

Tính năng này hiển thị quá trình suy nghĩ (thinking process) của các agent cùng với tên agent đang thực hiện trên giao diện chat theo thời gian thực.

This feature displays the thinking processes of agents along with their names in real-time on the chat interface.

## Các thay đổi chính (Main Changes)

### 1. WebSocket Emitter mới (New WebSocket Emitter)
**File**: `/plexe/internal/common/utils/chain_of_thought/websocket_emitter.py`

- Tạo `WebSocketEmitter` class kế thừa từ `ChainOfThoughtEmitter`
- Gửi messages về thinking process của agent qua WebSocket
- Xử lý async context để tương thích với FastAPI WebSocket

**Chức năng**:
- Broadcast agent thinking messages đến WebSocket clients
- Theo dõi step count cho mỗi bước suy nghĩ
- Xử lý gracefully khi gọi từ synchronous context

### 2. Cập nhật Server (Server Updates)
**File**: `/plexe/server.py`

**Thay đổi**:
- Import `WebSocketEmitter`, `ChainOfThoughtCallable`, `MultiEmitter`, `ConsoleEmitter`
- Tạo multi-emitter kết hợp WebSocket và Console output
- Khởi tạo `ConversationalAgent` với chain of thought callback
- Agent giờ sẽ phát cả thinking messages và response messages

**Luồng hoạt động**:
```
User Message → Agent Processing → Thinking Steps (via WebSocket) → Final Response
```

### 3. Cập nhật Conversational Agent
**File**: `/plexe/agents/conversational.py`

**Thay đổi**:
- Thêm parameter `chain_of_thought_callable` vào constructor
- Truyền callback vào `ToolCallingAgent` qua `step_callbacks`
- Agent bây giờ emit thinking messages trong quá trình xử lý

### 4. Cập nhật Frontend (Vite/React)
**File**: `/plexe/ui/frontend/src/components/Chat.jsx`

**Thay đổi**:
- Cập nhật `Message` component để xử lý `thinking` role
- Hiển thị thinking messages với format đặc biệt:
  - Agent name
  - Step number
  - Thinking content
- Giữ nguyên display cho user và assistant messages

### 5. Cập nhật CSS Styling
**File**: `/plexe/ui/frontend/src/styles.css`

**Thêm styles cho**:
- `.message.thinking` - container cho thinking messages
- `.thinking-bubble` - bubble với gradient background
- `.thinking-header` - header hiển thị agent name và step
- `.thinking-content` - nội dung thinking message
- `.agent-name` và `.step-number` - styling cho metadata

**Thiết kế**:
- Gradient background (blue tones)
- Border-left accent color
- Compact font size
- Centered alignment

### 6. Cập nhật Legacy UI (index.html)
**File**: `/plexe/ui/index.html`

**Thay đổi**:
- Cập nhật `Message` component để hỗ trợ thinking messages
- Sử dụng Tailwind CSS classes cho styling
- Tương tự layout như Vite frontend

### 7. Cập nhật Module Exports
**File**: `/plexe/internal/common/utils/chain_of_thought/__init__.py`

**Thay đổi**:
- Export `WebSocketEmitter` để có thể import
- Thêm vào `__all__` list

## Cấu trúc Message (Message Structure)

### Thinking Message Format
```json
{
  "type": "thinking",
  "role": "thinking",
  "agent_name": "ModelDefinitionAssistant",
  "message": "💭 Thought: Analyzing user requirements...",
  "step_number": 1
}
```

### Regular Message Format
```json
{
  "role": "assistant",
  "content": "I can help you build a model...",
  "id": "uuid-here"
}
```

## Cách hoạt động (How It Works)

1. **User gửi message** → WebSocket server nhận request
2. **Server khởi tạo**:
   - WebSocketEmitter (gửi đến UI)
   - ConsoleEmitter (log ra console)
   - MultiEmitter (kết hợp cả hai)
   - ChainOfThoughtCallable (callback handler)
3. **Agent xử lý**:
   - Mỗi step trong quá trình reasoning
   - Callback được gọi với step information
   - WebSocketEmitter gửi thinking message đến UI
4. **Frontend nhận và hiển thị**:
   - Thinking messages xuất hiện theo thời gian thực
   - Mỗi message hiển thị agent name và step number
   - Final response hiển thị như bình thường

## Ví dụ UI (UI Example)

```
┌────────────────────────────────────────┐
│ ModelDefinitionAssistant · Step 1     │
│ 💭 Thought: Understanding the request  │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ ModelDefinitionAssistant · Step 2     │
│ 🔧 Tool: validate_dataset_files(...)   │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Assistant Response                     │
│ I can help you with that...           │
└────────────────────────────────────────┘
```

## Testing

Để test tính năng:

1. Khởi động server:
   ```bash
   cd /home/admin1/plexe-clone
   docker compose -f docker-compose.dev.yml up -d
   ```

2. Mở browser và truy cập UI

3. Gửi một message yêu cầu model building

4. Quan sát thinking messages xuất hiện theo thời gian thực

## Notes

- Thinking messages được gửi qua WebSocket trong async context
- Console output vẫn hoạt động song song để debugging
- Frontend tự động scroll xuống khi có messages mới
- CSS responsive cho mobile devices
- Hỗ trợ cả Vite build và legacy HTML UI

## Future Enhancements

Có thể cải thiện thêm:
- Toggle để ẩn/hiện thinking messages
- Filter theo agent name
- Export thinking log
- Collapse/expand thinking details
- Syntax highlighting cho code trong thinking messages

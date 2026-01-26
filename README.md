# Plexe: Agentic ML Framework with MCP Integration

Plexe là một framework đa tác tử (multi-agent) được xây dựng trên nền tảng **LangGraph**, được thiết kế để tự động hóa toàn bộ quy trình xây dựng mô hình Machine Learning từ ngôn ngữ tự nhiên.

Hệ thống đã được nâng cấp với **Model Context Protocol (MCP)** để mở rộng khả năng kết nối với các công cụ học thuật và dữ liệu bên ngoài một cách chuẩn hóa.

## 🚀 Các tính năng chính sau điều chỉnh

1.  **Kiến trúc Đa tác tử LangGraph**: Điều phối luồng công việc giữa các Agent chuyên biệt (EDA, Dataset Builder, Task Builder, GNN Specialist).
2.  **Tích hợp MCP (Model Context Protocol)**:
    *   **Google Scholar**: Tìm kiếm bài báo khoa học, trích xuất thông tin tác giả trực tiếp qua MCP.
    *   **Kaggle**: Tìm kiếm và tải xuống tập dữ liệu từ Kaggle API thông qua MCP server.
    *   **Khả năng mở rộng**: Dễ dàng thêm các MCP server mới chỉ bằng cách cập nhật `mcp_config.json`.
3.  **Hỗ trợ GPU**: Tối ưu hóa cho việc huấn luyện Graph Neural Networks (GNNs) sử dụng CUDA.

## 🏗️ Cấu trúc hệ thống MCP

*   `mcp_config.json`: Cấu hình danh sách các MCP server và tham số khởi chạy.
*   `plexe/langgraph/mcp_manager.py`: Quản lý kết nối, khám phá tools và chuyển đổi MCP tools thành LangChain tools.
*   `plexe/langgraph/mcp_servers/`: Thư mục chứa các tùy chỉnh MCP server (Scholar, Kaggle).

## 🛠️ Cài đặt & Sử dụng

### 1. Cấu hình biến môi trường
Tạo file `.env` hoặc cập nhật `docker-compose.gpu.yml` với các thông tin sau:
```env
# LLM Keys
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key

# Kaggle (Bắt buộc cho Kaggle MCP tool)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 2. Chạy với Docker
Sử dụng Docker Compose để khởi chạy toàn bộ hệ thống (bao gồm MLflow, Postgres, và Plexe Backend):
```bash
docker compose -f docker-compose.gpu.yml up -d
```

### 3. Cách thức hoạt động của Agent
Mọi Agent kế thừa từ `BaseAgent` sẽ tự động tải các tools từ các MCP server được cấu hình trong `mcp_config.json`. Bạn có thể yêu cầu Agent trong Chat UI:
- *"Tìm các bài báo mới nhất về GNN trên Google Scholar"*
- *"Tải tập dữ liệu Titanic từ Kaggle và phân tích nó"*

## 📝 Ghi chú cho Docker
Hệ thống sử dụng `Dockerfile.gpu` dựa trên `pytorch/pytorch:2.7.0-cuda12.8` để đảm bảo hiệu suất huấn luyện mô hình. Các thư viện bổ sung như `scholarly`, `kaggle`, và `mcp[all]` đã được tích hợp sẵn trong quá trình build image.

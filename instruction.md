# Tài liệu Kỹ thuật Mô hình KAM-CoT (Knowledge Augmented Multimodal Chain-of-Thoughts)

Tài liệu này tổng hợp cấu trúc kiến trúc, cấu hình tensor, các mô hình tiền huấn luyện và cơ chế tương tác đa phương thức của hệ thống KAM-CoT phục vụ cho việc tự động lập trình (coding).

---

## 1. Kiến trúc Mô hình, Luồng Input/Output & Số Chiều (Dimensions)

 Mô hình KAM-CoT bao gồm cấu trúc **Encoder-Decoder** kết hợp với mạng nơ-ron đồ thị (GNN) và các khối chú ý chéo (Cross-Attention) để xử lý đồng thời 3 luồng dữ liệu: Văn bản (Text), Hình ảnh (Image), và Đồ thị tri thức (Knowledge Graph - KG).

### Ký hiệu kích thước chung:
-  $d$: Kích thước embedding ẩn của mô hình ngôn ngữ (Ví dụ với T5-Base / FLAN-T5-Base, $d = 768$).
-  $n$: Số lượng tokens trong chuỗi văn bản đầu vào
-  $m$: Số lượng vùng patch của hình ảnh sau khi qua Vision Encoder.
-  $p$: Số lượng thực thể (nodes) được trích xuất trong đồ thị con ($p \le 200$).

### Chi tiết các thành phần trong Encoder:

#### A. Bộ mã hóa văn bản (Language Encoder)
-  **Input:** Chuỗi văn bản $X_{lang}$ (đã gộp câu hỏi, ngữ cảnh, các lựa chọn đáp án và/hoặc chuỗi suy luận rationale). Kích thước: `(batch_size, n)`.
-  **Output:** Tensor trạng thái ẩn $H_{lang}$.  Kích thước: `(batch_size, n, d)`.

#### B. Bộ mã hóa hình ảnh (Image Encoder)
-  **Input:** Hình ảnh đầu vào $X_{img}$.
-  **Output:** Trích xuất các đặc trưng patch ẩn và nhân với ma trận chiếu đường thẳng $W_{img}$ để đưa về cùng kích thước ẩn với văn bản ($d$).
  - Đặc trưng gốc: `(batch_size, m, d_img_original)`
  -  Output cuối ($H_{img}$): `(batch_size, m, d)`.

#### C. Bộ trích xuất & Mã hóa đồ thị (Graph Extraction & Graph Encoder)
- **Quy trình trích xuất đồ thị con ($X_{kg}$):**
  1.  Sử dụng các thực thể xuất hiện trong câu hỏi, ngữ cảnh, đáp án (và ảnh nếu có) để định vị các node tương ứng trên ConceptNet.
  2.  Mở rộng đồ thị bằng cách lấy các node chung trong vòng 1-hop và 2-hop lân cận để kết nối các node ban đầu với nhau (độ dài đường đi tối đa bằng 2)
  3.  Cắt tỉa (Pruning) để giữ lại tối đa $p = 200$ nodes cho mỗi mẫu dữ liệu.  Nếu ít hơn, thực hiện zero-padding.
- **Sử dụng dữ liệu của ConceptNet:** Sử dụng bằng cách tải db về local và sử dụng thư viện conceptnet-lite để lấy ra thông tin
-  **Nhúng node ban đầu (Initial Node Embeddings):** Lấy từ checkpoint của Language Encoder bằng cách tính trung bình cộng embedding các token cấu thành tên node.  Kích thước ban đầu: `(batch_size, p, d)`.
-  **Nhúng cạnh (Edge Embeddings):** ConceptNet có 17 loại quan hệ, xét hai chiều (xuôi/ngược) tổng cộng có 34 loại thuộc tính cạnh.  Một bảng ma trận nhúng (Embedding Table) được huấn luyện để chuyển đổi loại cạnh sang vector có kích thước $e_{edge} = 64$.  Kích thước: `(34, 64)`.
-  **Cấu trúc Graph Encoder:** Gồm 2 tầng xử lý bằng PyTorch Geometric:
  1.  *Tầng 1 (Relational Graph Attention Layer - RGAT):* Tiếp nhận node features kích thước $d$ và edge features kích thước $64$.  Output: `(batch_size, p, d)`.
  2.  *Tầng 2 (Graph Convolutional Network - GCN):* Nhận output tầng 1.  Output cuối ($H_{kg}$): `(batch_size, p, d)`.

#### D. Khối tương tác đa phương thức (Cross-Attention Modules)
 Sử dụng hai module Chú ý chéo đơn đầu (Single-headed Cross-Attention), lấy văn bản ($H_{lang}$) làm truy vấn (Query):
1. **Language-Image Attention:**
   $$\text{Query} = H_{lang}, \text{Key} = H_{img}, \text{Value} = H_{img}$$
   -  Output ($H_{img}^{attn}$): `(batch_size, n, d)`.
2. **Language-Knowledge Graph Attention:**
   $$\text{Query} = H_{lang}, \text{Key} = H_{kg}, \text{Value} = H_{kg}$$
   -  Output ($H_{kg}^{attn}$): `(batch_size, n, d)`.

#### E. Khối hợp nhất đặc trưng (Gated Fusion - Fusion-1)
 Sử dụng cơ chế cổng điều hướng (gated fusion) để tính toán trọng số động $\alpha, \beta, \gamma$ cho từng vị trí token văn bản:
-  **Các phép chiếu tuyến tính (với $W_1$ đến $W_9 \in \mathbb{R}^{d \times d}$):** 
   $$S_{\alpha} = H_{lang}W_1 + H_{img}^{attn}W_2 + H_{kg}^{attn}W_3 \in \mathbb{R}^{n \times d}$$ 
   $$S_{\beta} = H_{lang}W_4 + H_{img}^{attn}W_5 + H_{kg}^{attn}W_6 \in \mathbb{R}^{n \times d}$$ 
   $$\gamma' = H_{lang}W_7 + H_{img}^{attn}W_8 + H_{kg}^{attn}W_9 \in \mathbb{R}^{n \times d}$$ 
-  **Tính toán trọng số qua Softmax dọc theo trục modality tại mỗi phần tử (element-wise):** 
   $$\alpha_{ij}, \beta_{ij}, \gamma_{ij} = \text{softmax}([S_{\alpha_{ij}}, S_{\beta_{ij}}, S_{\gamma_{ij}}])$$ 
-  **Output kết hợp ($H_{fuse}$):** Kích thước `(batch_size, n, d)`.
   $$H_{fuse} = \alpha \cdot H_{lang} + \beta \cdot H_{img}^{attn} + \gamma \cdot H_{kg}^{attn}$$ 

### Chi tiết Decoder:

#### Transformer Decoder
-  **Input:** Khối đặc trưng đã hợp nhất $H_{fuse}$ từ Encoder `(batch_size, n, d)` và chuỗi token đích đã được dịch phải (shifted right) `(batch_size, target_len)`.
-  **Output:** Phân phối xác suất của các token văn bản tiếp theo (`Output probabilities`).  Kích thước: `(batch_size, target_len, vocab_size)`.

---

## 2. Danh sách các Mô hình Pretrained sử dụng trong các thành phần

| Thành phần mô hình | Tên Kiến trúc / Bộ mã hóa mặc định | Checkpoint mã nguồn mở (HuggingFace / PyTorch) |
| :--- | :--- | :--- |
| **Language Backbone** (Encoder & Decoder) |  T5-Base (254M) hoặc **FLAN-T5-Base** (280M)  |  `google/flan-t5-base`   <br> `allenai/unifiedqa-t5-base`  |
| **Vision Encoder** |  **DETR** (Mặc định - cho kết quả tốt nhất)   <br> Hoặc CLIP  |  `facebook/detr-resnet-101-dc5` (Bỏ đầu phân loại)   <br> `google/vit-base-patch16-384`  |
| **Image Captioning** (Bổ trợ ngữ cảnh) |  ViT-GPT2  |  `nlpconnect/vit-gpt2-image-captioning`  |
| **Cơ sở dữ liệu Đồ thị ngoại vi** |  ConceptNet 5.5  |  Tải và cấu trúc hóa dưới dạng Triplets mạng cục bộ. |

---

## 3. Cơ chế tương tác và Quy trình hoạt động 2 giai đoạn (2-Stage Training)

 Mô hình phân rã quá trình suy luận thành hai giai đoạn tuần tự nối tiếp nhau sử dụng chung một cấu trúc kiến trúc nhưng được khởi tạo và huấn luyện độc lập (Decoupled 2-Stage).
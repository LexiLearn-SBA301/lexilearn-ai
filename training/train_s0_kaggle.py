# =============================================================================
# TRAIN S0 — Fine-tune Qwen2.5-3B-Instruct trên Kaggle (QLoRA + Unsloth)
# Bản TỐI ƯU — có eval, kiểm tra mask, export GGUF cho Ollama.
# =============================================================================
# CÁCH CHẠY TRÊN KAGGLE (làm đúng thứ tự):
#   1. Tạo Notebook mới. Bên phải: Settings ->
#        - Accelerator: GPU T4 x2 (hoặc P100). 3B + QLoRA chỉ cần 1 GPU.
#        - Internet: ON  (bắt buộc: pip install + tải model từ HuggingFace)
#   2. Bấm "+ Add Input" -> Upload -> chọn train.jsonl (và dev.jsonl nếu muốn eval).
#        Kaggle đặt ở: /kaggle/input/<ten-dataset>/train.jsonl
#        -> Sửa DATA_PATH / DEV_PATH bên dưới cho khớp đường dẫn THẬT.
#   3. Copy TỪNG "CELL" vào TỪNG ô notebook, chạy lần lượt từ trên xuống.
#
# LƯU Ý: cách cài Unsloth trên Kaggle đôi khi đổi theo phiên bản. Nếu CELL 1
# lỗi version -> mở github.com/unslothai/unsloth phần "Kaggle" lấy lệnh mới.
# =============================================================================


# ===================== CELL 1 — Cài thư viện =================================
# Dùng "!" (magic của notebook) để thấy log cài đặt. Nếu chạy file .py thuần
# (không phải notebook) thì đổi lại thành os.system(...).
!pip install -q unsloth
# Fallback nếu bản trên lỗi (bỏ dấu # để dùng):
# !pip install -q --no-deps "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q --no-deps trl peft accelerate bitsandbytes


# ===================== CELL 2 — Cấu hình (chỉ chỉnh ở đây) ===================
# Đây là chỗ DUY NHẤT bạn cần sửa cho mỗi lần chạy.

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"   # bản Unsloth tối ưu sẵn cho 4bit

# --- Đường dẫn data (SỬA cho khớp nơi bạn upload) ---
DATA_PATH  = "/kaggle/input/test-s0/train.jsonl"   # tập huấn luyện (bắt buộc)
DEV_PATH   = "/kaggle/input/test-s0/dev.jsonl"     # tập kiểm định (chỉ dùng nếu USE_EVAL=True)
OUTPUT_DIR = "/kaggle/working/qwen2.5-3b-s0-lora"  # nơi lưu adapter

# --- Siêu tham số (hyperparameters) ---
MAX_SEQ_LEN = 2048    # độ dài tối đa 1 mẫu (token). S0 ngắn nên 2048 dư dùng.
LORA_R      = 16      # "độ lớn" adapter; 16 an toàn cho 3B
LORA_ALPHA  = 32      # quy tắc phổ biến: alpha = 2*r  -> học mạnh hơn alpha=r
EPOCHS      = 3       # data nhỏ (300-500 mẫu) -> 2-3 epoch hợp lý
LR          = 2e-4    # learning rate điển hình cho LoRA
BATCH_SIZE  = 2       # số mẫu/bước; T4 chịu được
GRAD_ACCUM  = 4       # batch hiệu dụng = BATCH_SIZE * GRAD_ACCUM = 8

# --- Công tắc bật/tắt tính năng ---
SMOKE_TEST  = True    # True = chạy thử ~50 bước để kiểm pipeline cho nhanh.
                      # Pipeline OK rồi -> đặt False để train đủ EPOCHS.
USE_EVAL    = True    # True = dùng dev.jsonl để theo dõi overfit (cần DEV_PATH).
EXPORT_GGUF = True    # True = xuất file .gguf + Modelfile để nạp vào Ollama.


# ===================== CELL 3 — Nạp model + gắn LoRA =========================
import torch
from unsloth import FastLanguageModel

# from_pretrained: tải model nền + tokenizer, nén xuống 4bit (QLoRA).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,        # None = tự chọn bf16/fp16 theo GPU (T4 -> fp16)
    load_in_4bit   = True,        # QLoRA: nén trọng số xuống 4bit cho vừa VRAM
)

# get_peft_model: "khâu" các lớp LoRA (nhỏ, học được) vào model nền (đóng băng).
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = 0,        # 0 = nhanh & tối ưu nhất cho Unsloth
    bias           = "none",   # không train bias -> nhẹ hơn
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",   # 4 lớp attention
                      "gate_proj", "up_proj", "down_proj"],     # 3 lớp MLP
    use_gradient_checkpointing = "unsloth",   # đánh đổi chút tốc độ lấy tiết kiệm VRAM
    random_state   = 3407,     # cố định để chạy lại ra kết quả giống nhau
)


# ===================== CELL 4 — Nạp data + áp chat template ==================
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

# Áp đúng định dạng hội thoại ChatML của Qwen2.5 (parity với lúc serve thật).
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def to_text(batch):
    """Biến cột messages (list các {role, content}) -> chuỗi text đã format."""
    texts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,            # trả về CHUỖI, chưa cắt thành token
            add_generation_prompt=False,  # train: GIỮ luôn câu trả lời assistant
        )
        for msgs in batch["messages"]
    ]
    return {"text": texts}

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(to_text, batched=True)
print("Số mẫu train:", len(dataset))

# Tập eval (tùy chọn) — chỉ nạp nếu bật USE_EVAL và file tồn tại.
import os
eval_dataset = None
if USE_EVAL and os.path.exists(DEV_PATH):
    eval_dataset = load_dataset("json", data_files=DEV_PATH, split="train")
    eval_dataset = eval_dataset.map(to_text, batched=True)
    print("Số mẫu eval :", len(eval_dataset))
else:
    if USE_EVAL:
        print("CẢNH BÁO: không thấy DEV_PATH -> bỏ qua eval.")

print("----- Ví dụ 1 mẫu sau khi format -----")
print(dataset[0]["text"][:800])


# ===================== CELL 5 — Cấu hình train ==============================
from trl import SFTTrainer, SFTConfig

sft_args = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_ratio                = 0.05,    # 5% bước đầu tăng LR từ từ -> ổn định
    num_train_epochs            = EPOCHS,
    max_steps                   = 50 if SMOKE_TEST else -1,  # -1 = chạy đủ epoch
    learning_rate               = LR,
    logging_steps               = 1,       # in loss mỗi bước để theo dõi
    optim                       = "adamw_8bit",  # optimizer tiết kiệm VRAM
    weight_decay                = 0.01,
    lr_scheduler_type           = "linear",
    seed                        = 3407,
    dataset_text_field          = "text",  # tên cột chứa văn bản đã format
    max_seq_length              = MAX_SEQ_LEN,
    report_to                   = "none",  # không gửi log lên wandb/...
    # --- phần eval (chỉ có tác dụng khi truyền eval_dataset) ---
    per_device_eval_batch_size  = BATCH_SIZE,
    eval_strategy               = "steps" if eval_dataset is not None else "no",
    eval_steps                  = 25,
)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,        # nếu TRL mới báo lỗi -> đổi thành processing_class
    train_dataset = dataset,
    eval_dataset  = eval_dataset,
    args          = sft_args,
)

# CHỈ tính loss trên phần TRẢ LỜI của assistant (bỏ qua system + câu hỏi).
# Then chốt: model học "viết phân tích", không học thuộc lòng đề bài lặp lại.
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n", 
    # tắt chế độ tính loss (response_part chỉ bật chứ không biết tắt), cái tham số này hưu ích khi input vào trian 2 câu hội thoại 1 lúc 
    """
    system -> user -> assistant -> user -> assistant
    """
    response_part    = "<|im_start|>assistant\n", 
    # đánh bỏ qua những token ở trước nó và bắt đầu bật công tắc tính loss
)


# ===================== CELL 5.5 — Kiểm chứng mask (nên chạy 1 lần) ===========
# Mục tiêu: xác nhận phần input đã bị mask (= -100), chỉ còn câu trả lời có label.
# Nếu thấy phần system/user vẫn hiện ra -> mask SAI, dừng lại kiểm tra marker.
sample = trainer.train_dataset[0]
labels = sample["labels"]
visible = tokenizer.decode([t for t in labels if t != -100])
print("----- PHẦN MODEL THỰC SỰ HỌC (label != -100) -----")
print(visible[:800])
print("\n(Nếu chỉ thấy câu trả lời, KHÔNG thấy [Ngữ liệu]/[Đề] -> mask ĐÚNG.)")


# ===================== CELL 6 — Train =======================================
# In VRAM trước khi train để biết còn dư bao nhiêu.
gpu = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print(f"GPU: {gpu.name} | VRAM tổng: {round(gpu.total_memory/1024**3, 2)} GB | đã dùng: {start_mem} GB")

stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print(stats)
print(f"VRAM đỉnh khi train: {used} GB")
# Mẹo đọc log: cột 'loss' phải GIẢM DẦN. 'eval_loss' (nếu có) cũng nên giảm;
# nếu eval_loss tăng lên trong khi train loss giảm -> dấu hiệu OVERFIT.


# ===================== CELL 7 — Lưu adapter =================================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Đã lưu LoRA adapter tại:", OUTPUT_DIR)

# (Tùy chọn) Gộp adapter vào model nền thành bản 16bit độc lập:
# model.save_pretrained_merged("/kaggle/working/qwen2.5-3b-s0-merged",
#                              tokenizer, save_method="merged_16bit")


# ===================== CELL 8 — Test nhanh đầu ra ===========================
# Sinh thử trên 1 đề S0 MỚI (không có trong tập train) để cảm nhận chất lượng.
FastLanguageModel.for_inference(model)   # bật chế độ suy luận nhanh (2x)

SYSTEM = ("Bạn là trợ lý phân tích văn học tiếng Việt. Chỉ phân tích dựa trên "
          "ngữ liệu được cung cấp trong [Ngữ liệu]. Mọi dẫn chứng phải trích "
          "nguyên văn từ ngữ liệu, đặt trong ngoặc kép. Nếu ngữ liệu không đủ "
          "căn cứ để trả lời, hãy nói rõ điều đó thay vì suy đoán.")

test_user = ("[Ngữ liệu]\n"
             "Yêu nhau cởi áo cho nhau\n"
             "Về nhà dối mẹ qua cầu gió bay\n\n"
             "[Đề]\nNêu cảm nhận chung về câu ca dao.")

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": test_user}]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

# do_sample=False (greedy) cho KHỚP với RAG serve (temperature=0.0).
# Muốn câu trả lời "sáng tạo" hơn thì đặt do_sample=True, temperature=0.7.
out = model.generate(input_ids=inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))


# ===================== CELL 9 — Export GGUF + Modelfile cho Ollama ==========
# Chỉ chạy khi EXPORT_GGUF=True. Bước này biên dịch llama.cpp -> có thể mất
# 10-20 phút lần đầu. Cần Internet ON.
if EXPORT_GGUF:
    import glob
    GGUF_DIR = "/kaggle/working/qwen2.5-3b-s0-gguf"
    # save_pretrained_gguf: gộp adapter + xuất 1 file .gguf lượng tử hóa q4_k_m.
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")

    # Tìm tên file .gguf vừa tạo để ghi vào Modelfile.
    gguf_files = glob.glob(f"{GGUF_DIR}/*.gguf")
    gguf_name = os.path.basename(gguf_files[0]) if gguf_files else "model.gguf"
    print("File GGUF:", gguf_name)

    # Modelfile: cấu hình cho Ollama. TEMPLATE = ChatML (khớp lúc train),
    # stop = <|im_end|> để model biết dừng, temperature 0 khớp RAG service.
    modelfile = f'''FROM ./{gguf_name}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0
'''
    with open(f"{GGUF_DIR}/Modelfile", "w", encoding="utf-8") as f:
        f.write(modelfile)
    print("Đã ghi Modelfile tại:", f"{GGUF_DIR}/Modelfile")


# ===================== CELL 10 — Nén & tạo link tải về ======================
import shutil
from IPython.display import FileLink

# Nén thư mục adapter (nhẹ, vài MB) để tải nhanh.
shutil.make_archive("/kaggle/working/lora_adapter", "zip", OUTPUT_DIR)
print("Tải LoRA adapter:")
display(FileLink("/kaggle/working/lora_adapter.zip"))

# Nén thư mục GGUF (nặng ~2GB) nếu có.
if EXPORT_GGUF:
    shutil.make_archive("/kaggle/working/gguf_bundle", "zip",
                        "/kaggle/working/qwen2.5-3b-s0-gguf")
    print("Tải GGUF + Modelfile:")
    display(FileLink("/kaggle/working/gguf_bundle.zip"))

# Cách khác: vào tab "Output" của Kaggle để tải thẳng, hoặc "Save Version".

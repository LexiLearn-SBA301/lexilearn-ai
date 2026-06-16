# =============================================================================
# TRAIN S5 — DPO (Direct Preference Optimization) trên Kaggle (QLoRA + Unsloth)
# Tinh chỉnh CHẤT LƯỢNG sau SFT: model học ƯU TIÊN câu "chosen" hơn "rejected".
# Pipeline chuẩn: base Qwen -> (SFT, đã xong) -> DPO (file này).
# =============================================================================
# >>> ĐỌC TRƯỚC: DPO KHÁC SFT Ở ĐÂU? (đừng bê nguyên config SFT sang) <<<
#   - Data:   prompt/chosen/rejected  (KHÔNG phải messages như SFT).
#   - Trainer: DPOTrainer + DPOConfig (cần PatchDPOTrainer() gọi TRƯỚC).
#   - Loss:   preference loss (so log-prob chosen vs rejected), có beta.
#   - Ref:    cần reference model -> với LoRA, ref = base TẮT adapter (ref_model=None).
#   - LR:     PHẢI THẤP (~5e-6). Để 2e-4 như SFT là PHÁ model.
#   - Xuất phát: nạp ADAPTER SFT (HF) rồi train TIẾP, KHÔNG train từ base sạch.
#   - Metric: nhìn rewards/accuracies (>0.5, lý tưởng 0.7-0.9) + rewards/margins
#             TĂNG. eval_loss DPO KHÔNG so sánh trực tiếp với loss SFT.
# =============================================================================
# CÁCH CHẠY TRÊN KAGGLE (làm đúng thứ tự):
#   1. Notebook mới -> Settings: Accelerator GPU T4 x2 (hoặc P100), Internet ON.
#   2. "+ Add Input" -> Upload -> chọn dpo_train.jsonl (+ dpo_dev.jsonl để eval).
#        Kaggle đặt ở: /kaggle/input/<ten-dataset>/dpo_train.jsonl
#        -> Sửa DATA_PATH / DEV_PATH bên dưới cho khớp đường dẫn THẬT.
#   3. Copy TỪNG "CELL" vào TỪNG ô notebook, chạy lần lượt từ trên xuống.
# =============================================================================


# ===================== CELL 1 — Cài thư viện =================================
# Dùng "!" (magic của notebook) để thấy log cài đặt.
!pip install -q unsloth
# DPOTrainer nằm trong trl; unsloth thường kéo theo, nhưng cài kèm cho chắc.
!pip install -q --upgrade trl peft accelerate bitsandbytes
# Fallback nếu lệnh trên lỗi version (bỏ dấu # để dùng):
# !pip install -q --no-deps "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"


# ===================== CELL 2 — Cấu hình (chỉ chỉnh ở đây) ===================
# Đây là chỗ DUY NHẤT bạn cần sửa cho mỗi lần chạy.

# Điểm XUẤT PHÁT của DPO = adapter SFT đã train (KHÔNG phải base sạch).
# Trỏ vào repo adapter SFT trên HuggingFace (cùng repo bạn dùng ở phần eval SFT).
SFT_ADAPTER_REPO = "Tobi2904/Test-Adapter"   # <-- SỬA nếu repo adapter SFT của bạn khác

# --- Đường dẫn data DPO (SỬA cho khớp nơi bạn upload) ---
DATA_PATH   = "/kaggle/input/dpo-s5/dpo_train.jsonl"   # tập huấn luyện (bắt buộc)
DEV_PATH    = "/kaggle/input/dpo-s5/dpo_dev.jsonl"     # tập kiểm định (nếu USE_EVAL=True)
OUTPUT_DIR  = "/kaggle/working/qwen2.5-3b-dpo-ckpt"    # trainer ghi CHECKPOINT (để resume nếu crash)
ADAPTER_DIR = "/kaggle/working/qwen2.5-3b-dpo-lora"    # adapter DPO CUỐI (sạch) -> tải về / push HF

# --- System prompt CỐ ĐỊNH (PHẢI giống y nguyên SFT s0-s4 để parity) ---
# s5.jsonl không kèm system; ta GẮN system này vào prompt để policy được tinh
# chỉnh trên CÙNG phân phối input như lúc SFT và lúc serve qua Ollama.
SYSTEM_FIXED = ("Bạn là trợ lý phân tích văn học tiếng Việt. Chỉ phân tích dựa trên "
                "ngữ liệu được cung cấp trong [Ngữ liệu]. Mọi dẫn chứng phải trích "
                "nguyên văn từ ngữ liệu, đặt trong ngoặc kép. Nếu ngữ liệu không đủ "
                "căn cứ để trả lời, hãy nói rõ điều đó thay vì suy đoán.")

# --- Siêu tham số DPO ---
# (KHÔNG có LORA_R/ALPHA ở đây: ta train TIẾP chính adapter SFT đã nạp, nên hình
#  học LoRA được KẾ THỪA từ SFT — không tạo adapter mới.)
# Lưu ý: bản TRL mới BỎ max_prompt_length, chỉ còn max_length (prompt+completion).
MAX_LEN        = 2048   # độ dài tối đa prompt + completion (= MAX_SEQ_LEN lúc SFT)
EPOCHS         = 1      # DPO chỉ cần 1-2 epoch; nhiều dễ over-optimize -> lệch base
DPO_LR         = 5e-6   # THẤP hơn SFT ~40 lần. Đừng để 2e-4!
BETA           = 0.1    # độ "ghì" về reference: cao=bám ref, thấp=học mạnh preference
BATCH_SIZE     = 1      # mỗi mẫu DPO = 2 lượt forward (chosen+rejected) -> để nhỏ
GRAD_ACCUM     = 8      # batch hiệu dụng = BATCH_SIZE * GRAD_ACCUM = 8

# --- Công tắc bật/tắt tính năng ---
SMOKE_TEST  = True    # True = chạy thử ~30 bước để kiểm pipeline. OK rồi -> False, CHẠY LẠI TỪ CELL 3.
USE_EVAL    = True    # True = dùng dpo_dev.jsonl theo dõi rewards/accuracies (cần DEV_PATH).
EXPORT_GGUF = True    # True = xuất .gguf + Modelfile để nạp Ollama (gộp adapter DPO + base).


# ===================== CELL 3 — Patch DPO + nạp ADAPTER SFT ==================
# PatchDPOTrainer() PHẢI gọi TRƯỚC khi tạo DPOTrainer (Unsloth tối ưu DPO).
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()

import torch

# KHÁC SFT: model_name trỏ vào ADAPTER SFT (không phải base) -> Unsloth tải base
# Qwen + GẮN SẴN adapter SFT, ở chế độ train. Ta train TIẾP chính adapter này.
#
# !!! KHÔNG gọi get_peft_model() ở đây !!!
#   - get_peft_model dùng khi nạp BASE SẠCH để TẠO MỚI adapter.
#   - Ở đây adapter SFT ĐÃ có sẵn; gọi lại sẽ chồng/khởi tạo NGẪU NHIÊN adapter mới
#     -> mất toàn bộ phần đã học ở SFT (sai mục tiêu base->SFT->DPO).
#   - Hình học LoRA (r, alpha, target_modules) được KẾ THỪA từ adapter SFT.
#
# Reference model: KHÔNG nạp model thứ 2 — với PEFT, DPOTrainer tính reference
# bằng cách TẮT adapter trên chính model này (tiết kiệm VRAM). Để ref_model=None.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = SFT_ADAPTER_REPO,
    max_seq_length = MAX_LEN,
    dtype          = None,        # None = tự chọn bf16/fp16 theo GPU (T4 -> fp16)
    load_in_4bit   = True,        # QLoRA: nén base xuống 4bit cho vừa VRAM
)

# Đảm bảo các lớp LoRA đang ở chế độ train (phòng khi adapter nạp ở mode inference).
# (Bản Unsloth cũ không có for_training -> bỏ qua an toàn, mặc định đã train được.)
if hasattr(FastLanguageModel, "for_training"):
    FastLanguageModel.for_training(model)


# ===================== CELL 4 — Nạp data + format ChatML =====================
# DPOTrainer cần 3 cột: prompt / chosen / rejected (đều là CHUỖI đã format).
# Ta GẮN system + render đúng ChatML qwen-2.5 để parity với SFT & serve:
#   prompt   = <system><user>...<assistant\n   (kết thúc ở chỗ assistant sắp nói)
#   chosen   = <nội dung tốt> + <|im_end|>
#   rejected = <nội dung kém> + <|im_end|>
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import os

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def to_dpo_format(batch):
    prompts, chosens, rejecteds = [], [], []
    for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
        msgs = [
            {"role": "system", "content": SYSTEM_FIXED},
            {"role": "user",   "content": prompt},
        ]
        # add_generation_prompt=True -> chuỗi kết thúc bằng "<|im_start|>assistant\n"
        # (phần model SẮP sinh) -> chosen/rejected nối tiếp ngay sau đó.
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
        # completion: nội dung + token kết thúc lượt assistant.
        chosens.append(chosen + tokenizer.eos_token)
        rejecteds.append(rejected + tokenizer.eos_token)
    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(to_dpo_format, batched=True,
                      remove_columns=[c for c in dataset.column_names
                                      if c not in ("prompt", "chosen", "rejected")])
print("Số cặp train:", len(dataset))

eval_dataset = None
if USE_EVAL and os.path.exists(DEV_PATH):
    eval_dataset = load_dataset("json", data_files=DEV_PATH, split="train")
    eval_dataset = eval_dataset.map(to_dpo_format, batched=True,
                                    remove_columns=[c for c in eval_dataset.column_names
                                                    if c not in ("prompt", "chosen", "rejected")])
    print("Số cặp eval :", len(eval_dataset))
else:
    if USE_EVAL:
        print("CẢNH BÁO: không thấy DEV_PATH -> bỏ qua eval.")

print("----- Ví dụ 1 cặp sau khi format -----")
print("[PROMPT]\n",   dataset[0]["prompt"][:600])
print("\n[CHOSEN]\n",  dataset[0]["chosen"][:300])
print("\n[REJECTED]\n",dataset[0]["rejected"][:300])


# ===================== CELL 5 — Cấu hình train DPO ==========================
# KHÁC SFT: DPOConfig (không phải SFTConfig); KHÔNG dùng train_on_responses_only
# (DPOTrainer tự mask prompt, chỉ tính loss trên 2 completion chosen/rejected).
from trl import DPOTrainer, DPOConfig

dpo_args = DPOConfig(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_ratio                = 0.1,
    num_train_epochs            = EPOCHS,
    max_steps                   = 30 if SMOKE_TEST else -1,   # -1 = chạy đủ epoch
    learning_rate               = DPO_LR,
    beta                        = BETA,         # tham số ĐẶC TRƯNG của DPO
    ld_alpha                    = 0.5,          # length-desensitization: phạt chọn câu DÀI hơn, tránh học vẹt
    max_length                  = MAX_LEN,      # prompt+completion; TRL mới bỏ max_prompt_length
    truncation_mode             = "keep_end",   # nếu quá dài: cắt ĐẦU prompt, GIỮ [Đề] ở cuối
    logging_steps               = 1,
    optim                       = "adamw_8bit",
    weight_decay                = 0.0,          # DPO thường để 0 (tránh kéo lệch quá)
    lr_scheduler_type           = "linear",
    seed                        = 3407,
    report_to                   = "none",
    # --- eval (chỉ tác dụng khi truyền eval_dataset) ---
    per_device_eval_batch_size  = BATCH_SIZE,
    eval_strategy               = "steps" if eval_dataset is not None else "no",
    eval_steps                  = 20,
    # --- checkpoint để resume khi crash; SMOKE_TEST thì tắt cho đỡ rác ---
    save_strategy               = "no" if SMOKE_TEST else "steps",
    save_steps                  = 20,
    save_total_limit            = 3,
    # cuối train: nạp lại checkpoint TỐT NHẤT theo rewards/accuracies (CÀNG CAO càng tốt).
    load_best_model_at_end      = (eval_dataset is not None) and (not SMOKE_TEST),
    metric_for_best_model       = "eval_rewards/accuracies",
    greater_is_better           = True,
)

trainer = DPOTrainer(
    model           = model,
    ref_model       = None,            # PEFT: ref = base TẮT adapter (không nạp model 2)
    args            = dpo_args,
    train_dataset   = dataset,
    eval_dataset    = eval_dataset,
    processing_class = tokenizer,      # TRL mới; bản cũ đổi thành tokenizer=tokenizer
)


# ===================== CELL 6 — Train =======================================
gpu = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print(f"GPU: {gpu.name} | VRAM tổng: {round(gpu.total_memory/1024**3, 2)} GB | đã dùng: {start_mem} GB")

# Tự dò checkpoint cũ trong OUTPUT_DIR: có -> train TIẾP, không -> train từ đầu.
# (Lưu ý: /kaggle/working bị xóa khi TẮT session. Muốn resume sang session KHÁC,
#  Save Version rồi add output làm Input và trỏ OUTPUT_DIR về thư mục đó.)
from transformers.trainer_utils import get_last_checkpoint
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
if last_ckpt:
    print("Tìm thấy checkpoint -> train TIẾP từ:", last_ckpt)
    stats = trainer.train(resume_from_checkpoint=last_ckpt)
else:
    print("Không có checkpoint -> train từ đầu.")
    stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print(stats)
print(f"VRAM đỉnh khi train: {used} GB")
# Mẹo đọc log DPO (KHÁC SFT):
#   - rewards/accuracies : % lần model chấm chosen > rejected. PHẢI > 0.5, lý tưởng 0.7-0.9.
#   - rewards/margins    : (reward chosen - reward rejected). Nên TĂNG dần.
#   - loss               : giảm, NHƯNG không so trực tiếp với loss SFT.
#   - Nếu rewards/rejected tụt quá sâu mà chosen cũng tụt -> beta thấp/LR cao, model lệch.


# ===================== CELL 7 — Lưu adapter DPO =============================
# load_best_model_at_end=True nên 'model' là bản rewards/accuracies CAO NHẤT.
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Đã lưu LoRA adapter DPO (bản tốt nhất) tại:", ADAPTER_DIR)

# (Tùy chọn) push thẳng lên HF để session eval pull về:
# model.push_to_hub("Tobi2904/Test-Adapter-DPO", token="hf_...")
# tokenizer.push_to_hub("Tobi2904/Test-Adapter-DPO", token="hf_...")


# ===================== CELL 8 — Test nhanh đầu ra ===========================
# Sinh thử trên 1 đề MỚI để cảm nhận chất lượng sau DPO.
FastLanguageModel.for_inference(model)

test_user = ("[Ngữ liệu]\n"
             "Thân em như giếng giữa đàng\n"
             "Người khôn rửa mặt, người phàm rửa chân\n\n"
             "[Đề]\nPhân tích thân phận người phụ nữ qua câu ca dao.")

messages = [{"role": "system", "content": SYSTEM_FIXED},
            {"role": "user",   "content": test_user}]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

# do_sample=False (greedy) cho KHỚP với RAG serve (temperature=0.0).
out = model.generate(input_ids=inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))


# ===================== CELL 9 — Export GGUF + Modelfile cho Ollama ==========
# Gộp base + adapter DPO -> 1 file .gguf q4_k_m. Cần Internet ON; lần đầu 10-20'.
if EXPORT_GGUF:
    import glob
    GGUF_DIR = "/kaggle/working/qwen2.5-3b-dpo-gguf"
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")

    # Unsloth có thể đặt file .gguf ở thư mục KHÁC (vd ...-gguf_gguf) -> DÒ ĐỆ QUY
    # toàn bộ /kaggle/working và chọn file .gguf LỚN NHẤT (= file model thật ~2GB).
    all_gguf = glob.glob("/kaggle/working/**/*.gguf", recursive=True)
    if not all_gguf:
        raise FileNotFoundError("Không thấy file .gguf nào — kiểm tra lại bước save_pretrained_gguf.")
    GGUF_PATH = max(all_gguf, key=os.path.getsize)     # full path tới file model
    gguf_name = os.path.basename(GGUF_PATH)            # tên file (vd unsloth.Q4_K_M.gguf)
    GGUF_REAL_DIR = os.path.dirname(GGUF_PATH)         # thư mục THẬT chứa file .gguf
    print("File GGUF:", GGUF_PATH, "|", round(os.path.getsize(GGUF_PATH)/1e9, 2), "GB")

    # Modelfile: ChatML (khớp lúc train), stop <|im_end|>, temperature 0 khớp RAG.
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
    # Ghi Modelfile CẠNH file .gguf thật (để FROM ./{gguf_name} trỏ đúng).
    MODELFILE_PATH = os.path.join(GGUF_REAL_DIR, "Modelfile")
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile)
    print("Đã ghi Modelfile tại:", MODELFILE_PATH)


# ===================== CELL 9.5 — Push GGUF lên HuggingFace =================
# Mục đích: đẩy file .gguf vừa tạo (CELL 9) lên 1 repo HF -> serve chỉ cần
#   ollama pull hf.co/<GGUF_HF_REPO>
# GIỐNG hệt cách bản SFT đang chạy (hf.co/Tobi2904/qwen-finetuned-gguf trong
# docker-compose.yml). Ollama đọc thẳng .gguf trên HF, KHÔNG cần Modelfile riêng
# (template ChatML + stop token đã nhúng sẵn trong .gguf).
#
# YÊU CẦU: repo phải PUBLIC để Ollama pull khỏi cần auth; token HF phải có quyền WRITE.
if EXPORT_GGUF:
    from huggingface_hub import HfApi, create_repo

    # <-- SỬA: <user>/<ten-repo>.
    #   - Dùng repo MỚI (vd ...-dpo-gguf) để giữ riêng bản DPO, KHÔNG đè bản SFT.
    #   - HOẶC push thẳng vào repo SFT cũ ("Tobi2904/qwen-finetuned-gguf") để
    #     serve "hot-swap" sang bản DPO mà KHỎI đổi .env/docker-compose.
    GGUF_HF_REPO = "Tobi2904/qwen-finetuned-dpo-gguf"

    # Lấy token: ưu tiên Kaggle Secrets (Add-ons -> Secrets, tạo secret tên HF_TOKEN).
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = "hf_xxx"   # <-- hoặc dán thẳng token WRITE (đừng commit lên git!)

    # Tạo repo public nếu chưa có; đã có thì bỏ qua (exist_ok=True).
    create_repo(GGUF_HF_REPO, repo_type="model", private=False,
                exist_ok=True, token=HF_TOKEN)

    api = HfApi()
    # Đẩy file .gguf (bắt buộc, đủ cho Ollama). Dùng GGUF_PATH/MODELFILE_PATH THẬT
    # đã dò ở CELL 9 (không ghép tay đường dẫn -> tránh sai folder ...-gguf_gguf).
    print(f"Đang upload {gguf_name} (~2GB) lên {GGUF_HF_REPO} ...")
    api.upload_file(path_or_fileobj=GGUF_PATH,
                    path_in_repo=gguf_name,
                    repo_id=GGUF_HF_REPO, repo_type="model", token=HF_TOKEN)
    api.upload_file(path_or_fileobj=MODELFILE_PATH,
                    path_in_repo="Modelfile",
                    repo_id=GGUF_HF_REPO, repo_type="model", token=HF_TOKEN)

    print("Đã push GGUF lên:", f"https://huggingface.co/{GGUF_HF_REPO}")
    print("\n--- SERVE: chạy trên máy chạy Ollama ---")
    print(f"   ollama pull hf.co/{GGUF_HF_REPO}")
    print("Rồi đặt trong .env / docker-compose.yml:")
    print(f"   FINE_TUNED_OLLAMA_LLM_MODEL=hf.co/{GGUF_HF_REPO}:latest")


# ===================== CELL 10 — Nén & tạo link tải về ======================
import shutil
from IPython.display import FileLink

shutil.make_archive("/kaggle/working/dpo_lora_adapter", "zip", ADAPTER_DIR)
print("Tải LoRA adapter DPO:")
display(FileLink("/kaggle/working/dpo_lora_adapter.zip"))

if EXPORT_GGUF:
    # Nén thư mục THẬT chứa .gguf (GGUF_REAL_DIR đã dò ở CELL 9), không đoán tên.
    shutil.make_archive("/kaggle/working/dpo_gguf_bundle", "zip", GGUF_REAL_DIR)
    print("Tải GGUF + Modelfile:")
    display(FileLink("/kaggle/working/dpo_gguf_bundle.zip"))


# #############################################################################
# ###  PHẦN 2 — EVAL DPO (chạy ĐỘC LẬP, sau khi đã push adapter DPO lên HF) ###
# #############################################################################
#
# >>> SESSION MỚI CHỈ ĐỂ CHẤM (KHÔNG train lại) <<< chạy: CELL 1 -> 11 -> 12 -> 13.
#   TRƯỚC KHI CHẠY: + Add Input -> Upload -> dpo_test.jsonl, sửa TEST_PATH (CELL 11).
#
# Ý nghĩa eval DPO: với mỗi cặp test, tính log-prob model gán cho chosen vs rejected.
# "Đúng" = model cho chosen điểm CAO HƠN rejected. Đây là preference accuracy trên
# tập test HELD-OUT (không dính lúc train) -> bằng chứng DPO thực sự học được "gu".
# #############################################################################


# ===================== CELL 11 — Cấu hình EVAL DPO =========================
DPO_ADAPTER_REPO = "Tobi2904/Test-Adapter-DPO"        # repo adapter DPO trên HF (sau khi push)
EVAL_SEQ_LEN     = 2048                               # = MAX_LEN lúc train
TEST_PATH        = "/kaggle/input/dpo-s5/dpo_test.jsonl"   # SỬA cho khớp nơi upload
RESULT_PATH      = "/kaggle/working/dpo_test_results.jsonl"
EVAL_LIMIT       = None    # None = chạy HẾT. Đặt số (vd 5) để thử nhanh trước.

SYSTEM_FIXED = ("Bạn là trợ lý phân tích văn học tiếng Việt. Chỉ phân tích dựa trên "
                "ngữ liệu được cung cấp trong [Ngữ liệu]. Mọi dẫn chứng phải trích "
                "nguyên văn từ ngữ liệu, đặt trong ngoặc kép. Nếu ngữ liệu không đủ "
                "căn cứ để trả lời, hãy nói rõ điều đó thay vì suy đoán.")


# ===================== CELL 12 — Pull adapter DPO từ HF ====================
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = DPO_ADAPTER_REPO,
    max_seq_length = EVAL_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
FastLanguageModel.for_inference(model)
print("Đã nạp adapter DPO:", DPO_ADAPTER_REPO)


# ===================== CELL 13 — Chấm preference accuracy trên test =========
# Với mỗi cặp: tính TỔNG log-prob model gán cho phần completion (chosen / rejected).
# Completion nào log-prob cao hơn = model "thích" hơn. Đúng khi chosen > rejected.
import json, time
import torch.nn.functional as F

def seq_logprob(prompt_text, completion_text):
    """Tổng log-prob của các token completion, ĐK trên prompt (teacher forcing)."""
    p_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    full  = tokenizer(prompt_text + completion_text, add_special_tokens=False,
                      return_tensors="pt").input_ids.to("cuda")
    plen  = p_ids.shape[1]
    with torch.no_grad():
        logits = model(full).logits          # [1, T, V]
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt  = full[:, 1:]                        # token kế tiếp
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]   # [T-1]
    # chỉ cộng phần completion (token từ vị trí plen trở đi).
    return tok_lp[plen - 1:].sum().item()

rows = []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if {"prompt", "chosen", "rejected"} <= r.keys():
                rows.append(r)
if EVAL_LIMIT:
    rows = rows[:EVAL_LIMIT]
print("Tổng cặp test:", len(rows))

results, n_correct = [], 0
t0 = time.time()
for i, r in enumerate(rows, 1):
    msgs = [{"role": "system", "content": SYSTEM_FIXED},
            {"role": "user",   "content": r["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
    lp_chosen   = seq_logprob(prompt_text, r["chosen"]   + tokenizer.eos_token)
    lp_rejected = seq_logprob(prompt_text, r["rejected"] + tokenizer.eos_token)
    correct = lp_chosen > lp_rejected
    n_correct += int(correct)
    results.append({"id": r.get("id"), "lp_chosen": round(lp_chosen, 2),
                    "lp_rejected": round(lp_rejected, 2),
                    "margin": round(lp_chosen - lp_rejected, 2), "correct": correct})
    if i % 10 == 0 or i == len(rows):
        print(f"  {i}/{len(rows)} | {round(time.time()-t0)}s")

acc = round(100 * n_correct / max(len(rows), 1), 1)
print(f"\n=== PREFERENCE ACCURACY (test held-out): {n_correct}/{len(rows)} ({acc}%) ===")
print("(>50% là tốt hơn ngẫu nhiên; mong muốn 70-90%. So thêm với adapter SFT để thấy DPO cải thiện.)")

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from IPython.display import FileLink
print("Đã ghi:", RESULT_PATH)
display(FileLink(RESULT_PATH))

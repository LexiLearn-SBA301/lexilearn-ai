"""
State schema cho luồng: Supervisor -> route -> (Mode A | Deep pipeline) -> Output

Quy ước thiết kế:
- GraphState (TypedDict): state tổng chạy qua các node của graph. Được checkpoint/persist.
- Các cấu trúc con dùng Pydantic v2: vừa validate, vừa parse được output template
  của critic / supervisor (LLM trả structured).
- Phân biệt rõ 2 loại "thinking realtime":
    (1) MILESTONE events  -> nằm trong state.events (ít, có cấu trúc, để replay/resume).
    (2) TOKEN stream       -> KHÔNG vào state. Emit qua dispatcher ra ngoài (SSE/WS).
- Mọi trường mà nhiều node có thể ghi song song (vd 4 critic chạy parallel) đều
  khai báo reducer để merge an toàn.
"""

from __future__ import annotations

import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# =============================================================================
# 0. REDUCERS  (cách LangGraph merge khi nhiều node ghi cùng 1 key)
# =============================================================================

def merge_dict(left: Optional[dict], right: Optional[dict]) -> dict:
    """ left là dữ liệu cũ, right mới
        (left or {}) chống crash
        {A, B} lấy A và B gợp lại nếu trùng key B sẽ đè lên A
        **A unpacket A và B để tiến hành merge """
    return {**(left or {}), **(right or {})}

def take_last(left: Any, right: Any) -> Any:
    """Ghi đè: lấy giá trị mới nhất (dùng cho các field scalar do 1 node set)."""
    return right if right is not None else left


# list events: append-only -> operator.add


# =============================================================================
# 1. ENUM / hằng số
# =============================================================================

class Route(str, Enum):
    FACTUAL = "factual"            # Mode A: RAG trả lời ngắn có dẫn chứng
    DEEP = "deep_analysis"         # Deep pipeline: context -> debate -> essay


class Stage(str, Enum):
    INTENT = "intent"
    FACTUAL = "factual"
    PREPARE_CONTEXT = "prepare_context"     # Tool 1
    CRITICS_DEBATE = "critics_debate"       # Tool 2
    WRITE_ESSAY = "write_essay"             # Tool 3
    DONE = "done"


class CriticRole(str, Enum):
    HINH_THUC = "hinh_thuc"        # Hình thức: phong cách, ngôn từ
    LICH_SU = "lich_su"            # Lịch sử: bối cảnh xã hội
    TAM_LY = "tam_ly"              # Tâm lý: nội tâm nhân vật
    TIEP_NHAN = "tiep_nhan"        # Tiếp nhận: góc đương đại


CRITIC_DISPLAY = {
    CriticRole.HINH_THUC: "Nhà phê bình Hình thức",
    CriticRole.LICH_SU: "Nhà phê bình Lịch sử",
    CriticRole.TAM_LY: "Nhà phê bình Tâm lý",
    CriticRole.TIEP_NHAN: "Nhà phê bình Tiếp nhận",
}


class Verdict(str, Enum):
    PASS = "pass"          # judge: đủ tiêu chuẩn -> đi tiếp
    RETRY = "retry"        # judge: chưa đạt -> lặp lại tool (kèm feedback)
    APPROVE = "approve"    # supervisor review cuối: duyệt
    REJECT = "reject"      # quá số lần retry / fail cứng


# =============================================================================
# 2. CÁC KHỐI DỮ LIỆU CHUNG
# =============================================================================

class SourceChunk(BaseModel):
    """1 đoạn văn bản gốc lấy từ Vector DB.

    Khớp trực tiếp với 1 phần tử trong `sources` của RAGService.query():
    model_validate(<dict RAG>) nuốt thẳng dict, field thừa (source_doc_id,
    embedding, search_text...) Pydantic tự bỏ qua. populate_by_name cho phép
    vẫn truyền theo tên field "sạch" (text=, score=) trong code.
    """
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str
    text: str = Field(validation_alias=AliasChoices("text", "content"))      # RAG: content
    source_ref: Optional[str] = None         # RAG không trả -> Optional. vd "Truyện Kiều, câu 1-6"
    score: Optional[float] = Field(           # RAG: rrf_score; độ liên quan retrieval
        default=None, validation_alias=AliasChoices("score", "rrf_score"))
    position: Optional[dict[str, Any]] = None  # RAG: {page, chunk_index, total_chunks}
    metadata: dict[str, Any] = Field(default_factory=dict)  # define rõ ràng hơn
    # BaseModel tự động ép kiểu và hàm init mặc định


class Citation(BaseModel):
    """1 trích dẫn trong essay, để Citation Checker (Python rule-based) verify."""
    citation_id: str
    quoted_text: str                         # đoạn được trích / diễn giải
    source_ref: str                          # nguồn người đọc thấy
    chunk_id: Optional[str] = None           # link về SourceChunk thật
    verified: bool = False                    # checker set
    verification_note: Optional[str] = None  # vì sao fail (nếu fail)


# =============================================================================
# 3. INTENT (Supervisor – Gemini 2.5)
# =============================================================================

class IntentAnalysis(BaseModel):
    raw_query: str
    route: Route
    confidence: float = 0.0
    work_title: Optional[str] = None         # tác phẩm
    author: Optional[str] = None
    detected_entities: list[str] = Field(default_factory=list)
    requested_dimensions: list[CriticRole] = Field(default_factory=list)  # user chỉ định góc nhìn (nếu có)
    reasoning: str = ""                       # supervisor "nghĩ gì" -> nên stream ra
    analyzed_at: Optional[datetime] = None


# =============================================================================
# 4. MODE A – FACTUAL
# =============================================================================

class FactualResult(BaseModel):
    answer: str
    chunks_used: list[SourceChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    model: str = "qwen-3b"


# =============================================================================
# 5. TOOL 1 – PREPARE_CONTEXT
# =============================================================================

class Entity(BaseModel):
    name: str
    type: Literal["character", "place", "theme", "motif", "event", "other"]
    description: str = ""
    mentions: list[str] = Field(default_factory=list)   # chunk_id liên quan


class PreparedContext(BaseModel):
    retrieval_query: str = ""
    chunks: list[SourceChunk] = Field(default_factory=list)   # văn bản gốc
    summary: str = ""
    entities: list[Entity] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    key_passages: list[SourceChunk] = Field(default_factory=list)
    token_count: int = 0
    retrieved_at: Optional[datetime] = None


# =============================================================================
# 6. TOOL 2 – CRITICS_DEBATE  (4 critic x 2 round)
# =============================================================================

class Argument(BaseModel):
    arg_id: str
    point: str                                # luận điểm
    support: str                              # diễn giải/lập luận
    citation_ids: list[str] = Field(default_factory=list)


class Rebuttal(BaseModel):
    """R2: critic này phản biện 1 luận điểm của critic khác."""
    target_critic: CriticRole
    target_arg_id: Optional[str] = None
    stance: Literal["agree", "disagree", "qualify"]   # đồng tình / phản đối / bổ sung-điều kiện
    reason: str
    citation_ids: list[str] = Field(default_factory=list)


class CriticTurn(BaseModel):
    """1 lượt nói của 1 critic trong 1 round."""
    critic: CriticRole
    round: Literal[1, 2]
    # R1: tự retrieve trước khi nói. R2: đọc bulletin (không bắt buộc retrieve lại).
    retrieval_query: Optional[str] = None
    retrieved_chunks: list[SourceChunk] = Field(default_factory=list)
    bulletin_seen: bool = False               # R2 đã đọc bulletin?
    thesis: str = ""
    arguments: list[Argument] = Field(default_factory=list)
    rebuttals: list[Rebuttal] = Field(default_factory=list)   # chỉ R2
    citations: list[Citation] = Field(default_factory=list)
    raw_output: str = ""                       # output thô từ Qwen-3B (debug/replay)
    parsed_ok: bool = True                     # parse từ template thành công?
    spoke_at: Optional[datetime] = None


class BulletinEntry(BaseModel):
    """1 dòng bulletin chung – tóm tắt R1 của 1 critic, để R2 đọc."""
    critic: CriticRole
    thesis: str
    key_points: list[str] = Field(default_factory=list)
    arg_ids: list[str] = Field(default_factory=list)


class DebateState(BaseModel):
    # 4 critic R1 chạy song song -> dict keyed theo CriticRole
    round1: dict[CriticRole, CriticTurn] = Field(default_factory=dict)
    bulletin: list[BulletinEntry] = Field(default_factory=list)   # parse từ 4 R1
    round2: dict[CriticRole, CriticTurn] = Field(default_factory=dict)
    # tổng hợp sau debate (judge / supervisor dùng để đánh giá substantive)
    consensus_points: list[str] = Field(default_factory=list)
    contested_points: list[str] = Field(default_factory=list)
    total_invocations: int = 0                 # kỳ vọng 8 (4 R1 + 4 R2)


# =============================================================================
# 7. TOOL 3 – WRITE_ESSAY  (Synthesizer + Citation Checker)
# =============================================================================

class EssaySection(BaseModel):
    heading: str
    body: str
    citation_ids: list[str] = Field(default_factory=list)


class CitationCheckResult(BaseModel):
    total: int = 0
    verified: int = 0
    failed: int = 0
    issues: list[str] = Field(default_factory=list)   # citation nào không khớp nguồn
    passed: bool = False


class EssayDraft(BaseModel):
    title: str = ""
    sections: list[EssaySection] = Field(default_factory=list)
    full_text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    citation_check: Optional[CitationCheckResult] = None
    word_count: int = 0
    draft_version: int = 1


# =============================================================================
# 8. JUDGE / REVIEW (Supervisor)
# =============================================================================

class JudgeVerdict(BaseModel):
    stage: Stage
    verdict: Verdict
    scores: dict[str, float] = Field(default_factory=dict)  # vd {"depth":.., "logic":.., "style":..}
    reasoning: str = ""                       # supervisor "nghĩ gì" -> stream ra
    feedback: str = ""                        # QUAN TRỌNG: gửi lại tool khi RETRY để có hướng sửa
    judged_at: Optional[datetime] = None


# =============================================================================
# 9. OUTPUT
# =============================================================================

class FinalOutput(BaseModel):
    answer: str                                # essay hoàn chỉnh hoặc câu trả lời factual
    route: Route
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceChunk] = Field(default_factory=list)
    finished_at: Optional[datetime] = None


# =============================================================================
# 10. STREAMING – realtime thinking ra ngoài
# =============================================================================

class EventType(str, Enum):
    STATUS = "status"                  # đổi node / trạng thái
    INTENT = "intent"                  # supervisor phân tích intent
    ROUTE = "route"                    # quyết định route
    RETRIEVAL = "retrieval"            # đang/đã retrieve vector db
    THINKING = "thinking"              # reasoning trung gian (hiển thị cho user)
    TOKEN = "token"                    # 1 token/chunk văn bản (essay/critic) – chỉ stream
    CRITIC_TURN = "critic_turn"        # 1 critic vừa nói xong (milestone)
    BULLETIN = "bulletin"              # bulletin chung sẵn sàng
    JUDGE = "judge"                    # phán quyết của supervisor
    RETRY = "retry"                    # kích hoạt retry
    CITATION_CHECK = "citation_check"  # kết quả check trích dẫn
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    """
    1 sự kiện đẩy ra UI. Lưu ý:
    - type=TOKEN, is_partial=True  => CHỈ stream ra ngoài, KHÔNG nên append vào state.events.
    - các type khác (milestone)    => append vào state.events (replay/resume được).
    """
    seq: int                                   # thứ tự tăng dần, để FE sắp xếp
    type: EventType
    node: str                                  # vd "critic:hinh_thuc:r1", "supervisor:judge_context"
    actor: str = ""                            # tên hiển thị: "Giám khảo", "Nhà phê bình Tâm lý"
    title: str = ""                            # nhãn ngắn cho UI
    content: str = ""                          # nội dung thinking / token
    payload: dict[str, Any] = Field(default_factory=dict)  # data có cấu trúc kèm theo, màu sắc trạng thái cho FE đọc
    is_partial: bool = False                   # Biến streaming dữ liệu True là bật, FE nhìn vào để biết nhỏ con trỏ ra để chat tiếp
    parent_seq: Optional[int] = None           # để FE lồng UI (debate nằm trong Tool 2), nghĩa là chỉ mục cha để lồng như comment trên FB
    ts: Optional[datetime] = None


# =============================================================================
# 11. GRAPH STATE  (top-level, chạy qua mọi node)
# =============================================================================

class GraphState(TypedDict, total=False):
    # --- định danh & vòng đời ---
    run_id: str
    thread_id: str                                   # để checkpoint/resume
    status: Annotated[Literal["running", "awaiting_retry", "done", "failed"], take_last]
    current_stage: Annotated[Stage, take_last]
    current_node: Annotated[str, take_last]

    # --- input ---
    user_query: str
    conversation_history: list[dict[str, str]]       # nếu multi-turn

    # --- routing / intent ---
    intent: Annotated[Optional[IntentAnalysis], take_last]
    route: Annotated[Optional[Route], take_last]

    # --- Mode A ---
    factual: Annotated[Optional[FactualResult], take_last]

    # --- Deep pipeline ---
    context: Annotated[Optional[PreparedContext], take_last]   # Tool 1
    debate: Annotated[Optional[DebateState], take_last]        # Tool 2
    essay: Annotated[Optional[EssayDraft], take_last]          # Tool 3

    # --- judge / review (key theo stage) ---
    judges: Annotated[dict[str, JudgeVerdict], merge_dict]
    # vd: {"prepare_context": JudgeVerdict, "critics_debate": ..., "write_essay": ...}

    # --- điều khiển retry ---
    retry_counts: Annotated[dict[str, int], merge_dict]   # {"prepare_context":1, ...}
    retry_limits: dict[str, int]                          # cấu hình: vd {"write_essay":1}
    last_feedback: Annotated[dict[str, str], merge_dict]  # feedback judge -> tool đọc khi retry

    # --- output ---
    final_output: Annotated[Optional[FinalOutput], take_last]

    # --- realtime thinking (chỉ MILESTONE; token đi kênh riêng) ---
    events: Annotated[list[StreamEvent], operator.add]
    event_seq: Annotated[int, take_last]                  # bộ đếm seq


# Cấu hình mặc định gợi ý
DEFAULT_RETRY_LIMITS: dict[str, int] = {
    "prepare_context": 2,    # diagram: "No retry" (không giới hạn cứng) -> đặt trần an toàn
    "critics_debate": 2,
    "write_essay": 1,        # diagram: "No max 1"
}


# =============================================================================
# 12. HELPER – emit event (minh họa cách stream realtime)
# =============================================================================

class EventEmitter:
    """
    Bọc việc phát event. Trong LangGraph: dùng get_stream_writer() để đẩy
    custom event ra ngoài ngay lập tức (realtime), đồng thời trả về list
    milestone-event để node merge vào state.events.

    Cách dùng trong 1 node:
        emitter = EventEmitter(state, writer=get_stream_writer())
        emitter.thinking("supervisor", "Đang phân tích intent...")
        emitter.token("critic:tam_ly:r1", "Nhân vật ")   # stream, không vào state
        ...
        return {"events": emitter.milestones, "event_seq": emitter.seq, ...}
    """

    def __init__(self, state: GraphState, writer=None):
        self.seq = state.get("event_seq", 0)
        self.writer = writer            # hàm đẩy ra ngoài (SSE/WS). None = không stream.
        self.milestones: list[StreamEvent] = []

    def _next(self) -> int:
        self.seq += 1
        return self.seq

    def emit(self, ev: StreamEvent, *, persist: bool) -> None:
        # luôn đẩy ra ngoài realtime nếu có writer
        if self.writer is not None:
            self.writer(ev.model_dump(mode="json"))
        # chỉ milestone mới vào state để được checkpoint
        if persist:
            self.milestones.append(ev)

    def thinking(self, node: str, content: str, actor: str = "") -> None:
        self.emit(
            StreamEvent(seq=self._next(), type=EventType.THINKING, node=node,
                        actor=actor, content=content, ts=datetime.utcnow()),
            persist=True,
        )

    def token(self, node: str, text: str) -> None:
        # token-level: stream dở, KHÔNG persist (tránh phình state)
        self.emit(
            StreamEvent(seq=self._next(), type=EventType.TOKEN, node=node,
                        content=text, is_partial=True, ts=datetime.utcnow()),
            persist=False,
        )

    def judge(self, node: str, verdict: JudgeVerdict) -> None:
        self.emit(
            StreamEvent(seq=self._next(), type=EventType.JUDGE, node=node,
                        actor="Giám khảo", content=verdict.reasoning,
                        payload={"verdict": verdict.verdict, "scores": verdict.scores},
                        ts=datetime.utcnow()),
            persist=True,
        )

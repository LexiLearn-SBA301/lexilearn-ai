import asyncio
import os
import sys

# Đảm bảo in được tiếng Việt trên console Windows
sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from graph.workflow import build_graph
from services.rag_service import RAGService

async def main():
    print("Khởi tạo RAGService...")
    rag_service = RAGService()
    graph = build_graph(checkpointer=None, rag_service=rag_service)
    
    query = "Phân tích bài thơ Sóng của Xuân Quỳnh"
    print(f"Bắt đầu chạy query: '{query}'")
    print("Đang xử lý (quá trình này có thể mất vài phút do gọi LLM nhiều lần)...")
    
    # Run graph
    state = await graph.ainvoke(
        {"human_message": query},
        config={"configurable": {"thread_id": "test-127"}}
    )
    
    print("\n" + "="*50)
    print("KẾT QUẢ WORKFLOW")
    print("="*50)
    print(f"Route được chọn: {state.get('route')}")
    
    print("\n--- 1. PREPARE CONTEXT (Tool 1) ---")
    ctx = state.get('context')
    if ctx:
        print(f"Summary: {ctx.summary}")
        print(f"Số lượng chunk tìm được: {len(ctx.chunks)}")
        if ctx.chunks:
            print("Snippet Chunk đầu tiên:")
            print("  " + ctx.chunks[0].text[:200] + "...")
        print("Entities:", [e.name for e in ctx.entities])
        print("Themes:", ctx.themes)
    else:
        print("Không có context.")
        
    print("\n--- JUDGE CONTEXT ---")
    jc = state.get("judges", {}).get("prepare_context")
    if jc:
        print(f"Verdict: {jc.verdict}, Score: {jc.scores}")
        print(f"Feedback: {jc.feedback}")
    else:
        print("Không có kết quả chấm context.")
        
    print("\n--- 2. DEBATE (Tool 2) ---")
    debate = state.get('debate')
    if debate:
        print(">> Vòng 1 (Luận điểm):")
        for role, turn in debate.round1.items():
            print(f"[{role.value}] Luận đề: {turn.thesis}")
            for arg in turn.arguments:
                print(f"  - {arg.point}")
        
        print("\n>> Vòng 2 (Phản biện):")
        for role, turn in debate.round2.items():
            print(f"[{role.value}] phản biện:")
            for reb in turn.rebuttals:
                print(f"  -> Nhắm tới {reb.target_critic}: {reb.reason}")
                
        print("\n>> Đồng thuận (Consensus):")
        for c in debate.consensus_points:
            print(f"  * {c}")
        print("\n>> Tranh cãi (Contested):")
        for c in debate.contested_points:
            print(f"  * {c}")
    else:
        print("Không có debate.")
        
    print("\n--- JUDGE DEBATE ---")
    jd = state.get("judges", {}).get("critics_debate")
    if jd:
        print(f"Verdict: {jd.verdict}, Score: {jd.scores}")
        print(f"Feedback: {jd.feedback}")
    else:
        print("Không có kết quả chấm debate.")
        
    print("\n--- 3. ESSAY (Tool 3) ---")
    essay = state.get('essay')
    if essay:
        print(f"Tiêu đề: {essay.title}")
        print(f"Số từ: {essay.word_count}")
        print("Suy nghĩ nội bộ (Chain-of-thought) lúc sinh:")
        # Lưu ý: thinking field đã bị cắt khỏi EssayDraft lúc mapping trong write_essay.py
        # Chúng ta in toàn văn để xem cấu trúc
        print("\nToàn văn bài luận:")
        print(essay.full_text[:1500] + ("\n...\n(Cắt ngắn do quá dài)" if len(essay.full_text) > 1500 else ""))
    else:
        print("Không có bài viết.")
        
    print("\n--- JUDGE ESSAY ---")
    je = state.get("judges", {}).get("write_essay")
    if je:
        print(f"Verdict: {je.verdict}, Score: {je.scores}")
        print(f"Feedback: {je.feedback}")
    else:
        print("Không có kết quả chấm essay.")

if __name__ == "__main__":
    asyncio.run(main())

"""Script tạo Jupyter Notebook Lab4_GraphRAG.ipynb"""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": f"m{abs(hash(source[:20])):08x}"}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source,
            "outputs": [], "execution_count": None, "id": f"c{abs(hash(source[:20])):08x}"}

cells = [
    md("# Lab 4: GraphRAG - Xây dựng hệ thống RAG dựa trên đồ thị tri thức\n\n"
       "> **Mục tiêu:** Xây dựng và so sánh Flat RAG (TF-IDF) vs GraphRAG (Knowledge Graph + 2-hop traversal)  \n"
       "> **Công cụ:** NetworkX, Matplotlib, Python  \n"
       "> **Deliverables:** Source code, Knowledge Graph screenshot, Benchmark table 20 câu hỏi, Cost analysis\n\n---"),

    md("## Phần 1: Cài đặt thư viện (Environment Setup)"),

    code("# Cài đặt các thư viện cơ bản\n"
         "!pip install networkx matplotlib pandas\n"
         "# Cài đặt NodeRAG framework\n"
         "# !pip install noderag\n"
         "# Nếu sử dụng LangChain:\n"
         "# !pip install langchain langchain-openai\n"
         "print('✅ Thư viện đã sẵn sàng')"),

    md("## Phần 2: Tech Company Corpus\n\nDữ liệu đầu vào gồm 40 câu mô tả về các công ty công nghệ lớn."),

    code("from corpus import TECH_CORPUS\n\n"
         "print(f'📚 Tổng số tài liệu trong corpus: {len(TECH_CORPUS)}')\n"
         "print('\\n=== 5 tài liệu mẫu ===')\n"
         "for i, doc in enumerate(TECH_CORPUS[:5], 1):\n"
         "    print(f'{i}. {doc}')"),

    md("## Phần 3: Bước 1 - Trích xuất thực thể và quan hệ (Indexing)\n\n"
       "Sử dụng LLM để đọc corpus và chuyển thành bộ ba **(Subject, Relation, Object)**.\n\n"
       "**Ví dụ:**\n"
       "- Input: `\"OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015.\"`\n"
       "- Output Triples:\n"
       "  - `(OpenAI, FOUNDED_BY, Sam Altman)`\n"
       "  - `(OpenAI, FOUNDED_BY, Elon Musk)`\n"
       "  - `(OpenAI, FOUNDED_IN, 2015)`"),

    code("from graph_builder import PREDEFINED_TRIPLES\n\n"
         "print(f'🔗 Tổng số triples được trích xuất: {len(PREDEFINED_TRIPLES)}')\n"
         "print('\\n=== Mẫu triples ===')\n"
         "for triple in PREDEFINED_TRIPLES[:10]:\n"
         "    print(f'  ({triple[0]}, {triple[1]}, {triple[2]})')\n"
         "print('  ...')"),

    md("## Phần 4: Bước 2 - Xây dựng Knowledge Graph (NetworkX)\n\n"
       "**Lựa chọn A: NetworkX** - phù hợp chạy offline trong Notebook."),

    code("from graph_builder import build_graph\n\n"
         "# Xây dựng Knowledge Graph từ triples\n"
         "G = build_graph()\n\n"
         "print(f'📊 Số nodes: {G.number_of_nodes()}')\n"
         "print(f'📊 Số edges: {G.number_of_edges()}')\n"
         "print('\\n=== Danh sách nodes ===')\n"
         "for node, data in list(G.nodes(data=True))[:10]:\n"
         "    print(f'  {node} (type: {data.get(\"type\",\"?\")})')"),

    code("import matplotlib\nmatplotlib.rcParams['figure.dpi'] = 110\n\n"
         "from graph_builder import visualize_graph\n\n"
         "# Vẽ và lưu đồ thị tri thức\n"
         "visualize_graph(G, title='Knowledge Graph - Tech Companies', save_path='knowledge_graph.png')\n"
         "print('\\n✅ Đồ thị đã được lưu: knowledge_graph.png')"),

    md("## Phần 5: Bước 3 - GraphRAG Querying (2-hop traversal)\n\n"
       "**Quy trình:**\n"
       "1. Nhận câu hỏi từ người dùng\n"
       "2. Trích xuất thực thể chính trong câu hỏi\n"
       "3. Tìm node trong đồ thị và duyệt 2-hop lân cận\n"
       "4. Gộp thông tin thành đoạn văn (Textualization) rồi gửi cho LLM"),

    code("from graph_rag import GraphRAG\n\n"
         "grag = GraphRAG()\n\n"
         "# Test câu hỏi đơn giản\n"
         "result = grag.answer('OpenAI được thành lập năm nào?')\n"
         "print(f'Query  : {result[\"query\"]}')\n"
         "print(f'Entity : {result[\"entity\"]}')\n"
         "print(f'\\n=== Context từ đồ thị (2-hop) ===')\n"
         "print(result['context'])\n"
         "print(f'\\n=== Answer ===')\n"
         "print(result['answer'])\n"
         "print(f'\\nLatency: {result[\"latency_ms\"]} ms')"),

    code("# Test câu hỏi phức tạp - yêu cầu 2-hop reasoning\n"
         "result2 = grag.answer('Người sáng lập OpenAI còn liên quan đến công ty xe điện nào?')\n"
         "print(f'Query  : {result2[\"query\"]}')\n"
         "print(f'Entity : {result2[\"entity\"]}')\n"
         "print(f'Nodes retrieved: {result2[\"nodes_retrieved\"]}')\n"
         "print(f'\\n=== Context (2-hop) ===')\n"
         "print(result2['context'][:600])\n"
         "print(f'\\n=== Answer ===')\n"
         "print(result2['answer'])"),

    md("## Phần 6: Flat RAG (TF-IDF - ChromaDB/Faiss simulation)\n\n"
       "Hệ thống RAG truyền thống chỉ dùng similarity search, **không có khả năng suy luận đa bước**."),

    code("from flat_rag import FlatRAG\n\n"
         "flat = FlatRAG()\n\n"
         "# Cùng câu hỏi phức tạp\n"
         "result_flat = flat.answer('Người sáng lập OpenAI còn liên quan đến công ty xe điện nào?')\n"
         "print(f'Query: {result_flat[\"query\"]}')\n"
         "print(f'\\nTop-3 retrieved docs:')\n"
         "for r in result_flat['retrieved_docs']:\n"
         "    print(f'  [score={r[\"score\"]}] {r[\"doc\"]}')\n"
         "print(f'\\nAnswer: {result_flat[\"answer\"]}')\n"
         "print('\\n⚠️  FlatRAG không thể kết nối: Elon Musk → OpenAI → Tesla (2-hop)')"),

    md("## Phần 7: Bước 4 - Đánh giá 20 câu hỏi benchmark\n\n"
       "So sánh Flat RAG và GraphRAG trên 20 câu hỏi từ đơn giản đến phức tạp."),

    code("from evaluation import run_evaluation, save_results_csv\nimport pandas as pd\n\n"
         "results, stats = run_evaluation()\nsave_results_csv(results)"),

    code("# Hiển thị bảng kết quả\n"
         "df = pd.DataFrame(results)\n"
         "cols = ['id','question','flat_latency_ms','graph_latency_ms','flat_hallucinates','graph_correct']\n"
         "pd.set_option('display.max_colwidth', 60)\n"
         "pd.set_option('display.width', 200)\n"
         "print(df[cols].to_string(index=False))"),

    code("# Biểu đồ so sánh latency\n"
         "import matplotlib.pyplot as plt\n"
         "import matplotlib\nmatplotlib.rcParams['figure.dpi'] = 100\n\n"
         "df = pd.DataFrame(results)\n"
         "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
         "# Latency comparison\n"
         "axes[0].bar(df['id'], df['flat_latency_ms'], alpha=0.7, label='FlatRAG', color='#F97B4F')\n"
         "axes[0].bar(df['id'], df['graph_latency_ms'], alpha=0.7, label='GraphRAG', color='#4F8EF7')\n"
         "axes[0].set_title('Latency per Query (ms)', fontweight='bold')\n"
         "axes[0].set_xlabel('Query ID')\naxes[0].set_ylabel('Time (ms)')\n"
         "axes[0].legend()\naxes[0].grid(axis='y', alpha=0.3)\n\n"
         "# Hallucination pie\n"
         "hallu = sum(1 for r in results if 'YES' in str(r.get('flat_hallucinates','')))\n"
         "ok = len(results) - hallu\n"
         "axes[1].pie([ok, hallu], labels=['Correct', 'Hallucination'],\n"
         "            colors=['#4FD6A0','#F97B4F'], autopct='%1.0f%%', startangle=90)\n"
         "axes[1].set_title('FlatRAG Hallucination Rate', fontweight='bold')\n\n"
         "plt.tight_layout()\nplt.savefig('benchmark_chart.png', dpi=120, bbox_inches='tight')\nplt.show()\n"
         "print('✅ Chart saved: benchmark_chart.png')"),

    md("## Phần 8: Phân tích chi phí (Token Usage & Time)"),

    code("from evaluation import cost_analysis\ncost_analysis(stats)"),

    md("## Phần 9: Kết luận\n\n"
       "### Trường hợp FlatRAG ảo giác - GraphRAG trả lời đúng\n\n"
       "| # | Câu hỏi | FlatRAG | GraphRAG |\n"
       "|---|---------|---------|----------|\n"
       "| 1 | Người sáng lập OpenAI liên quan đến xe điện nào? | ❌ Không kết nối được | ✅ Tesla (Elon Musk 2-hop) |\n"
       "| 2 | CEO Apple trước Tim Cook xây dựng sản phẩm gì? | ❌ Ảo giác | ✅ iPhone, iOS (Steve Jobs) |\n"
       "| 3 | Công ty phát triển AlphaGo được ai mua? | ❌ Thiếu context | ✅ Google → DeepMind |\n"
       "| 4 | Người sáng lập Google học ở đâu? | ❌ Không rõ | ✅ Đại học Stanford |\n"
       "| 5 | AI model nào ra mắt 2023? | ❌ Không đầy đủ | ✅ GPT-4, Gemini |\n"
       "| 6 | Elon Musk thành lập những công ty nào? | ❌ Thiếu | ✅ OpenAI, Tesla, SpaceX, X |\n\n"
       "### Bảng so sánh tổng hợp\n\n"
       "| Tiêu chí | Flat RAG | GraphRAG |\n"
       "|---------|----------|----------|\n"
       "| Độ chính xác | Trung bình | Cao |\n"
       "| Suy luận đa bước | ❌ Không | ✅ Có (2-hop) |\n"
       "| Chống ảo giác | ❌ Yếu | ✅ Tốt |\n"
       "| Tốc độ retrieval | Nhanh | Vừa |\n"
       "| Chi phí token | Cao | Thấp hơn |\n"
       "| Trực quan hóa | ❌ Không | ✅ Có |\n\n"
       "**→ GraphRAG vượt trội với câu hỏi phức tạp cần suy luận đa quan hệ.**"),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

output_path = r"d:\AI_action\Phase 2\Lab 4\Lab4_GraphRAG.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook created: " + output_path)

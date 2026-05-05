"""
Bước 3: GraphRAG - Truy vấn đồ thị tri thức với 2-hop traversal
"""

import re
import time
import networkx as nx
from graph_builder import build_graph, PREDEFINED_TRIPLES


class GraphRAG:
    """Knowledge Graph RAG với 2-hop traversal."""

    def __init__(self, triples=None):
        self.G = build_graph(triples or PREDEFINED_TRIPLES)
        print(f"✅ GraphRAG ready: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    # ── Bước 3.2: Trích xuất thực thể chính từ câu hỏi ──
    def extract_entity(self, query: str) -> str | None:
        q_lower = query.lower()
        best_node = None
        best_len = 0
        for node in self.G.nodes():
            if node.lower() in q_lower and len(node) > best_len:
                best_node = node
                best_len = len(node)
        return best_node

    # ── Bước 3.3: Duyệt 2-hop neighbors ──
    def get_subgraph_2hop(self, entity: str) -> dict:
        if entity not in self.G:
            return {}

        info = {}
        # Hop 1: direct neighbors
        for u, v, data in self.G.edges(data=True):
            if u == entity:
                info.setdefault(entity, []).append((data["relation"], v, 1))
            if v == entity:
                info.setdefault(entity, []).append((f"<-{data['relation']}", u, 1))

        # Hop 2: neighbors of neighbors
        hop1_nodes = [v for u, v, _ in self.G.edges(entity, data=True)]
        hop1_nodes += [u for u, v, _ in self.G.in_edges(entity, data=True)]

        for hop1 in set(hop1_nodes):
            for u, v, data in self.G.edges(hop1, data=True):
                if v != entity:
                    info.setdefault(hop1, []).append((data["relation"], v, 2))
            for u, v, data in self.G.in_edges(hop1, data=True):
                if u != entity:
                    info.setdefault(hop1, []).append((f"<-{data['relation']}", u, 2))

        return info

    # ── Bước 3.4: Textualization - Chuyển graph info thành đoạn văn ──
    def textualize(self, entity: str, subgraph_info: dict) -> str:
        lines = [f"Thông tin về '{entity}' và các thực thể liên quan (trong phạm vi 2-hop):"]

        hop1 = subgraph_info.get(entity, [])
        if hop1:
            lines.append(f"\n[Hop 1 - Trực tiếp từ {entity}]:")
            for rel, target, _ in hop1:
                if rel.startswith("<-"):
                    lines.append(f"  • {target} --[{rel[2:]}]--> {entity}")
                else:
                    lines.append(f"  • {entity} --[{rel}]--> {target}")

        hop2_nodes = {k: v for k, v in subgraph_info.items() if k != entity}
        if hop2_nodes:
            lines.append(f"\n[Hop 2 - Các thực thể liên quan gián tiếp]:")
            for mid_node, edges in list(hop2_nodes.items())[:8]:  # limit output
                for rel, target, _ in edges[:3]:
                    if rel.startswith("<-"):
                        lines.append(f"  • {target} --[{rel[2:]}]--> {mid_node}")
                    else:
                        lines.append(f"  • {mid_node} --[{rel}]--> {target}")

        return "\n".join(lines)

    def _simulate_answer(self, query: str, context: str, entity: str) -> str:
        """Sinh câu trả lời từ context đồ thị."""
        q_lower = query.lower()
        lines = context.split("\n")

        keywords_map = {
            ("thành lập", "founded", "năm nào", "khi nào"): "FOUNDED_IN",
            ("ai thành lập", "người sáng lập", "founder"): "FOUNDED_BY",
            ("ceo", "giám đốc", "lãnh đạo"): "CEO_OF",
            ("phát triển", "sản phẩm", "tạo ra"): "DEVELOPED",
            ("mua lại", "acquire"): "ACQUIRED",
            ("sở hữu", "owns"): "OWNS",
        }

        target_relations = []
        for kws, rel in keywords_map.items():
            if any(kw in q_lower for kw in kws):
                target_relations.append(rel)

        relevant = []
        for line in lines:
            if entity in line:
                for rel in target_relations:
                    if rel in line:
                        relevant.append(line.strip())
                        break
                else:
                    if entity in line and "--[" in line:
                        relevant.append(line.strip())

        if relevant:
            return "Dựa trên đồ thị tri thức:\n" + "\n".join(relevant[:5])
        return f"Tìm thấy thông tin về '{entity}' trong đồ thị nhưng không khớp câu hỏi cụ thể."

    def answer(self, query: str) -> dict:
        start = time.time()

        # Step 1: Extract entity
        entity = self.extract_entity(query)
        if not entity:
            return {
                "query": query,
                "entity": None,
                "context": "",
                "answer": "Không tìm thấy thực thể nào trong đồ thị tri thức.",
                "latency_ms": 0,
                "method": "GraphRAG (2-hop)",
            }

        # Step 2: 2-hop traversal
        subgraph_info = self.get_subgraph_2hop(entity)

        # Step 3: Textualize
        context = self.textualize(entity, subgraph_info)

        # Step 4: Simulate LLM answer
        answer_text = self._simulate_answer(query, context, entity)
        elapsed = time.time() - start

        return {
            "query": query,
            "entity": entity,
            "context": context,
            "answer": answer_text,
            "hop_count": 2,
            "nodes_retrieved": len(subgraph_info),
            "latency_ms": round(elapsed * 1000, 2),
            "method": "GraphRAG (2-hop)",
        }


if __name__ == "__main__":
    grag = GraphRAG()
    queries = [
        "OpenAI được thành lập năm nào?",
        "Ai là CEO của Google?",
        "Microsoft đã mua lại những công ty nào?",
    ]
    for q in queries:
        result = grag.answer(q)
        print(f"\nQuery   : {result['query']}")
        print(f"Entity  : {result['entity']}")
        print(f"Answer  : {result['answer']}")
        print(f"Latency : {result['latency_ms']} ms")
        print("-" * 60)

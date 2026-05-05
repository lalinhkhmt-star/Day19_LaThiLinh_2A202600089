"""
Bước 1 & 2: Trích xuất thực thể - quan hệ và xây dựng Knowledge Graph
Sử dụng NetworkX (Lựa chọn A - offline)
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import time
from corpus import TECH_CORPUS

# ── Triples được trích xuất thủ công từ corpus (mô phỏng LLM extraction) ──
PREDEFINED_TRIPLES = [
    ("OpenAI", "FOUNDED_BY", "Sam Altman"),
    ("OpenAI", "FOUNDED_BY", "Elon Musk"),
    ("OpenAI", "FOUNDED_IN", "2015"),
    ("Sam Altman", "CEO_OF", "OpenAI"),
    ("Sam Altman", "ROLE_SINCE", "2019"),
    ("OpenAI", "DEVELOPED", "GPT-4"),
    ("GPT-4", "RELEASED_IN", "2023"),
    ("OpenAI", "DEVELOPED", "ChatGPT"),
    ("ChatGPT", "RELEASED_IN", "2022"),
    ("Microsoft", "INVESTED_IN", "OpenAI"),
    ("Microsoft", "INVESTMENT_AMOUNT", "10 tỷ đô"),
    ("Google", "FOUNDED_BY", "Larry Page"),
    ("Google", "FOUNDED_BY", "Sergey Brin"),
    ("Google", "FOUNDED_IN", "1998"),
    ("Larry Page", "STUDIED_AT", "Đại học Stanford"),
    ("Sergey Brin", "STUDIED_AT", "Đại học Stanford"),
    ("Google", "DEVELOPED", "Gemini"),
    ("Gemini", "RELEASED_IN", "2023"),
    ("Google", "ACQUIRED", "DeepMind"),
    ("Google", "ACQUIRED_IN", "2014"),
    ("DeepMind", "DEVELOPED", "AlphaGo"),
    ("Apple", "FOUNDED_BY", "Steve Jobs"),
    ("Apple", "FOUNDED_BY", "Steve Wozniak"),
    ("Apple", "FOUNDED_BY", "Ronald Wayne"),
    ("Apple", "FOUNDED_IN", "1976"),
    ("Steve Jobs", "CEO_OF", "Apple"),
    ("Apple", "DEVELOPED", "iOS"),
    ("Apple", "DEVELOPED", "iPhone"),
    ("iPhone", "RELEASED_IN", "2007"),
    ("Tim Cook", "CEO_OF", "Apple"),
    ("Tim Cook", "ROLE_SINCE", "2011"),
    ("Meta", "FOUNDED_BY", "Mark Zuckerberg"),
    ("Meta", "FOUNDED_IN", "2004"),
    ("Meta", "FORMERLY_KNOWN_AS", "Facebook"),
    ("Meta", "RENAMED_IN", "2021"),
    ("Meta", "DEVELOPED", "LLaMA"),
    ("Mark Zuckerberg", "CEO_OF", "Meta"),
    ("Meta", "OWNS", "Instagram"),
    ("Meta", "OWNS", "WhatsApp"),
    ("Amazon", "FOUNDED_BY", "Jeff Bezos"),
    ("Amazon", "FOUNDED_IN", "1994"),
    ("Jeff Bezos", "CEO_OF", "Amazon"),
    ("Amazon", "DEVELOPED", "AWS"),
    ("Andy Jassy", "CEO_OF", "Amazon"),
    ("Andy Jassy", "ROLE_SINCE", "2021"),
    ("Amazon", "ACQUIRED", "Whole Foods"),
    ("Microsoft", "FOUNDED_BY", "Bill Gates"),
    ("Microsoft", "FOUNDED_BY", "Paul Allen"),
    ("Microsoft", "FOUNDED_IN", "1975"),
    ("Satya Nadella", "CEO_OF", "Microsoft"),
    ("Satya Nadella", "ROLE_SINCE", "2014"),
    ("Microsoft", "DEVELOPED", "Windows"),
    ("Microsoft", "ACQUIRED", "LinkedIn"),
    ("Microsoft", "ACQUIRED", "GitHub"),
    ("NVIDIA", "FOUNDED_BY", "Jensen Huang"),
    ("NVIDIA", "FOUNDED_IN", "1993"),
    ("Jensen Huang", "CEO_OF", "NVIDIA"),
    ("NVIDIA", "DEVELOPED", "H100"),
    ("NVIDIA", "PARTNERS_WITH", "Microsoft"),
    ("Tesla", "FOUNDED_BY", "Elon Musk"),
    ("Tesla", "FOUNDED_IN", "2003"),
    ("Elon Musk", "CEO_OF", "Tesla"),
    ("Tesla", "DEVELOPED", "Autopilot"),
    ("SpaceX", "FOUNDED_BY", "Elon Musk"),
    ("SpaceX", "FOUNDED_IN", "2002"),
    ("Elon Musk", "OWNS", "X"),
    ("X", "FORMERLY_KNOWN_AS", "Twitter"),
    ("X", "ACQUIRED_IN", "2022"),
]

# ── Màu sắc cho từng loại node ──
NODE_COLORS = {
    "company":  "#4F8EF7",
    "person":   "#F97B4F",
    "product":  "#4FD6A0",
    "year":     "#C084FC",
    "other":    "#94A3B8",
}

COMPANIES = {"OpenAI","Google","Apple","Meta","Amazon","Microsoft","NVIDIA","Tesla","SpaceX",
             "DeepMind","Instagram","WhatsApp","LinkedIn","GitHub","Whole Foods","X","AWS",
             "Facebook","Twitter"}
PERSONS   = {"Sam Altman","Elon Musk","Larry Page","Sergey Brin","Steve Jobs","Steve Wozniak",
             "Ronald Wayne","Tim Cook","Mark Zuckerberg","Jeff Bezos","Andy Jassy","Bill Gates",
             "Paul Allen","Satya Nadella","Jensen Huang","Martin Eberhard","Marc Tarpenning"}
PRODUCTS  = {"GPT-4","ChatGPT","Gemini","AlphaGo","iOS","iPhone","LLaMA","Windows","H100","Autopilot"}


def classify_node(node: str) -> str:
    if node in COMPANIES: return "company"
    if node in PERSONS:   return "person"
    if node in PRODUCTS:  return "product"
    if node.isdigit() or (len(node) == 4 and node.startswith("20") or node.startswith("19")):
        return "year"
    return "other"


def build_graph(triples=None) -> nx.MultiDiGraph:
    """Xây dựng Knowledge Graph từ danh sách triples."""
    G = nx.MultiDiGraph()
    if triples is None:
        triples = PREDEFINED_TRIPLES

    for subj, rel, obj in triples:
        G.add_node(subj, type=classify_node(subj))
        G.add_node(obj,  type=classify_node(obj))
        G.add_edge(subj, obj, relation=rel)

    print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def visualize_graph(G: nx.MultiDiGraph, title="Knowledge Graph - Tech Companies",
                    figsize=(22, 16), save_path=None):
    """Vẽ đồ thị tri thức bằng Matplotlib."""
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=18, fontweight="bold", pad=20)

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # Màu node
    color_map = [NODE_COLORS.get(G.nodes[n].get("type","other"), NODE_COLORS["other"])
                 for n in G.nodes()]
    size_map  = [2800 if G.nodes[n].get("type") in ("company","person") else 1600
                 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=size_map, alpha=0.92)
    nx.draw_networkx_labels(G, pos, font_size=7.5, font_weight="bold")

    # Edges
    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, edge_color="#64748B", arrows=True,
                           arrowsize=15, alpha=0.6,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=6, alpha=0.8)

    # Legend
    legend_handles = [mpatches.Patch(color=v, label=k.capitalize())
                      for k, v in NODE_COLORS.items()]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=10)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📸 Graph saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    G = build_graph()
    visualize_graph(G, save_path="knowledge_graph.png")

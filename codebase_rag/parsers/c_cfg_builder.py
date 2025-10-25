from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tree_sitter import Node
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 公開資料結構
# ──────────────────────────────────────────────────────────

@dataclass
class BasicBlock:
    idx: int
    kind: str  # ENTRY / NORMAL / EXIT / LATCH / AFTER / MERGE / SWITCH_DISPATCH 等

@dataclass
class StmtRec:
    id: str
    kind: str
    src: str
    s: Tuple[int, int]   # (row, col) 0-based
    e: Tuple[int, int]
    block_idx: int

@dataclass
class Edge:
    src: int
    dst: int
    label: str  # 'next' / 'T' / 'F' / 'loop' / 'back' / 'break' / 'continue' / 'ret' / 'goto' / 'case:<v>' / 'fallthrough'

# ──────────────────────────────────────────────────────────
# 小工具
# ──────────────────────────────────────────────────────────

_TS_KIND_MAP = {
    # statements
    "if_statement": "IF",
    "switch_statement": "SWITCH",
    "for_statement": "LOOP",
    "while_statement": "LOOP",
    "do_statement": "LOOP",
    "labeled_statement": "LABEL",
    "case_statement": "CASE",
    "default_statement": "DEFAULT",
    "break_statement": "BREAK",
    "continue_statement": "CONTINUE",
    "return_statement": "RETURN",
    "goto_statement": "GOTO",
    "declaration": "DECL",
    "expression_statement": "EXPR",
    # expressions（供後續 PDG/CPG 用，這版先記 kind）
    "assignment_expression": "ASSIGN",
    "call_expression": "CALL",
    "binary_expression": "BINOP",
    "conditional_expression": "COND_EXPR",
}

def _text(n: Node, src: bytes) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "ignore")

def _mk_stmt_id(fn: Node, n: Node) -> str:
    # 穩定 id：以檔內位移為主
    return f"{fn.start_byte}:{n.start_byte}-{n.end_byte}"

def _is_executable_stmt(n: Node) -> bool:
    return n.type in (
        "if_statement","switch_statement",
        "for_statement","while_statement","do_statement",
        "labeled_statement","case_statement","default_statement",
        "break_statement","continue_statement","return_statement",
        "goto_statement","declaration","expression_statement",
    )

# ──────────────────────────────────────────────────────────
# 主流程：build_cfg_for_function
# ──────────────────────────────────────────────────────────

def build_cfg_for_function(src: bytes, fn_node: Node) -> dict:
    """
    把一個 function_definition 轉成 {blocks, stmts, cfg_edges}
    """
    blocks: List[BasicBlock] = []
    stmts:  List[StmtRec]   = []
    edges:  List[Edge]      = []

    def new_block(kind="NORMAL") -> int:
        idx = len(blocks)
        blocks.append(BasicBlock(idx, kind))
        return idx

    def add_edge(a: int, b: int, label="next") -> None:
        edges.append(Edge(a, b, label))

    # 預先建立 ENTRY/EXIT 與第一個執行塊
    entry = new_block("ENTRY")
    exitb = new_block("EXIT")
    cur   = new_block("NORMAL")
    add_edge(entry, cur, "enter")

    # label/goto 的回填表
    label_block: Dict[str, int] = {}
    pending_gotos: List[Tuple[int, str]] = []  # (from_block, label)

    # 巢狀控制結構堆疊（for/while/do/switch）以便回填 break/continue
    ctrl_stack: List[Dict] = []  # item: {"type": "loop"/"switch", "latch": int, "after": int}

    body = fn_node.child_by_field_name("body")  # compound_statement

    # 遞迴展開 compound_statement 的「直接語句序列」
    def iter_statements(node: Optional[Node]):
        if node is None:
            return
        if node.type == "compound_statement":
            for ch in node.named_children:
                if _is_executable_stmt(ch):
                    yield ch
                elif ch.type == "compound_statement":
                    for s in iter_statements(ch):
                        yield s
        else:
            yield node

    def push_stmt(n: Node, kind: Optional[str] = None):
        nonlocal cur
        if n.type != "compound_statement":
            k = kind or _TS_KIND_MAP.get(n.type, n.type.upper())
            stmts.append(StmtRec(
                id=_mk_stmt_id(fn_node, n),
                kind=k,
                src=_text(n, src),
                s=n.start_point, e=n.end_point,
                block_idx=cur
            ))

    def handle_if(n: Node):
        nonlocal cur
        then_node = n.child_by_field_name("consequence")
        else_node = n.child_by_field_name("alternative")

        then_b = new_block("NORMAL")
        add_edge(cur, then_b, "T")

        else_b = None
        if else_node:
            else_b = new_block("NORMAL")
            add_edge(cur, else_b, "F")

        merge_b = new_block("MERGE")

        # then
        prev_pred = cur
        cur = then_b
        for s in iter_statements(then_node):
            cur = dispatch_stmt(s)
        add_edge(cur, merge_b, "next")

        # else
        if else_b is not None:
            cur = else_b
            for s in iter_statements(else_node):
                cur = dispatch_stmt(s)
            add_edge(cur, merge_b, "next")
        else:
            # 無 else：直接補 F→merge（pred 已存在）
            add_edge(prev_pred, merge_b, "F")

        cur = merge_b
        return cur

    def handle_loop(n: Node, loop_kind: str):
        nonlocal cur
        body_b  = new_block("NORMAL")
        latch_b = new_block("LATCH")
        after_b = new_block("AFTER")

        add_edge(cur, body_b, "loop")
        ctrl_stack.append({"type": "loop", "latch": latch_b, "after": after_b})

        cur = body_b
        body_node = n.child_by_field_name("body")
        for s in iter_statements(body_node):
            cur = dispatch_stmt(s)

        add_edge(cur, latch_b, "next")
        add_edge(latch_b, body_b, "back")

        ctrl_stack.pop()
        cur = after_b
        return cur

    def handle_switch(n: Node):
        nonlocal cur
        after_b = new_block("AFTER")

        # 掃各 case/default 作為 leaders
        cases: List[Tuple[str, int, Node]] = []
        body = n.child_by_field_name("body")
        if body:
            for ch in body.named_children:
                if ch.type in ("case_statement","default_statement","labeled_statement"):
                    label_txt = (
                        ch.type if ch.type != "labeled_statement"
                        else (_text(ch.child_by_field_name("label"), src) if ch.child_by_field_name("label") else "label")
                    )
                    bidx = new_block("NORMAL")
                    cases.append((label_txt, bidx, ch))

        # pred → 各 case（symbolic dispatch）
        for (lbl, bidx, _node) in cases:
            add_edge(cur, bidx, f"case:{lbl}")

        ctrl_stack.append({"type": "switch", "after": after_b})

        # 順序執行每個 case：若沒有 break → fall-through
        for i, (_lbl, bidx, node) in enumerate(cases):
            cur = bidx
            # case/default/labeled_statement 的 body：最後一個 named child
            named = [c for c in node.named_children]
            body_stmt = named[-1] if named else None
            if body_stmt is not None:
                for s in iter_statements(body_stmt):
                    cur = dispatch_stmt(s)
            # fall-through or exit to after
            if i + 1 < len(cases):
                add_edge(cur, cases[i+1][1], "fallthrough")
            else:
                add_edge(cur, after_b, "next")

        ctrl_stack.pop()
        cur = after_b
        return cur

    def dispatch_stmt(n: Node) -> int:
        nonlocal cur
        t = n.type
        push_stmt(n)

        # label：把目前 block 綁到該 label（若尚未綁）
        if t == "labeled_statement":
            label_node = n.child_by_field_name("label")
            if label_node:
                name = _text(label_node, src)
                if name not in label_block:
                    label_block[name] = cur
            # 繼續處理 body（在同一個 block）
            inner = None
            for ch in n.named_children[::-1]:
                if ch.type != "identifier":
                    inner = ch
                    break
            if inner:
                cur = dispatch_stmt(inner)
            return cur

        if t == "if_statement":
            return handle_if(n)

        if t in ("for_statement","while_statement","do_statement"):
            return handle_loop(n, "do" if t == "do_statement" else "loop")

        if t == "switch_statement":
            return handle_switch(n)

        if t == "break_statement":
            # 連到最近 switch/loop 的 after
            for k in reversed(ctrl_stack):
                if k["type"] in ("loop","switch"):
                    add_edge(cur, k["after"], "break")
                    cur = new_block("NORMAL")
                    return cur
            return cur

        if t == "continue_statement":
            for k in reversed(ctrl_stack):
                if k["type"] == "loop":
                    add_edge(cur, k["latch"], "continue")
                    cur = new_block("NORMAL")
                    return cur
            return cur

        if t == "goto_statement":
            name_node = n.child_by_field_name("name")
            if name_node:
                pending_gotos.append((cur, _text(name_node, src)))
            cur = new_block("NORMAL")
            return cur

        if t == "return_statement":
            add_edge(cur, exitb, "ret")
            cur = new_block("NORMAL")
            return cur

        # 其他（DECL/EXPR…）維持在同一塊
        return cur

    # 走 body
    for s in iter_statements(body):
        cur = dispatch_stmt(s)

    # function 正常結束：最後落點到 EXIT 前的收斂
    add_edge(cur, exitb, "next")

    # 回填 gotos
    for (src_b, name) in pending_gotos:
        if name in label_block:
            add_edge(src_b, label_block[name], "goto")

    # 轉成可序列化 dict；span 轉 list（避免 tuple 造成寫入失敗）
    bundle = {
        "blocks": [b.__dict__ for b in blocks],
        "stmts":  [s.__dict__ for s in stmts],
        "cfg_edges":[e.__dict__ for e in edges],
    }
    for s in bundle["stmts"]:
        if isinstance(s["s"], tuple):
            s["s"] = list(s["s"])
        if isinstance(s["e"], tuple):
            s["e"] = list(s["e"])

    logger.debug("CFG built: blocks=%d, stmts=%d, edges=%d",
                 len(bundle["blocks"]), len(bundle["stmts"]), len(bundle["cfg_edges"]))
    return bundle

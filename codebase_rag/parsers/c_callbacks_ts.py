# codebase_rag/parsers/c_callbacks_ts.py
from __future__ import annotations
from typing import List, Tuple, Optional,Iterable
from tree_sitter import Node, Query, Language
from loguru import logger

# (kind, field, callee, line)
CBRec = Tuple[str, str, str, int]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap_ident_like(node: Optional[Node], src: bytes) -> Optional[Node]:
    """
    把各種包裝拆到 identifier：
      identifier
      unary_expression: &identifier
      parenthesized_expression: (expr)
      cast_expression: (type)expr
    """
    cur = node
    while cur is not None:
        if cur.type == "identifier":
            return cur
        if cur.type == "unary_expression":
            cur = cur.child_by_field_name("argument") or cur.child_by_field_name("operand")
            continue
        if cur.type == "parenthesized_expression":
            inner = cur.child_by_field_name("expression")
            if inner is None and len(cur.children) > 1:
                inner = cur.children[1]
            cur = inner
            continue
        if cur.type == "cast_expression":
            cur = cur.children[-1] if cur.children else None
            continue
        break
    return None

def _collect_field(expr: Node, src: bytes, out: List[str]) -> bool:
    """
    field_expression := <argument> ('.'|'->') <field_identifier>
    支援巢狀 a->b.c（會把最內層 field 放在 out[0]）
    """
    if expr.type != "field_expression":
        return False
    fld = expr.child_by_field_name("field")
    arg = expr.child_by_field_name("argument")
    if not fld or fld.type != "field_identifier":
        return False
    out.append(_text(fld, src))
    return _collect_field(arg, src, out) if arg is not None else True

# ──────────────────────────────────────────────────────────────────────────────
# Queries（依你的 grammar：有 initializer_pair/field_designator/value/designator）
# ──────────────────────────────────────────────────────────────────────────────

# 1) 指定初始化：.field = <expr>
#   grammar：initializer_pair + designator: (field_designator (field_identifier))
#   注意：這個 grammar 沒有 designator_list / designated_initializer
_INIT_PAIR_QUERY = r"""
(
  initializer_pair @pair
    designator: (field_designator (field_identifier) @field)
    value:      (_) @val
)
"""

_ASSIGN_QUERY = r"""
(
  assignment_expression @pair
    left:  (field_expression
             argument: (_)
             field: (field_identifier) @field)
    right: (identifier) @val
)
"""

_API_MAP = {
    "INIT_WORK":          ("WORK",   "work.handler",          1),
    "INIT_DELAYED_WORK":  ("WORK",   "delayed_work.handler",  1),
    "timer_setup":        ("TIMER",  "timer.handler",         1),
    "setup_timer":        ("TIMER",  "timer.handler",         1),
    "tasklet_init":       ("TASKLET","tasklet.handler",       1),
    "request_irq":        ("IRQ",    "irq.handler",           1),
    "kthread_run":        ("THREAD", "kthread.entry",         0),
}

def match_designated_pairs(root: Node, src: bytes, lang: Language) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    try:
        q = Query(lang, _INIT_PAIR_QUERY)   # ← 用 lang
    except Exception as e:
        logger.debug(f"init_pair query compile error: {e}")
        return out

    caps = q.captures(root)
    cur_pair: Optional[Node] = None
    field_name: Optional[str] = None
    val_node: Optional[Node] = None

    def flush():
        nonlocal field_name, val_node, cur_pair
        if not field_name or val_node is None or cur_pair is None:
            return
        ident = _unwrap_ident_like(val_node, src)
        if ident is None:
            for ch in val_node.children:
                ident = _unwrap_ident_like(ch, src)
                if ident is not None:
                    break
        if ident is None:
            return
        callee = _text(ident, src)
        line_no = cur_pair.start_point[0] + 1
        out.append((field_name, callee, line_no))

    for node, name in caps:
        if name == "pair":
            if cur_pair is not None:
                flush()
            cur_pair = node
            field_name = None
            val_node = None
        elif name == "field":
            field_name = _text(node, src)
        elif name == "val":
            val_node = node
    if cur_pair is not None:
        flush()

    return out

def match_assignment_exprs(root: Node, src: bytes, lang: Language) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    try:
        q = Query(lang, _ASSIGN_QUERY)      # ← 用 lang
    except Exception as e:
        logger.debug(f"assign query compile error: {e}")
        return out

    caps = q.captures(root)
    cur_pair: Optional[Node] = None
    field_name: Optional[str] = None
    val_node: Optional[Node] = None

    def flush():
        nonlocal field_name, val_node, cur_pair
        if not field_name or val_node is None or cur_pair is None:
            return
        ident = _unwrap_ident_like(val_node, src)
        if ident is None:
            for ch in val_node.children:
                ident = _unwrap_ident_like(ch, src)
                if ident is not None:
                    break
        if ident is None:
            return
        callee = _text(ident, src)
        line_no = cur_pair.start_point[0] + 1
        out.append((field_name, callee, line_no))

    for node, name in caps:
        if name == "pair":
            if cur_pair is not None:
                flush()
            cur_pair = node
            field_name = None
            val_node = None
        elif name == "field":
            field_name = _text(node, src)
        elif name == "val":
            val_node = node
    if cur_pair is not None:
        flush()

    return out

def match_api_calls(node: Node, src: bytes) -> Optional[Tuple[str, str, str, int]]:
    # 不需語言物件
    if node.type != "call_expression":
        return None
    func_node = node.child_by_field_name("function")
    if func_node is None:
        return None
    func_name = _text(func_node, src)
    base = func_name.split(".")[-1]
    if base not in _API_MAP:
        return None
    kind, field, idx = _API_MAP[base]

    args = node.child_by_field_name("arguments")
    if args is None:
        return None
    exprs = [c for c in args.children if c.type not in (",", "(", ")")]
    if idx >= len(exprs):
        return None
    target = _unwrap_ident_like(exprs[idx], src)
    if target is None:
        return None
    callee = _text(target, src)
    line_no = node.start_point[0] + 1
    return (kind, field, callee, line_no)

def _text(n: Node, src: bytes) -> str:
    """Return source slice text for a node (呼叫端都以 (_node, src) 傳入)。"""
    return src[n.start_byte:n.end_byte].decode('utf-8', 'ignore')

def _walk(n: Node) -> Iterable[Node]:
    yield n
    for c in n.children:
        yield from _walk(c)

def _nearest_decl_container(root: Node, pos: int) -> Optional[Node]:
    n = root.named_descendant_for_byte_range(pos, pos+1)
    while n and n.type not in ("declaration","init_declarator","initializer_list","translation_unit"):
        n = n.parent
    return n

def _iter_struct_inits(container: Node) -> Iterable[Node]:
    """在宣告/初始化子樹裡找到所有 initializer_list（就是 {...} 那顆）"""
    for n in _walk(container):
        if n.type == "initializer_list":
            yield n

def _pick_field_identifier(designator: Node) -> Optional[Node]:
    # designator 是 field_designator，孩子裡會有 field_identifier
    for ch in designator.children:
        if ch.type == "field_identifier":
            return ch
    # 有些版本會多一層
    fi = designator.child_by_field_name("field")
    return fi

def from_designated_init(root: Node, src: bytes, lang: Language) -> List[CBRec]:
    recs: List[CBRec] = []
    for field, callee, line in match_designated_pairs(root, src, lang):
        recs.append(("SET", field, callee, line))
    logger.debug(f"from_designated_init: {len(recs)} hits")
    return recs

def from_assignment(root: Node, src: bytes, lang: Language) -> List[CBRec]:
    recs: List[CBRec] = []
    for field, callee, line in match_assignment_exprs(root, src, lang):
        recs.append(("SET", field, callee, line))
    logger.debug(f"from_assignment: {len(recs)} hits")
    return recs

def from_known_apis(root: Node, src: bytes) -> List[CBRec]:
    out: List[CBRec] = []
    for n in _walk(root):
        if n.type == "call_expression":
            res = match_api_calls(n, src)
            if res:
                kind, field, callee, line = res
                out.append((kind, field, callee, line))
    logger.debug(f"from_known_apis: {len(out)} hits")
    return out

def _from_initializer_pair(pair: Node, src: bytes) -> Optional[Tuple[str, str, int]]:
    # 形如 ".scan = cfg80211_rtw_scan"
    designator = pair.child_by_field_name("designator")
    value = pair.child_by_field_name("value")
    if not designator or not value:
        return None
    fi = _pick_field_identifier(designator) if designator.type == "field_designator" else None
    if not fi:
        return None
    # 只收值是 identifier 的（函式名）
    if value.type != "identifier":
        return None
    field = _text(fi, src)
    fn = _text(value, src)
    line = fi.start_point.row + 1
    return (field, fn, line)

def _from_assignment_expr(expr: Node, src: bytes) -> Optional[Tuple[str, str, int]]:
    # 形如 "(...) .scan = cfg80211_rtw_scan" 或 "ops.scan = foo"
    if expr.type != "assignment_expression":
        return None
    left = expr.child_by_field_name("left")
    right = expr.child_by_field_name("right")
    if not left or not right:
        return None
    # left 需是 field_expression，且要有 field_identifier
    if left.type != "field_expression":
        return None
    fi = left.child_by_field_name("field")
    if not fi or fi.type != "field_identifier":
        return None
    if right.type != "identifier":
        return None
    field = _text(fi, src)
    fn = _text(right, src)
    line = fi.start_point.row + 1
    return (field, fn, line)

def extract_struct_init_kv(container: Node, src: bytes) -> List[Tuple[str,str,int]]:
    """通吃：在 container（通常是 init_declarator 或 declaration）裡，把
       1) designated initializer 的 pair
       2) 被巨集切成 assignment_expression 的項目
       都抓出來。"""
    out: List[Tuple[str,str,int]] = []
    for il in _iter_struct_inits(container):
        for ch in il.children:
            if ch.type == "initializer_pair":
                rec = _from_initializer_pair(ch, src)
                if rec: out.append(rec)
            elif ch.type == "assignment_expression":
                rec = _from_assignment_expr(ch, src)
                if rec: out.append(rec)
            # 其它像逗號、ERROR、preproc_directive 一律略過
    return out

#def extract_callbacks_c(root: Node, src: bytes) -> List[Tuple[str,str,int]]:
#    """你的對外主函式：走整棵檔案，把所有『struct 初始化』裡的 (field -> fn) 都吐出來"""
#    results: List[Tuple[str,str,int]] = []
#    # 快速路：掃所有 init_declarator；若你有目標白名單（如 cfg80211_ops），可先用 bytes 搜再縮小
#    for n in _walk(root):
#        if n.type == "init_declarator":
#            pairs = extract_struct_init_kv(n, src)
#            results.extend(pairs)
#    logger.debug(results)
#    return results

def extract_callbacks_c(root: Node, src: bytes) -> List[CBRec]:
    """走整棵檔案，把所有『struct 初始化』裡的 (field -> fn) 都吐出來，統一為 4 欄位 CBRec。"""
    out: List[CBRec] = []
    for n in _walk(root):
        if n.type == "init_declarator":
            pairs: List[Tuple[str, str, int]] = extract_struct_init_kv(n, src)
            # 補上 kind="SET"
            out.extend([("SET", field, fn, line) for (field, fn, line) in pairs])
    logger.debug(out)
    return out
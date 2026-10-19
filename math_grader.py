#!/usr/bin/env python3
"""
Answer equivalence for MATH with every answer type (used by --dataset math35*; GSM8K / MATH L1-3 keep their numeric
grader and geometry3k keeps expressions_match).

math_equal(pred, gold) compares two LaTeX answers (e.g. the contents of the last \\boxed{}):
  1. equal after normalisation: Hendrycks-style (\\dfrac, \\left/\\right, spacing, $, degrees, %, units, \\text{}),
     plus thousands separators, number-base subscripts and 'x =' prefixes;
  2. structured answers element by element: tuples and intervals (ordered; bracket types must match), sets and
     unbracketed lists (unordered; 'or'/'and' separate too), a \\pm b as {a+b, a-b}, unions of intervals,
     pmatrix/bmatrix (a column vector also matches a tuple);
  3. scalars with sympy: exact equality for exact inputs (1e-9 relative numerically, simplify() for symbols), 1e-4
     relative when either side has a decimal point; i is the imaginary unit; option letters and words as text.
Degrees vs radians and percent vs fraction are not converted (0.25 does not match 25\\%).
"""

from __future__ import annotations

import math
import re
import signal
import threading
from contextlib import contextmanager
from typing import List, Optional, Tuple

import sympy
from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                        standard_transformations)

_UNIT_WORDS = (r"(?:sq(?:uare)?\s*)?(?:units?|cm|mm|km|m|meters?|in|inches|inch|ft|feet|foot|yd|yards?|miles?|degrees?|"
               r"dollars?|cents?|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|years?|mph|pounds?|lbs?|ounces?|oz|"
               r"grams?|kg|g|liters?|ml|gallons?|people|students|ways|points?)")


@contextmanager
def _time_limit(seconds: float):
    """SIGALRM time limit around sympy calls (main thread only; elsewhere no limit)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _unwrap(s: str, cmd: str) -> str:
    """Replace every \\cmd{X} by X (nested braces allowed)."""
    key = "\\" + cmd + "{"
    while True:
        start = s.find(key)
        if start < 0:
            return s
        depth, i = 0, start + len(key)
        while i < len(s):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                if depth == 0:
                    break
                depth -= 1
            i += 1
        if i >= len(s):
            return s
        s = s[:start] + s[start + len(key):i] + s[i + 1:]


def normalize(s: str) -> str:
    s = s.strip().replace("\n", " ")
    s = _unwrap(_unwrap(s, "boxed"), "fbox")
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = s.replace("\\!", "").replace("{,}", "").replace("\\$", "").replace("$", "")
    s = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big|bigg|Bigg)(?![a-zA-Z])", "", s)
    s = re.sub(r"\\displaystyle|(?<!\\)\\[,;: ]|\\q?quad|~", " ", s)
    s = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ|°|\\degree", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"(?<=[\d}])\s*(?:\\text\{\s*(?:or|and)\s*\}|\bor\b|\band\b)\s*(?=[-\d\\(])", ",", s)
    # units after a number: \text{ cm}^2, \mbox{ inches}, or bare words
    s = re.sub(r"(?<=[\d})])\s*\\(?:text|mbox|mathrm|textrm)\{\s*" + _UNIT_WORDS + r"\s*\}(?:\^\s*\{?[23]\}?|²|³)?\s*\.?$", "", s)
    s = re.sub(r"(?<=[\d})])\s+" + _UNIT_WORDS + r"(?:\^\s*\{?[23]\}?|²|³)?\s*\.?$", "", s)
    for cmd in ("text", "textbf", "textit", "textrm", "mathrm", "mathbf", "mbox", "operatorname"):
        s = _unwrap(s, cmd)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"_\{(\w+)\}", r"_\1", s)  # 1011_{2} -> 1011_2
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):  # 10,000
        s = s.replace(",", "")
    s = s.rstrip(".")
    if s.startswith("."):
        s = "0" + s
    return s


def _assignments(p: str, g: str) -> Tuple[str, str]:
    """Drop 'x=' style left sides: from pred when gold has none, from gold when its left side is one variable."""
    if "=" in p and "=" not in g:
        p = re.sub(r"(^|,)[a-zA-Z](?:_\w+)?=", r"\1", p)
        if "=" in p:
            p = p.rsplit("=", 1)[1]
    elif "=" in g and "=" not in p and re.fullmatch(r"[a-zA-Z](?:_\w+)?=[^=]+", g):
        g = g.split("=", 1)[1]
    elif "=" in p and "=" in g:
        (pl, pr), (gl, gr) = p.split("=", 1), g.split("=", 1)
        if pl == gl:
            p, g = pr, gr
    return p, g


def _split_top(s: str, sep: str = ",") -> List[str]:
    out, depth, cur, i = [], 0, "", 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth == 0 and s.startswith(sep, i):
            out.append(cur)
            cur, i = "", i + len(sep)
            continue
        cur += c
        i += 1
    out.append(cur)
    return out


def _outer(s: str) -> Optional[Tuple[str, str, str]]:
    """(open, inside, close) if s is one bracketed group ( ) [ ] \\{ \\} covering the whole string."""
    if s.startswith("\\{") and s.endswith("\\}"):
        return "{", s[2:-2], "}"
    if len(s) < 2 or s[0] not in "([" or s[-1] not in ")]":
        return None
    depth = 0
    for i, c in enumerate(s):
        depth += c in "(["
        depth -= c in ")]"
        if depth == 0 and i < len(s) - 1:
            return None
    return s[0], s[1:-1], s[-1]


def structure(s: str):
    """(kind, open, items, close) for structured answers, else None. kind: matrix | set | list | ordered | union | pm."""
    m = re.fullmatch(r"\\begin\{[pbv]matrix\}(.*)\\end\{[pbv]matrix\}", s)
    if m:
        rows = [r.split("&") for r in m.group(1).split("\\\\") if r != ""]
        if len({len(r) for r in rows}) == 1:
            return ("matrix", f"{len(rows)}x{len(rows[0])}", [x for r in rows for x in r], "")
    if len(_split_top(s, "\\cup")) > 1:
        return ("union", "", _split_top(s, "\\cup"), "")
    outer = _outer(s)
    if outer and outer[0] == "{":
        return ("set", "{", _split_top(outer[1]), "}")
    if outer and len(_split_top(outer[1])) > 1:
        return ("ordered", outer[0], _split_top(outer[1]), outer[2])
    items = _split_top(s)
    if len(items) > 1:
        return ("list", "", items, "")
    if s.count("\\pm") == 1 and len(_split_top(s, "\\pm")) == 2:
        a, b = _split_top(s, "\\pm")
        return ("pm", "", [f"{a}+{b}", f"{a}-({b})"] if a else [b, f"-({b})"], "")
    return None


def _match_unordered(ps: List[str], gs: List[str]) -> bool:
    if len(ps) != len(gs):
        return False
    left = list(ps)
    for g in gs:
        hit = next((k for k, p in enumerate(left) if _equal_norm(p, g)), None)
        if hit is None:
            return False
        left.pop(hit)
    return True


def _match_structures(ps, gs, p: str, g: str) -> bool:
    if gs is None:  # gold scalar, pred structured: only a one-item structure can match
        return False
    pk = ps[0] if ps else None
    gk = gs[0]
    if gk in ("set", "list", "pm"):
        if ps is None:
            return False
        if pk in ("set", "list", "pm") or (pk == "ordered" and ps[1] == "(" and ps[3] == ")"):
            return _match_unordered(ps[2], gs[2])
        return False
    if gk == "union":
        return pk == "union" and _match_unordered(ps[2], gs[2])
    if gk == "ordered":
        if ps is None:
            return False
        if pk == "ordered" and (ps[1], ps[3]) == (gs[1], gs[3]):
            return len(ps[2]) == len(gs[2]) and all(_equal_norm(a, b) for a, b in zip(ps[2], gs[2]))
        if pk in ("list", "matrix") and (gs[1], gs[3]) == ("(", ")"):  # point written without parentheses / as a vector
            return len(ps[2]) == len(gs[2]) and all(_equal_norm(a, b) for a, b in zip(ps[2], gs[2]))
        return False
    if gk == "matrix":
        if pk == "matrix" and ps[1] == gs[1]:
            return all(_equal_norm(a, b) for a, b in zip(ps[2], gs[2]))
        if pk == "ordered" and gs[1].endswith("x1") and (ps[1], ps[3]) == ("(", ")"):
            return len(ps[2]) == len(gs[2]) and all(_equal_norm(a, b) for a, b in zip(ps[2], gs[2]))
        return False
    return False


_FUNCS = {"sqrt": sympy.sqrt, "root": sympy.root, "pi": sympy.pi, "oo": sympy.oo, "I": sympy.I,
          "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "log": sympy.log, "ln": sympy.log, "exp": sympy.exp}
_NAMES = {**{c: sympy.Symbol(c) for c in "abcdefghjklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}, **_FUNCS}

_REWRITES = [
    (r"\\infty", "oo"),
    (r"\\pi", "pi"),
    (r"\\(sin|cos|tan|log|ln|exp)(?![a-zA-Z])", r"\1"),
    (r"\\cdot|\\times", "*"),
    (r"\\div", "/"),
    (r"(?<![\d.^])(\d+)\\frac\{(\d+)\}\{(\d+)\}", r"(\1+\2/\3)"),  # mixed number 4\frac{4}{11}
    (r"\\frac(\d)(\d)", r"(\1/\2)"),
    (r"\\frac(\d)\{", r"\\frac{\1}{"),
    (r"\\sqrt\[([^\[\]]+)\]\{([^{}]*)\}", r" root(\2,\1)"),
    (r"\\sqrt(\d|[a-zA-Z])", r" sqrt(\1)"),
    (r"(?<![a-zA-Z])i(?![a-zA-Z])", "I"),  # imaginary unit
    (r"\^", "**"),
]


def to_sympy(s: str):
    """sympy value of a normalised scalar LaTeX answer, or None."""
    for pattern, repl in _REWRITES:
        s = re.sub(pattern, repl, s)
    while True:  # innermost groups first
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", r" sqrt(\1)", t)
        if t == s:
            t = re.sub(r"(?<!\\frac)(?<!\\sqrt)(?<!\})\{([^{}]*)\}", r"(\1)", s)
        if t == s:
            break
        s = t
    names = set(re.findall(r"[a-zA-Z]+", s))
    if (not re.fullmatch(r"[a-zA-Z0-9.+\-*/(), ]+", s) or len(s) > 200 or s.count("**") > 2
            or re.search(r"\*\*\s*\(*\s*\d{3}", s) or re.search(r"\d{40}", s)
            or not all(n in _NAMES or (len(n) <= 2 and n.isalpha() and n.lower() == n) for n in names)):
        return None
    try:
        with _time_limit(2.0):
            return parse_expr(s, local_dict=dict(_NAMES),
                              transformations=standard_transformations + (implicit_multiplication_application,))
    except BaseException as e:  # parse_expr raises many error types on malformed input; TimeoutError too
        if isinstance(e, KeyboardInterrupt):
            raise
        return None


def _text(s: str) -> Optional[str]:
    """Lower-case word answer ('even', 'Devon', '(C)' -> 'c'), or None if s is not plain text."""
    m = re.fullmatch(r"\(?([A-Z])\)?", s)
    if m:
        return m.group(1).lower()
    if re.fullmatch(r"[A-Za-z][A-Za-z',.\-]{2,}", s) and not re.search(r"\\", s):
        return re.sub(r"[^a-z]", "", s.lower())
    return None


def _scalar_equal(p: str, g: str) -> bool:
    base = re.fullmatch(r"([0-9A-Fa-f]+)_(\d+)", g)
    if base and p in (base.group(1), g):  # number bases: 1011_2 vs 1011
        return True
    tp, tg = _text(p), _text(g)
    if tp is not None or tg is not None:
        return tp == tg
    a, b = to_sympy(p), to_sympy(g)
    if a is None or b is None:
        return False
    try:
        with _time_limit(3.0):
            if a == b:
                return True
            if a.free_symbols or b.free_symbols:
                return sympy.simplify(a - b) == 0
            if a.has(sympy.oo, -sympy.oo, sympy.zoo) or b.has(sympy.oo, -sympy.oo, sympy.zoo):
                return False
            va, vb = complex(sympy.N(a, 30)), complex(sympy.N(b, 30))
            tol = 1e-4 if ("." in p or "." in g) else 1e-9
            return abs(va - vb) <= tol * max(abs(va), abs(vb)) + 1e-12
    except BaseException as e:  # TypeError, ValueError, TimeoutError, sympy errors
        if isinstance(e, KeyboardInterrupt):
            raise
        return False


def _equal_norm(p: str, g: str) -> bool:
    if p == g:
        return True
    p, g = _assignments(p, g)
    if p == g or (p.lower() == g.lower() and not re.search(r"\d", g) and len(g) > 1):
        return True
    ps, gs = structure(p), structure(g)
    if ps is not None or gs is not None:
        return _match_structures(ps, gs, p, g)
    return _scalar_equal(p, g)


def math_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return _equal_norm(normalize(pred.replace("\\approx", "=")), normalize(gold))
    except RecursionError:
        return False

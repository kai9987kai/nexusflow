from __future__ import annotations

import base64
import copy
import csv
import gc
import hashlib
import html as py_html
import http.server
import itertools
import json
import math
import os
import platform as py_platform
import random
import re
import shutil
import socket
import sqlite3
import struct
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import uuid
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # Native PyTorch path (optional at runtime)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)

try:  # pragma: no cover - platform dependent
    import resource as py_resource
except Exception:  # pragma: no cover - Windows typically lacks this
    py_resource = None  # type: ignore[assignment]


class NexusFlowError(Exception):
    pass


class ParseError(NexusFlowError):
    pass


class RuntimeErrorNF(NexusFlowError):
    pass


@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int


KEYWORDS = {
    "project",
    "config",
    "state",
    "metric",
    "channel",
    "agent",
    "count",
    "field",
    "on",
    "tick",
    "if",
    "else",
    "emit",
    "ui",
    "panel",
    "text",
    "button",
    "template",
    "stat",
    "progress",
    "scene3d",
    "pipeline",
    "step",
    "model",
    "dataset",
    "true",
    "false",
    "for",
    "in",
    "range",
    "while",
    "fn",
    "return",
    "watch",
    "null",
}


TOKEN_RE = re.compile(
    r"""
    (?P<COMMENT>\#.*)
  | (?P<NUMBER>\d+(?:\.\d+)?)
  | (?P<STRING>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
  | (?P<OP>\+=|-=|\*=|/=|==|!=|<=|>=|&&|\|\||[+\-*/%<>=!,;:{}()[\],])
  | (?P<ID>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<NEWLINE>\n)
  | (?P<SKIP>[ \t\r]+)
  | (?P<MISMATCH>.)
    """,
    re.VERBOSE,
)


def tokenize(source: str) -> List[Token]:
    line = 1
    col = 1
    pos = 0
    tokens: List[Token] = []
    while pos < len(source):
        m = TOKEN_RE.match(source, pos)
        if not m:
            raise ParseError(f"Tokenizer stalled at {line}:{col}")
        kind = m.lastgroup or "MISMATCH"
        value = m.group(0)
        if kind in {"SKIP", "COMMENT"}:
            pass
        elif kind == "NEWLINE":
            line += 1
            col = 0
        elif kind == "ID" and value in KEYWORDS:
            tokens.append(Token("KW", value, line, col))
        elif kind == "MISMATCH":
            raise ParseError(f"Unexpected character {value!r} at {line}:{col}")
        else:
            tokens.append(Token(kind, value, line, col))
        pos = m.end()
        col += len(value)
    tokens.append(Token("EOF", "", line, col))
    return tokens


@dataclass
class Expr:
    pass


@dataclass
class Literal(Expr):
    value: Any


@dataclass
class Var(Expr):
    name: str


@dataclass
class Unary(Expr):
    op: str
    expr: Expr


@dataclass
class Binary(Expr):
    left: Expr
    op: str
    right: Expr


@dataclass
class Call(Expr):
    name: str
    args: List[Expr]


@dataclass
class ListLiteral(Expr):
    items: List[Expr]


@dataclass
class DictLiteral(Expr):
    items: List[tuple[str, Expr]]


@dataclass
class Stmt:
    pass


@dataclass
class Assign(Stmt):
    name: str
    op: str
    expr: Expr


@dataclass
class If(Stmt):
    cond: Expr
    then_body: List[Stmt]
    else_body: List[Stmt] = field(default_factory=list)


@dataclass
class Emit(Stmt):
    expr: Expr


@dataclass
class ExprStmt(Stmt):
    expr: Expr


@dataclass
class ForLoop(Stmt):
    var_name: str
    start_expr: Expr
    end_expr: Expr
    body: List[Stmt]


@dataclass
class WhileLoop(Stmt):
    cond: Expr
    body: List[Stmt]


@dataclass
class ReturnStmt(Stmt):
    expr: Optional[Expr]


@dataclass
class PropertyDecl:
    name: str
    expr: Expr


@dataclass
class StateDecl:
    name: str
    expr: Expr


@dataclass
class MetricDecl:
    name: str
    expr: Expr


@dataclass
class AgentDecl:
    name: str
    count_expr: Expr
    fields: List[PropertyDecl]
    on_tick: List[Stmt]


@dataclass
class TextWidget:
    label: str
    expr: Expr


@dataclass
class ButtonWidget:
    label: str
    action: str


@dataclass
class StatWidget:
    label: str
    expr: Expr


@dataclass
class ProgressWidget:
    label: str
    expr: Expr


@dataclass
class JsonWidget:
    label: str
    expr: Expr


@dataclass
class Scene3DWidget:
    label: str
    expr: Expr


Widget = TextWidget | ButtonWidget | StatWidget | ProgressWidget | JsonWidget | Scene3DWidget


@dataclass
class PanelDecl:
    title: str
    widgets: List[Widget]


@dataclass
class UIDecl:
    panels: List[PanelDecl]
    templates: List[str] = field(default_factory=list)
    theme: Optional[str] = None


@dataclass
class PipelineDecl:
    name: str
    steps: List[Call]


@dataclass
class ModelDecl:
    name: str
    properties: List[PropertyDecl]


@dataclass
class DatasetDecl:
    name: str
    properties: List[PropertyDecl]


@dataclass
class FnDecl:
    name: str
    params: List[str]
    body: List[Stmt]


@dataclass
class WatchDecl:
    state_name: str
    body: List[Stmt]


@dataclass
class Project:
    name: str
    configs: List[PropertyDecl]
    states: List[StateDecl]
    metrics: List[MetricDecl]
    channels: List[str]
    agents: List[AgentDecl]
    models: List[ModelDecl]
    datasets: List[DatasetDecl]
    ui: Optional[UIDecl]
    pipelines: List[PipelineDecl]
    functions: List[FnDecl] = field(default_factory=list)
    watches: List[WatchDecl] = field(default_factory=list)


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        self.i = 0

    def cur(self) -> Token:
        return self.tokens[self.i]

    def peek(self, offset: int = 1) -> Token:
        j = min(self.i + offset, len(self.tokens) - 1)
        return self.tokens[j]

    def match(self, kind: str, value: Optional[str] = None) -> Optional[Token]:
        t = self.cur()
        if t.kind != kind:
            return None
        if value is not None and t.value != value:
            return None
        self.i += 1
        return t

    def match_kw(self, value: str) -> Optional[Token]:
        return self.match("KW", value)

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.cur()
        if t.kind != kind or (value is not None and t.value != value):
            want = f"{kind}:{value}" if value else kind
            raise ParseError(f"Expected {want} at {t.line}:{t.col}, got {t.kind}:{t.value!r}")
        self.i += 1
        return t

    def expect_kw(self, value: str) -> Token:
        return self.expect("KW", value)

    def parse_name(self) -> str:
        if self.cur().kind == "STRING":
            return self.parse_string()
        return self.expect("ID").value

    def parse_string(self) -> str:
        raw = self.expect("STRING").value
        return raw[1:-1].encode("utf-8").decode("unicode_escape")

    def parse(self) -> Project:
        self.expect_kw("project")
        name = self.parse_name()
        self.expect("OP", "{")

        configs: List[PropertyDecl] = []
        states: List[StateDecl] = []
        metrics: List[MetricDecl] = []
        channels: List[str] = []
        agents: List[AgentDecl] = []
        models: List[ModelDecl] = []
        datasets: List[DatasetDecl] = []
        ui: Optional[UIDecl] = None
        pipelines: List[PipelineDecl] = []
        functions: List[FnDecl] = []
        watches: List[WatchDecl] = []

        while not self.match("OP", "}"):
            if self.match_kw("config"):
                configs.append(self.parse_property_decl())
            elif self.match_kw("state"):
                name_tok = self.expect("ID")
                self.expect("OP", "=")
                expr = self.parse_expr()
                self.expect("OP", ";")
                states.append(StateDecl(name_tok.value, expr))
            elif self.match_kw("metric"):
                name_tok = self.expect("ID")
                self.expect("OP", "=")
                expr = self.parse_expr()
                self.expect("OP", ";")
                metrics.append(MetricDecl(name_tok.value, expr))
            elif self.match_kw("channel"):
                name_tok = self.expect("ID")
                self.expect("OP", ";")
                channels.append(name_tok.value)
            elif self.match_kw("agent"):
                agents.append(self.parse_agent())
            elif self.match_kw("model"):
                models.append(self.parse_named_property_block(ModelDecl))
            elif self.match_kw("dataset"):
                datasets.append(self.parse_named_property_block(DatasetDecl))
            elif self.match_kw("ui"):
                if ui is not None:
                    t = self.cur()
                    raise ParseError(f"Only one ui block allowed near {t.line}:{t.col}")
                ui = self.parse_ui()
            elif self.match_kw("pipeline"):
                pipelines.append(self.parse_pipeline())
            elif self.match_kw("fn"):
                functions.append(self.parse_fn())
            elif self.match_kw("watch"):
                watches.append(self.parse_watch())
            else:
                t = self.cur()
                raise ParseError(f"Unexpected token in project block at {t.line}:{t.col}: {t.value!r}")

        self.expect("EOF")
        return Project(name, configs, states, metrics, channels, agents, models, datasets, ui, pipelines, functions, watches)

    def parse_property_decl(self) -> PropertyDecl:
        name = self.expect("ID").value
        self.expect("OP", "=")
        expr = self.parse_expr()
        self.expect("OP", ";")
        return PropertyDecl(name, expr)

    def parse_named_property_block(self, cls):
        name = self.parse_name()
        self.expect("OP", "{")
        props: List[PropertyDecl] = []
        while not self.match("OP", "}"):
            props.append(self.parse_property_decl())
        return cls(name=name, properties=props)

    def parse_agent(self) -> AgentDecl:
        name = self.expect("ID").value
        self.expect_kw("count")
        count_expr = self.parse_expr()
        self.expect("OP", "{")
        fields: List[PropertyDecl] = []
        on_tick: List[Stmt] = []
        while not self.match("OP", "}"):
            if self.match_kw("field"):
                fields.append(self.parse_property_decl())
            elif self.match_kw("on"):
                self.expect_kw("tick")
                on_tick.extend(self.parse_stmt_block())
            else:
                t = self.cur()
                raise ParseError(f"Unexpected token in agent block at {t.line}:{t.col}: {t.value!r}")
        return AgentDecl(name, count_expr, fields, on_tick)

    def parse_ui(self) -> UIDecl:
        self.expect("OP", "{")
        panels: List[PanelDecl] = []
        templates: List[str] = []
        theme: Optional[str] = None
        while not self.match("OP", "}"):
            if self.match_kw("template"):
                templates.append(self.parse_name())
                self.expect("OP", ";")
                continue

            if self.cur().kind == "ID" and self.cur().value == "theme":
                self.expect("ID")
                self.expect("OP", "=")
                theme = self.parse_name()
                self.expect("OP", ";")
                continue

            self.expect_kw("panel")
            title = self.parse_name()
            self.expect("OP", "{")
            widgets: List[Widget] = []
            while not self.match("OP", "}"):
                if self.match_kw("text"):
                    label = self.parse_name()
                    self.expect("OP", "=")
                    expr = self.parse_expr()
                    self.expect("OP", ";")
                    widgets.append(TextWidget(label, expr))
                elif self.match_kw("button"):
                    label = self.parse_name()
                    self.expect("OP", "=")
                    action = self.parse_name()
                    self.expect("OP", ";")
                    widgets.append(ButtonWidget(label, action))
                elif self.match_kw("stat"):
                    label = self.parse_name()
                    self.expect("OP", "=")
                    expr = self.parse_expr()
                    self.expect("OP", ";")
                    widgets.append(StatWidget(label, expr))
                elif self.match_kw("progress"):
                    label = self.parse_name()
                    self.expect("OP", "=")
                    expr = self.parse_expr()
                    self.expect("OP", ";")
                    widgets.append(ProgressWidget(label, expr))
                elif self.cur().value == "json" and self.cur().kind in {"ID", "KW"}:
                    self.i += 1
                    label = self.parse_name()
                    self.expect("OP", "=")
                    expr = self.parse_expr()
                    self.expect("OP", ";")
                    widgets.append(JsonWidget(label, expr))
                elif self.match_kw("scene3d"):
                    label = self.parse_name()
                    self.expect("OP", "=")
                    expr = self.parse_expr()
                    self.expect("OP", ";")
                    widgets.append(Scene3DWidget(label, expr))
                else:
                    t = self.cur()
                    raise ParseError(f"Unexpected widget token at {t.line}:{t.col}: {t.value!r}")
            panels.append(PanelDecl(title, widgets))
        return UIDecl(panels=panels, templates=templates, theme=theme)

    def parse_pipeline(self) -> PipelineDecl:
        name = self.parse_name()
        self.expect("OP", "{")
        steps: List[Call] = []
        while not self.match("OP", "}"):
            self.expect_kw("step")
            expr = self.parse_expr()
            if not isinstance(expr, Call):
                t = self.cur()
                raise ParseError(f"Pipeline step must be a call near {t.line}:{t.col}")
            steps.append(expr)
            self.expect("OP", ";")
        return PipelineDecl(name, steps)

    def parse_fn(self) -> FnDecl:
        name = self.expect("ID").value
        self.expect("OP", "(")
        params: List[str] = []
        if not self.match("OP", ")"):
            while True:
                params.append(self.expect("ID").value)
                if self.match("OP", ")"):
                    break
                self.expect("OP", ",")
        body = self.parse_stmt_block()
        return FnDecl(name, params, body)

    def parse_watch(self) -> WatchDecl:
        state_name = self.expect("ID").value
        body = self.parse_stmt_block()
        return WatchDecl(state_name, body)

    def parse_stmt_block(self) -> List[Stmt]:
        self.expect("OP", "{")
        body: List[Stmt] = []
        while not self.match("OP", "}"):
            body.append(self.parse_stmt())
        return body

    def parse_stmt(self) -> Stmt:
        if self.match_kw("if"):
            cond = self.parse_expr()
            then_body = self.parse_stmt_block()
            else_body: List[Stmt] = []
            if self.match_kw("else"):
                else_body = self.parse_stmt_block()
            return If(cond, then_body, else_body)

        if self.match_kw("emit"):
            expr = self.parse_expr()
            self.expect("OP", ";")
            return Emit(expr)

        if self.match_kw("for"):
            var_name = self.expect("ID").value
            self.expect_kw("in")
            self.expect_kw("range")
            self.expect("OP", "(")
            start_expr = self.parse_expr()
            self.expect("OP", ",")
            end_expr = self.parse_expr()
            self.expect("OP", ")")
            body = self.parse_stmt_block()
            return ForLoop(var_name, start_expr, end_expr, body)

        if self.match_kw("while"):
            cond = self.parse_expr()
            body = self.parse_stmt_block()
            return WhileLoop(cond, body)

        if self.match_kw("return"):
            if self.cur().kind == "OP" and self.cur().value == ";":
                self.expect("OP", ";")
                return ReturnStmt(None)
            expr = self.parse_expr()
            self.expect("OP", ";")
            return ReturnStmt(expr)

        if self.cur().kind == "ID" and self.peek().kind == "OP" and self.peek().value in {"=", "+=", "-=", "*=", "/="}:
            name = self.expect("ID").value
            op = self.expect("OP").value
            expr = self.parse_expr()
            self.expect("OP", ";")
            return Assign(name, op, expr)

        expr = self.parse_expr()
        self.expect("OP", ";")
        return ExprStmt(expr)

    def parse_expr(self) -> Expr:
        return self.parse_or()

    def parse_or(self) -> Expr:
        expr = self.parse_and()
        while self.match("OP", "||"):
            expr = Binary(expr, "||", self.parse_and())
        return expr

    def parse_and(self) -> Expr:
        expr = self.parse_eq()
        while self.match("OP", "&&"):
            expr = Binary(expr, "&&", self.parse_eq())
        return expr

    def parse_eq(self) -> Expr:
        expr = self.parse_cmp()
        while self.cur().kind == "OP" and self.cur().value in {"==", "!="}:
            op = self.expect("OP").value
            expr = Binary(expr, op, self.parse_cmp())
        return expr

    def parse_cmp(self) -> Expr:
        expr = self.parse_term()
        while self.cur().kind == "OP" and self.cur().value in {"<", ">", "<=", ">="}:
            op = self.expect("OP").value
            expr = Binary(expr, op, self.parse_term())
        return expr

    def parse_term(self) -> Expr:
        expr = self.parse_factor()
        while self.cur().kind == "OP" and self.cur().value in {"+", "-"}:
            op = self.expect("OP").value
            expr = Binary(expr, op, self.parse_factor())
        return expr

    def parse_factor(self) -> Expr:
        expr = self.parse_unary()
        while self.cur().kind == "OP" and self.cur().value in {"*", "/", "%"}:
            op = self.expect("OP").value
            expr = Binary(expr, op, self.parse_unary())
        return expr

    def parse_unary(self) -> Expr:
        if self.cur().kind == "OP" and self.cur().value in {"-", "!"}:
            op = self.expect("OP").value
            return Unary(op, self.parse_unary())
        return self.parse_call()

    def parse_call(self) -> Expr:
        expr = self.parse_primary()
        while self.match("OP", "("):
            if not isinstance(expr, Var):
                t = self.cur()
                raise ParseError(f"Only named calls are supported near {t.line}:{t.col}")
            args: List[Expr] = []
            if not self.match("OP", ")"):
                while True:
                    args.append(self.parse_expr())
                    if self.match("OP", ")"):
                        break
                    self.expect("OP", ",")
            expr = Call(expr.name, args)
        return expr

    def parse_primary(self) -> Expr:
        t = self.cur()
        if self.match("NUMBER"):
            return Literal(float(t.value) if "." in t.value else int(t.value))
        if t.kind == "STRING":
            return Literal(self.parse_string())
        if self.match_kw("true"):
            return Literal(True)
        if self.match_kw("false"):
            return Literal(False)
        if self.match_kw("null"):
            return Literal(None)
        if self.match_kw("tick"):
            return Var("tick")
        if self.match("ID"):
            return Var(t.value)
        if self.match("OP", "("):
            expr = self.parse_expr()
            self.expect("OP", ")")
            return expr
        if self.match("OP", "["):
            items: List[Expr] = []
            if not self.match("OP", "]"):
                while True:
                    items.append(self.parse_expr())
                    if self.match("OP", "]"):
                        break
                    self.expect("OP", ",")
            return ListLiteral(items)
        if self.match("OP", "{"):
            items: List[tuple[str, Expr]] = []
            if not self.match("OP", "}"):
                while True:
                    if self.cur().kind == "STRING":
                        key = self.parse_string()
                    else:
                        key = self.expect("ID").value
                    self.expect("OP", ":")
                    value = self.parse_expr()
                    items.append((key, value))
                    if self.match("OP", "}"):
                        break
                    self.expect("OP", ",")
            return DictLiteral(items)
        raise ParseError(f"Unexpected token in expression at {t.line}:{t.col}: {t.kind}:{t.value!r}")


def parse_source(source: str) -> Project:
    return Parser(tokenize(source)).parse()


def parse_file(path: str | Path) -> Project:
    return parse_source(Path(path).read_text(encoding="utf-8"))


def expr_to_json(expr: Expr) -> Any:
    if isinstance(expr, Literal):
        return {"type": "literal", "value": expr.value}
    if isinstance(expr, Var):
        return {"type": "var", "name": expr.name}
    if isinstance(expr, Unary):
        return {"type": "unary", "op": expr.op, "expr": expr_to_json(expr.expr)}
    if isinstance(expr, Binary):
        return {"type": "binary", "op": expr.op, "left": expr_to_json(expr.left), "right": expr_to_json(expr.right)}
    if isinstance(expr, Call):
        return {"type": "call", "name": expr.name, "args": [expr_to_json(a) for a in expr.args]}
    if isinstance(expr, ListLiteral):
        return {"type": "list", "items": [expr_to_json(i) for i in expr.items]}
    if isinstance(expr, DictLiteral):
        return {"type": "dict", "items": [{"key": k, "value": expr_to_json(v)} for k, v in expr.items]}
    raise TypeError(expr)


def stmt_to_json(stmt: Stmt) -> Any:
    if isinstance(stmt, Assign):
        return {"type": "assign", "name": stmt.name, "op": stmt.op, "expr": expr_to_json(stmt.expr)}
    if isinstance(stmt, If):
        return {
            "type": "if",
            "cond": expr_to_json(stmt.cond),
            "then": [stmt_to_json(s) for s in stmt.then_body],
            "else": [stmt_to_json(s) for s in stmt.else_body],
        }
    if isinstance(stmt, Emit):
        return {"type": "emit", "expr": expr_to_json(stmt.expr)}
    if isinstance(stmt, ExprStmt):
        return {"type": "expr", "expr": expr_to_json(stmt.expr)}
    if isinstance(stmt, ForLoop):
        return {
            "type": "for",
            "var": stmt.var_name,
            "start": expr_to_json(stmt.start_expr),
            "end": expr_to_json(stmt.end_expr),
            "body": [stmt_to_json(s) for s in stmt.body],
        }
    if isinstance(stmt, WhileLoop):
        return {
            "type": "while",
            "cond": expr_to_json(stmt.cond),
            "body": [stmt_to_json(s) for s in stmt.body],
        }
    if isinstance(stmt, ReturnStmt):
        return {"type": "return", "expr": expr_to_json(stmt.expr) if stmt.expr else None}
    raise TypeError(stmt)


def project_to_json(project: Project) -> Dict[str, Any]:
    return {
        "project": project.name,
        "configs": [{p.name: expr_to_json(p.expr)} for p in project.configs],
        "states": [{s.name: expr_to_json(s.expr)} for s in project.states],
        "metrics": [{m.name: expr_to_json(m.expr)} for m in project.metrics],
        "channels": list(project.channels),
        "models": [{"name": m.name, "properties": [{p.name: expr_to_json(p.expr)} for p in m.properties]} for m in project.models],
        "datasets": [
            {"name": d.name, "properties": [{p.name: expr_to_json(p.expr)} for p in d.properties]} for d in project.datasets
        ],
        "agents": [
            {
                "name": a.name,
                "count": expr_to_json(a.count_expr),
                "fields": [{f.name: expr_to_json(f.expr)} for f in a.fields],
                "on_tick": [stmt_to_json(s) for s in a.on_tick],
            }
            for a in project.agents
        ],
        "ui": None
        if project.ui is None
        else {
            "templates": list(project.ui.templates),
            "theme": project.ui.theme,
            "panels": [
                {
                    "title": p.title,
                    "widgets": [
                        {"type": "text", "label": w.label, "expr": expr_to_json(w.expr)}
                        if isinstance(w, TextWidget)
                        else {"type": "button", "label": w.label, "action": w.action}
                        if isinstance(w, ButtonWidget)
                        else {"type": "stat", "label": w.label, "expr": expr_to_json(w.expr)}
                        if isinstance(w, StatWidget)
                        else {"type": "progress", "label": w.label, "expr": expr_to_json(w.expr)}
                        if isinstance(w, ProgressWidget)
                        else {"type": "json", "label": w.label, "expr": expr_to_json(w.expr)}
                        if isinstance(w, JsonWidget)
                        else {"type": "scene3d", "label": w.label, "expr": expr_to_json(w.expr)}
                        for w in p.widgets
                    ],
                }
                for p in project.ui.panels
            ]
        },
        "pipelines": [{"name": p.name, "steps": [expr_to_json(s) for s in p.steps]} for p in project.pipelines],
        "functions": [{"name": f.name, "params": f.params, "body": [stmt_to_json(s) for s in f.body]} for f in project.functions],
        "watches": [{"state": w.state_name, "body": [stmt_to_json(s) for s in w.body]} for w in project.watches],
    }


def _num(value: Any) -> float:
    if isinstance(value, (int, float)):
        return value
    raise RuntimeErrorNF(f"Expected number, got {value!r}")


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise RuntimeErrorNF(f"Expected integer-ish value, got {value!r}")


class _ReturnException(Exception):
    """Internal exception used to implement early return from user-defined functions."""
    def __init__(self, value: Any = None):
        super().__init__()
        self.value = value


class Executor:
    def __init__(self, project: Project, source_path: Optional[Path] = None, out_dir: Optional[Path] = None):
        self.project = project
        self.source_path = source_path
        self.base_dir = (source_path.parent if source_path else Path.cwd()).resolve()
        self.out_dir = (out_dir or self.base_dir).resolve()
        self.rand = random.Random()
        self._rng_local = threading.local()
        self._base_seed: Optional[int] = None
        self._lock = threading.RLock()
        self._metrics_stack = threading.local()
        self._query_cache: Dict[str, Dict[str, Any]] = {}

        self.config: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.metric_exprs: Dict[str, Expr] = {}
        self.agents: Dict[str, List[Dict[str, Any]]] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.channels: Dict[str, List[Any]] = {}
        self.assets3d: Dict[str, Dict[str, Any]] = {}
        self.scenes3d: Dict[str, Dict[str, Any]] = {}
        self.web_tools: Dict[str, Dict[str, Any]] = {}
        self.fusion_runs: Dict[str, Dict[str, Any]] = {}
        self.protein_runs: Dict[str, Dict[str, Any]] = {}
        self.events: List[Any] = []
        self.unsupported_steps: List[Dict[str, Any]] = []
        self.training_runs: List[Dict[str, Any]] = []
        self.benchmarks: List[Dict[str, Any]] = []
        self.windows_ops: List[Dict[str, Any]] = []
        self.vui_state: Dict[str, Any] = {
            "ops": [],
            "profiles": {},
            "voices_cache": None,
            "transcripts": [],
        }
        self.thread_stats: Dict[str, Any] = {
            "simulate_calls": 0,
            "simulate_mt_calls": 0,
            "pipeline_parallel_calls": 0,
            "max_workers_seen": 1,
            "thread_events": [],
        }
        self.torch_state: Dict[str, Any] = {
            "available": TORCH_AVAILABLE,
            "import_error": TORCH_IMPORT_ERROR,
            "trained_models": [],
            "exports": [],
        }
        self._torch_models: Dict[str, Any] = {}
        self._is_windows_host = os.name == "nt"
        self._pipeline_call_stack: List[str] = []
        self.tick_count = 0
        self.functions: Dict[str, FnDecl] = {}
        self._watches: List[WatchDecl] = []
        self._prev_watch_values: Dict[str, Any] = {}
        self.http_ops: List[Dict[str, Any]] = []
        self.http_auth_presets: Dict[str, Dict[str, str]] = {}
        self.mock_http_servers: Dict[str, Dict[str, Any]] = {}
        self._mock_http_server_handles: Dict[str, Dict[str, Any]] = {}
        self.lang_modules: Dict[str, Dict[str, Any]] = {}
        self.lang_module_runs: List[Dict[str, Any]] = []
        self._python_trained_models: Dict[str, Dict[str, Any]] = {}
        self.sqlite_state: Dict[str, Any] = {
            "ops": [],
            "databases": {},
        }
        self.archive_state: Dict[str, Any] = {
            "ops": [],
            "archives": {},
        }
        self.resource_state: Dict[str, Any] = {
            "ops": [],
            "limits": {
                "events_max": 500,
                "http_ops_max": 100,
                "windows_ops_max": 50,
                "vui_ops_max": 100,
                "sqlite_ops_max": 150,
                "archive_ops_max": 100,
                "lang_runs_max": 100,
                "thread_events_max": 50,
                "resource_ops_max": 120,
                "process_ops_max": 120,
                "wifi_ops_max": 80,
                "npu_probes_max": 30,
                "photo_ops_max": 80,
                "graph_ops_max": 120,
                "convert_ops_max": 120,
                "iso_ops_max": 40,
                "exe_ops_max": 80,
                "github_ops_max": 60,
                "max_sim_workers": None,
                "max_pipeline_workers": None,
            },
        }
        self.python_ml_state: Dict[str, Any] = {
            "trained_models": [],
            "exports": [],
        }
        self.process_state: Dict[str, Any] = {
            "ops": [],
            "profiles": {},
            "managed": {},
        }
        self._managed_process_handles: Dict[str, Any] = {}
        self.wifi_state: Dict[str, Any] = {
            "ops": [],
            "interfaces": [],
            "profiles": [],
            "last_scan": None,
        }
        self.npu_state: Dict[str, Any] = {
            "last_probe": None,
            "probes": [],
            "ops": [],
            "profiles": {},
            "plans": [],
            "runs": [],
            "benchmarks": [],
            "last_plan": None,
            "last_run": None,
            "last_benchmark": None,
        }
        self.photo_state: Dict[str, Any] = {
            "ops": [],
            "images": {},
            "batches": {},
        }
        self.graph_state: Dict[str, Any] = {
            "ops": [],
            "graphs": {},
        }
        self.convert_state: Dict[str, Any] = {
            "ops": [],
        }
        self.iso_state: Dict[str, Any] = {
            "ops": [],
            "last": None,
            "images": {},
            "tools": {},
        }
        self.exe_state: Dict[str, Any] = {
            "ops": [],
            "last": None,
            "artifacts": {},
            "tools": {},
        }
        self.github_local_state: Dict[str, Any] = {
            "ops": [],
            "last_report": None,
        }

        self._initialize()

    def _initialize(self) -> None:
        self.config = {p.name: self.eval_expr(p.expr, None) for p in self.project.configs}
        if "seed" in self.config and self.config["seed"] is not None:
            self._base_seed = _as_int(self.config["seed"])
            self.rand.seed(self._base_seed)

        self.state = {s.name: self.eval_expr(s.expr, None) for s in self.project.states}
        self.metric_exprs = {m.name: m.expr for m in self.project.metrics}
        self.models = {m.name: {p.name: self.eval_expr(p.expr, None) for p in m.properties} for m in self.project.models}
        self.datasets = {d.name: {p.name: self.eval_expr(p.expr, None) for p in d.properties} for d in self.project.datasets}
        self.channels = {name: [] for name in self.project.channels}

        for agent in self.project.agents:
            count = max(0, _as_int(self.eval_expr(agent.count_expr, None)))
            instances: List[Dict[str, Any]] = []
            for idx in range(count):
                inst: Dict[str, Any] = {"id": idx, "_type": agent.name}
                for f in agent.fields:
                    inst[f.name] = self.eval_expr(f.expr, inst)
                instances.append(inst)
            self.agents[agent.name] = instances

        # Register user-defined functions
        for fn in self.project.functions:
            self.functions[fn.name] = fn

        # Register watch blocks and snapshot initial values
        self._watches = list(self.project.watches)
        for w in self._watches:
            self._prev_watch_values[w.state_name] = copy.deepcopy(self.state.get(w.state_name))

    def _get_rng(self) -> random.Random:
        rng = getattr(self._rng_local, "rng", None)
        if rng is None:
            rng = random.Random()
            seed = self._base_seed if self._base_seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
            rng.seed(seed ^ threading.get_ident())
            self._rng_local.rng = rng
        return rng

    def _metric_stack(self) -> List[str]:
        stack = getattr(self._metrics_stack, "stack", None)
        if stack is None:
            stack = []
            self._metrics_stack.stack = stack
        return stack

    def _compute_metric(self, name: str, local: Optional[Dict[str, Any]]) -> Any:
        if name not in self.metric_exprs:
            raise RuntimeErrorNF(f"Unknown metric: {name}")
        stack = self._metric_stack()
        if name in stack:
            raise RuntimeErrorNF(f"Cyclic metric reference: {' -> '.join(stack + [name])}")
        stack.append(name)
        try:
            return self.eval_expr(self.metric_exprs[name], local)
        finally:
            stack.pop()

    def _append_event(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(payload)
            cap = self._history_cap("events_max", 500)
            self.events = self.events[-cap:]

    def _record_thread_event(self, name: str, **extra: Any) -> None:
        evt = {"kind": name, **extra}
        with self._lock:
            self.thread_stats["thread_events"].append(evt)
            cap = self._history_cap("thread_events_max", 50)
            self.thread_stats["thread_events"] = self.thread_stats["thread_events"][-cap:]

    def _channel_send(self, channel: str, value: Any) -> int:
        with self._lock:
            q = self.channels.setdefault(channel, [])
            q.append(value)
            return len(q)

    def _channel_recv(self, channel: str) -> Any:
        with self._lock:
            q = self.channels.setdefault(channel, [])
            if not q:
                return None
            return q.pop(0)

    def _channel_peek(self, channel: str) -> Any:
        with self._lock:
            q = self.channels.setdefault(channel, [])
            return q[0] if q else None

    def _channel_size(self, channel: str) -> int:
        with self._lock:
            return len(self.channels.setdefault(channel, []))

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _cache_get(self, key: str, ttl_ms: int) -> Any:
        now = time.time()
        with self._lock:
            rec = self._query_cache.get(str(key))
            if not isinstance(rec, dict):
                return None
            exp = rec.get("expires_at")
            if not isinstance(exp, (int, float)) or exp < now:
                self._query_cache.pop(str(key), None)
                return None
            return copy.deepcopy(rec.get("value"))

    def _cache_set(self, key: str, value: Any, ttl_ms: int) -> Any:
        expires = time.time() + max(1, int(ttl_ms)) / 1000.0
        with self._lock:
            self._query_cache[str(key)] = {"value": copy.deepcopy(value), "expires_at": expires}
            # keep cache bounded
            if len(self._query_cache) > 256:
                oldest = sorted(self._query_cache.items(), key=lambda kv: float((kv[1] or {}).get("expires_at", 0.0)))[:64]
                for k, _ in oldest:
                    self._query_cache.pop(k, None)
        return value

    def _cache_clear(self, prefix: Optional[str] = None) -> int:
        with self._lock:
            if not prefix:
                n = len(self._query_cache)
                self._query_cache.clear()
                return n
            keys = [k for k in self._query_cache.keys() if k.startswith(str(prefix))]
            for k in keys:
                self._query_cache.pop(k, None)
            return len(keys)

    def _host_info(self) -> Dict[str, Any]:
        return {
            "is_windows": self._is_windows_host,
            "platform": py_platform.platform(),
            "system": py_platform.system(),
            "release": py_platform.release(),
            "version": py_platform.version(),
            "hostname": socket.gethostname(),
            "cwd": str(self.base_dir),
        }

    def _record_windows_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **op}
        with self._lock:
            self.windows_ops.append(payload)
            cap = self._history_cap("windows_ops_max", 50)
            self.windows_ops = self.windows_ops[-cap:]

    def _record_vui_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **op}
        with self._lock:
            self.vui_state["ops"].append(payload)
            cap = self._history_cap("vui_ops_max", 100)
            self.vui_state["ops"] = self.vui_state["ops"][-cap:]
            if "transcript" in op:
                transcripts = self.vui_state.setdefault("transcripts", [])
                transcripts.append({"time": payload["time"], "role": op.get("role", "user"), "text": str(op.get("transcript", "")), "meta": copy.deepcopy(op.get("meta"))})
                transcripts[:] = transcripts[-cap:]

    def _record_http_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **op}
        with self._lock:
            self.http_ops.append(payload)
            cap = self._history_cap("http_ops_max", 100)
            self.http_ops = self.http_ops[-cap:]

    def _record_sqlite_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **op}
        db_key = str(op.get("db", ""))
        with self._lock:
            self.sqlite_state["ops"].append(payload)
            cap = self._history_cap("sqlite_ops_max", 150)
            self.sqlite_state["ops"] = self.sqlite_state["ops"][-cap:]
            dbs = self.sqlite_state.setdefault("databases", {})
            info = dbs.get(db_key, {"path": db_key, "ops": 0})
            info["ops"] = _as_int(info.get("ops", 0)) + 1
            info["last_op"] = payload.get("op")
            info["last_status"] = "ok" if payload.get("ok", True) else "error"
            if "rowcount" in payload:
                info["last_rowcount"] = payload.get("rowcount")
            if "rows" in payload:
                info["last_rows"] = payload.get("rows")
            if "sql" in payload:
                info["last_sql"] = str(payload.get("sql", ""))[:500]
            if db_key and db_key != ":memory:":
                p = Path(db_key)
                if p.exists():
                    try:
                        info["size_bytes"] = p.stat().st_size
                    except Exception:
                        pass
            dbs[db_key] = info

    def _record_archive_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **op}
        archive_key = str(op.get("archive", ""))
        with self._lock:
            self.archive_state["ops"].append(payload)
            cap = self._history_cap("archive_ops_max", 100)
            self.archive_state["ops"] = self.archive_state["ops"][-cap:]
            archives = self.archive_state.setdefault("archives", {})
            if archive_key:
                info = archives.get(archive_key, {"path": archive_key, "ops": 0})
                info["ops"] = _as_int(info.get("ops", 0)) + 1
                info["last_op"] = payload.get("op")
                info["last_status"] = "ok" if payload.get("ok", True) else "error"
                if "entries" in payload:
                    info["entries"] = payload.get("entries")
                if "bytes" in payload:
                    info["bytes"] = payload.get("bytes")
                archives[archive_key] = info

    def _record_iso_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        image_key = str(op.get("image", ""))
        with self._lock:
            self.iso_state["ops"].append(payload)
            cap = self._history_cap("iso_ops_max", 40)
            self.iso_state["ops"] = self.iso_state["ops"][-cap:]
            self.iso_state["last"] = copy.deepcopy(payload)
            images = self.iso_state.setdefault("images", {})
            if image_key:
                info = images.get(image_key, {"path": image_key, "ops": 0})
                info["ops"] = _as_int(info.get("ops", 0)) + 1
                info["last_op"] = payload.get("op")
                info["last_status"] = "ok" if payload.get("ok", True) else "error"
                for k in {
                    "label",
                    "source",
                    "tool",
                    "mode",
                    "dry_run",
                    "exists",
                    "bytes",
                    "size_bytes",
                    "manifest_preview",
                    "manifest_preview_truncated",
                    "manifest_count",
                    "source_file_count",
                    "source_dir_count",
                    "source_bytes",
                    "extract_dest",
                }:
                    if k in payload:
                        info[k] = copy.deepcopy(payload.get(k))
                if "entries" in payload:
                    entries_val = payload.get("entries")
                    if isinstance(entries_val, list):
                        info["entries"] = len(entries_val)
                        preview = entries_val[: min(50, len(entries_val))]
                        info["last_list_entries_preview"] = copy.deepcopy(preview)
                        info["last_list_entries_truncated"] = bool(len(entries_val) > len(preview) or payload.get("truncated"))
                    else:
                        info["entries"] = copy.deepcopy(entries_val)
                images[image_key] = info

    def _history_cap(self, key: str, default: int) -> int:
        with self._lock:
            limits = self.resource_state.get("limits", {})
            raw = limits.get(key, default) if isinstance(limits, dict) else default
        if raw is None:
            return max(1, int(default))
        try:
            return max(1, int(raw))
        except Exception:
            return max(1, int(default))

    def _record_resource_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        with self._lock:
            self.resource_state["ops"].append(payload)
            cap = self._history_cap("resource_ops_max", 120)
            self.resource_state["ops"] = self.resource_state["ops"][-cap:]

    def _process_memory_bytes(self) -> Optional[int]:
        if os.name == "nt":
            try:
                import ctypes  # local import to avoid platform issues
                from ctypes import wintypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
                GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                ok = GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb)
                if ok:
                    return int(counters.WorkingSetSize)
            except Exception:
                return None
            return None
        if py_resource is not None:
            try:
                rss = py_resource.getrusage(py_resource.RUSAGE_SELF).ru_maxrss  # type: ignore[attr-defined]
                if sys.platform == "darwin":
                    return int(rss)
                return int(rss) * 1024
            except Exception:
                return None
        return None

    def _dir_stats(self, rel_path: str, recursive: bool = True) -> Dict[str, Any]:
        p = self._resolve_runtime_path(rel_path)
        if not p.exists():
            return {"path": str(p), "exists": False, "files": 0, "dirs": 0, "bytes": 0}
        if p.is_file():
            size = p.stat().st_size
            return {"path": str(p), "exists": True, "is_file": True, "files": 1, "dirs": 0, "bytes": int(size)}
        files = 0
        dirs = 0
        total = 0
        it = p.rglob("*") if recursive else p.glob("*")
        for item in it:
            try:
                if item.is_dir():
                    dirs += 1
                elif item.is_file():
                    files += 1
                    total += int(item.stat().st_size)
            except Exception:
                continue
        return {"path": str(p), "exists": True, "is_file": False, "files": files, "dirs": dirs, "bytes": total}

    def _resource_host_snapshot(self, label: Optional[str] = None) -> Dict[str, Any]:
        proc_mem = self._process_memory_bytes()
        disk = None
        try:
            usage = shutil.disk_usage(self.out_dir)
            disk = {"total": int(usage.total), "used": int(usage.used), "free": int(usage.free)}
        except Exception:
            disk = None
        return {
            "time": self._now_iso(),
            "label": label,
            "host": self._host_info(),
            "cpu_count": os.cpu_count() or 1,
            "active_threads": threading.active_count(),
            "process_memory_bytes": proc_mem,
            "process_time_sec": round(time.process_time(), 6),
            "disk_out_dir": disk,
        }

    def _resource_runtime_snapshot(self, label: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            snap = {
                "time": self._now_iso(),
                "label": label,
                "tick": self.tick_count,
                "events": len(self.events),
                "http_ops": len(self.http_ops),
                "windows_ops": len(self.windows_ops),
                "sqlite_ops": len((self.sqlite_state or {}).get("ops", [])),
                "archive_ops": len((self.archive_state or {}).get("ops", [])),
                "iso_ops": len((self.iso_state or {}).get("ops", [])),
                "lang_runs": len(self.lang_module_runs),
                "thread_events": len((self.thread_stats or {}).get("thread_events", [])),
                "benchmarks": len(self.benchmarks),
                "training_runs": len(self.training_runs),
                "web_tools": len(self.web_tools),
                "assets3d": len(self.assets3d),
                "scenes3d": len(self.scenes3d),
                "resource_limits": copy.deepcopy(self.resource_state.get("limits", {})),
            }
        dir_out = self._dir_stats(".", recursive=False)
        snap["out_dir"] = dir_out
        return snap

    def _resource_snapshot(self, label: Optional[str] = None) -> Dict[str, Any]:
        host = self._resource_host_snapshot(label=label)
        runtime = self._resource_runtime_snapshot(label=label)
        payload = {"type": "resource_snapshot", "label": label, "host": host, "runtime": runtime}
        self._record_resource_op(payload)
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "resource_snapshot", "label": label or ""})
        return payload

    def _resource_set_limits(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(cfg, dict):
            raise RuntimeErrorNF("resource_set_limits expects a dict")
        allowed = {
            "events_max",
            "http_ops_max",
            "windows_ops_max",
            "vui_ops_max",
            "sqlite_ops_max",
            "archive_ops_max",
            "iso_ops_max",
            "lang_runs_max",
            "thread_events_max",
            "max_sim_workers",
            "max_pipeline_workers",
            "resource_ops_max",
        }
        applied: Dict[str, Any] = {}
        with self._lock:
            limits = self.resource_state.setdefault("limits", {})
            for k, v in cfg.items():
                key = str(k)
                if key not in allowed:
                    continue
                if v is None:
                    limits[key] = None
                    applied[key] = None
                else:
                    try:
                        limits[key] = max(1, int(v))
                        applied[key] = limits[key]
                    except Exception:
                        continue
            tick_now = self.tick_count
        self._record_resource_op({"type": "resource_set_limits", "applied": applied})
        self._append_event({"tick": tick_now, "event": "resource_set_limits", "keys": sorted(applied.keys())})
        return applied

    def _resource_trim(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        default_caps = {
            "events": self._history_cap("events_max", 500),
            "http_ops": self._history_cap("http_ops_max", 100),
            "windows_ops": self._history_cap("windows_ops_max", 50),
            "vui_ops": self._history_cap("vui_ops_max", 100),
            "sqlite_ops": self._history_cap("sqlite_ops_max", 150),
            "archive_ops": self._history_cap("archive_ops_max", 100),
            "iso_ops": self._history_cap("iso_ops_max", 40),
            "lang_runs": self._history_cap("lang_runs_max", 100),
            "thread_events": self._history_cap("thread_events_max", 50),
        }
        caps: Dict[str, int] = {}
        for k, default_v in default_caps.items():
            raw = cfg_map.get(k, default_v)
            try:
                caps[k] = max(1, int(raw))
            except Exception:
                caps[k] = default_v
        before: Dict[str, int]
        after: Dict[str, int]
        with self._lock:
            before = {
                "events": len(self.events),
                "http_ops": len(self.http_ops),
                "windows_ops": len(self.windows_ops),
                "vui_ops": len((self.vui_state or {}).get("ops", [])),
                "sqlite_ops": len((self.sqlite_state or {}).get("ops", [])),
                "archive_ops": len((self.archive_state or {}).get("ops", [])),
                "iso_ops": len((self.iso_state or {}).get("ops", [])),
                "lang_runs": len(self.lang_module_runs),
                "thread_events": len((self.thread_stats or {}).get("thread_events", [])),
            }
            self.events = self.events[-caps["events"] :]
            self.http_ops = self.http_ops[-caps["http_ops"] :]
            self.windows_ops = self.windows_ops[-caps["windows_ops"] :]
            self.vui_state["ops"] = self.vui_state.get("ops", [])[-caps["vui_ops"] :]
            self.vui_state["transcripts"] = self.vui_state.get("transcripts", [])[-caps["vui_ops"] :]
            self.sqlite_state["ops"] = self.sqlite_state.get("ops", [])[-caps["sqlite_ops"] :]
            self.archive_state["ops"] = self.archive_state.get("ops", [])[-caps["archive_ops"] :]
            self.iso_state["ops"] = self.iso_state.get("ops", [])[-caps["iso_ops"] :]
            self.lang_module_runs = self.lang_module_runs[-caps["lang_runs"] :]
            thread_ev = list((self.thread_stats or {}).get("thread_events", []))
            self.thread_stats["thread_events"] = thread_ev[-caps["thread_events"] :]
            after = {
                "events": len(self.events),
                "http_ops": len(self.http_ops),
                "windows_ops": len(self.windows_ops),
                "vui_ops": len((self.vui_state or {}).get("ops", [])),
                "sqlite_ops": len((self.sqlite_state or {}).get("ops", [])),
                "archive_ops": len((self.archive_state or {}).get("ops", [])),
                "iso_ops": len((self.iso_state or {}).get("ops", [])),
                "lang_runs": len(self.lang_module_runs),
                "thread_events": len((self.thread_stats or {}).get("thread_events", [])),
            }
            tick_now = self.tick_count
        result = {"before": before, "after": after, "caps": caps}
        self._record_resource_op({"type": "resource_trim", **result})
        self._append_event({"tick": tick_now, "event": "resource_trim", "before": before, "after": after})
        return result

    def _resource_gc(self, generation: Optional[int] = None) -> Dict[str, Any]:
        before_mem = self._process_memory_bytes()
        started = time.time()
        if generation is None:
            collected = gc.collect()
        else:
            collected = gc.collect(max(0, int(generation)))
        after_mem = self._process_memory_bytes()
        result = {
            "collected": int(collected),
            "generation": generation,
            "before_memory_bytes": before_mem,
            "after_memory_bytes": after_mem,
            "duration_ms": round((time.time() - started) * 1000, 2),
            "gc_counts": list(gc.get_count()),
        }
        self._record_resource_op({"type": "resource_gc", **result})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "resource_gc", "collected": int(collected)})
        return result

    def _export_resource_json(self, rel_path: str, limit: Optional[int] = None) -> Path:
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            ops = copy.deepcopy(self.resource_state.get("ops", []))
            limits = copy.deepcopy(self.resource_state.get("limits", {}))
        if limit is not None:
            ops = ops[-max(0, int(limit)) :]
        payload = {"ops": ops, "limits": limits, "latest": ops[-1] if ops else None}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "export_resource_json", "path": str(out)})
        return out

    def _resource_optimize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        if isinstance(cfg_map.get("limits"), dict):
            self._resource_set_limits(cfg_map["limits"])  # type: ignore[arg-type]
        trim_result = self._resource_trim(cfg_map.get("trim") if isinstance(cfg_map.get("trim"), dict) else None)
        gc_result = None
        if bool(cfg_map.get("gc", True)):
            gen = cfg_map.get("generation")
            gc_result = self._resource_gc(int(gen) if isinstance(gen, (int, float)) else None)
        snap = self._resource_snapshot(label=str(cfg_map.get("label", "optimize")))
        result = {"trim": trim_result, "gc": gc_result, "snapshot": snap}
        self._record_resource_op({"type": "resource_optimize", "summary": {"trim_after": trim_result.get("after", {})}})
        return result

    def _set_http_auth_preset(self, name: str, headers: Dict[str, Any]) -> Dict[str, str]:
        preset = {str(k): str(v) for k, v in headers.items()}
        with self._lock:
            self.http_auth_presets[name] = preset
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "http_auth_preset_set", "name": name, "headers": sorted(preset.keys())})
        return preset

    def _clear_http_auth_preset(self, name: str) -> bool:
        with self._lock:
            existed = name in self.http_auth_presets
            if existed:
                del self.http_auth_presets[name]
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "http_auth_preset_clear", "name": name, "existed": existed})
        return existed

    def _resolve_http_headers(self, headers: Optional[Dict[str, Any]] = None, auth_preset: Optional[str] = None) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        if auth_preset:
            with self._lock:
                preset = copy.deepcopy(self.http_auth_presets.get(auth_preset))
            if preset is None:
                raise RuntimeErrorNF(f"Unknown HTTP auth preset: {auth_preset}")
            merged.update({str(k): str(v) for k, v in preset.items()})
        if isinstance(headers, dict):
            merged.update({str(k): str(v) for k, v in headers.items()})
        return merged

    def _normalize_mock_routes(self, routes: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        if isinstance(routes, dict):
            items = routes.items()
        else:
            items = []
        for raw_path, raw_spec in items:
            path = str(raw_path or "/")
            if not path.startswith("/"):
                path = "/" + path
            if isinstance(raw_spec, dict):
                spec = copy.deepcopy(raw_spec)
            elif isinstance(raw_spec, (list, bool, int, float)) or raw_spec is None:
                spec = {"json": raw_spec}
            else:
                spec = {"body": str(raw_spec)}
            if "status" in spec:
                try:
                    spec["status"] = int(spec["status"])
                except Exception:
                    spec["status"] = 200
            methods = spec.get("methods", spec.get("method"))
            if methods is not None:
                if isinstance(methods, str):
                    spec["methods"] = [methods.upper()]
                elif isinstance(methods, list):
                    spec["methods"] = [str(m).upper() for m in methods]
                else:
                    spec["methods"] = [str(methods).upper()]
            if isinstance(spec.get("headers"), dict):
                spec["headers"] = {str(k): str(v) for k, v in (spec.get("headers") or {}).items()}
            normalized[path] = spec
        if "/health" not in normalized:
            normalized["/health"] = {"json": {"ok": True, "service": "nexusflow_mock"}}
        if "/echo" not in normalized:
            normalized["/echo"] = {"echo": True}
        if "/routes" not in normalized:
            normalized["/routes"] = {"list_routes": True}
        return normalized

    def _record_mock_http_request(self, server_name: str, item: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **item}
        with self._lock:
            meta = self.mock_http_servers.get(server_name)
            if not isinstance(meta, dict):
                return
            reqs = meta.setdefault("requests", [])
            if isinstance(reqs, list):
                reqs.append(payload)
                meta["requests"] = reqs[-200:]
                meta["request_count"] = len(reqs)
                meta["last_request"] = copy.deepcopy(payload)

    def _mock_http_server_start(
        self,
        name: str,
        port: int = 0,
        routes: Optional[Dict[str, Any]] = None,
        *,
        host: str = "127.0.0.1",
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        server_name = str(name)
        settings: Dict[str, Any] = {}
        if isinstance(cfg, dict):
            settings.update(cfg)
        host = str(settings.get("host", host) or host)
        port = max(0, min(65535, int(port)))
        allow_cors = bool(settings.get("allow_cors", True))
        default_headers = settings.get("headers") if isinstance(settings.get("headers"), dict) else {}
        default_headers = {str(k): str(v) for k, v in (default_headers or {}).items()}
        route_specs = self._normalize_mock_routes(routes)
        for spec in route_specs.values():
            headers = spec.get("headers") if isinstance(spec.get("headers"), dict) else {}
            merged_headers = {**default_headers, **{str(k): str(v) for k, v in (headers or {}).items()}}
            if merged_headers:
                spec["headers"] = merged_headers

        with self._lock:
            existing_handle = self._mock_http_server_handles.get(server_name)
        if existing_handle:
            self._mock_http_server_stop(server_name)

        executor = self

        class _NFMockServer(http.server.ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        class _NFMockHandler(http.server.BaseHTTPRequestHandler):
            server_version = "NexusFlowMock/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
                return

            def _send_json(self, status: int, payload: Any, extra_headers: Optional[Dict[str, str]] = None) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                if allow_cors:
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Headers", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
                if isinstance(extra_headers, dict):
                    for k, v in extra_headers.items():
                        self.send_header(str(k), str(v))
                self.end_headers()
                self.wfile.write(raw)

            def _send_body(
                self,
                status: int,
                body: bytes,
                *,
                content_type: str = "text/plain; charset=utf-8",
                extra_headers: Optional[Dict[str, str]] = None,
            ) -> None:
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if allow_cors:
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Headers", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
                if isinstance(extra_headers, dict):
                    for k, v in extra_headers.items():
                        self.send_header(str(k), str(v))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _read_body(self) -> bytes:
                try:
                    n = int(self.headers.get("Content-Length", "0") or "0")
                except Exception:
                    n = 0
                return self.rfile.read(max(0, n)) if n > 0 else b""

            def _parse_body(self, body: bytes) -> Dict[str, Any]:
                if not body:
                    return {"text": "", "json": None}
                text = body.decode("utf-8", errors="replace")
                ctype = (self.headers.get("Content-Type") or "").lower()
                body_json = None
                if "json" in ctype:
                    try:
                        body_json = json.loads(text)
                    except Exception:
                        body_json = None
                return {"text": text, "json": body_json}

            def _flatten_query(self, q: Dict[str, List[str]]) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for k, v in q.items():
                    out[k] = v[0] if len(v) == 1 else v
                return out

            def _handle_req(self) -> None:
                method = str(self.command or "GET").upper()
                parsed = urllib.parse.urlsplit(self.path or "/")
                path = parsed.path or "/"
                query_map = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                query = self._flatten_query(query_map)
                body = self._read_body() if method in {"POST", "PUT", "PATCH"} else b""
                body_info = self._parse_body(body)
                headers_map = {str(k): str(v) for k, v in self.headers.items()}
                with executor._lock:
                    meta = executor.mock_http_servers.get(server_name)
                    tick_now = executor.tick_count
                    route_map = copy.deepcopy((meta or {}).get("routes", {})) if isinstance(meta, dict) else {}
                    running = bool((meta or {}).get("running")) if isinstance(meta, dict) else False
                if not running:
                    self._send_json(503, {"ok": False, "error": "server_stopping"})
                    return

                spec = route_map.get(path)
                if spec is None and path.endswith("/") and path[:-1] in route_map:
                    spec = route_map.get(path[:-1])
                if spec is None and (path + "/") in route_map:
                    spec = route_map.get(path + "/")

                response_status = 200
                response_headers: Dict[str, str] = {}
                response_payload: Any = None
                response_body: Optional[bytes] = None
                content_type = "application/json; charset=utf-8"

                if path == "/__nexusflow__/status":
                    response_payload = {
                        "ok": True,
                        "name": server_name,
                        "path": path,
                        "routes": sorted([p for p in route_map.keys() if not p.startswith("/__nexusflow__/")]),
                    }
                elif isinstance(spec, dict):
                    allowed_methods = spec.get("methods")
                    if isinstance(allowed_methods, list) and allowed_methods and method not in allowed_methods:
                        response_status = 405
                        response_payload = {"ok": False, "error": "method_not_allowed", "allowed": allowed_methods}
                    else:
                        if spec.get("delay_ms") is not None:
                            try:
                                time.sleep(max(0.0, float(spec.get("delay_ms")) / 1000.0))
                            except Exception:
                                pass
                        response_status = int(spec.get("status", 200))
                        if isinstance(spec.get("headers"), dict):
                            response_headers.update({str(k): str(v) for k, v in (spec.get("headers") or {}).items()})
                        if bool(spec.get("echo")):
                            response_payload = {
                                "ok": True,
                                "echo": True,
                                "method": method,
                                "path": path,
                                "query": query,
                                "headers": headers_map,
                                "body_text": body_info["text"],
                                "body_json": body_info["json"],
                            }
                        elif bool(spec.get("list_routes")):
                            response_payload = {
                                "ok": True,
                                "routes": sorted([p for p in route_map.keys() if not p.startswith("/__nexusflow__/")]),
                                "count": len([p for p in route_map.keys() if not p.startswith("/__nexusflow__/")]),
                            }
                        elif "json" in spec:
                            response_payload = spec.get("json")
                        else:
                            body_value = spec.get("body", spec.get("text", ""))
                            response_body = self._safe_bytes(body_value)
                            content_type = str(spec.get("content_type", "text/plain; charset=utf-8"))
                else:
                    response_status = 404
                    response_payload = {"ok": False, "error": "not_found", "path": path}

                if response_body is None:
                    self._send_json(response_status, response_payload, response_headers)
                else:
                    self._send_body(response_status, response_body, content_type=content_type, extra_headers=response_headers)

                executor._record_mock_http_request(
                    server_name,
                    {
                        "tick": tick_now,
                        "method": method,
                        "path": path,
                        "query": query,
                        "status": response_status,
                        "request_bytes": len(body),
                        "response_bytes": len(response_body) if response_body is not None else len(json.dumps(response_payload, ensure_ascii=False).encode("utf-8")),
                    },
                )

            def _safe_bytes(self, value: Any) -> bytes:
                if isinstance(value, bytes):
                    return value
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False).encode("utf-8")
                return str(value).encode("utf-8", errors="replace")

            def do_GET(self) -> None:  # noqa: N802
                self._handle_req()

            def do_POST(self) -> None:  # noqa: N802
                self._handle_req()

            def do_PUT(self) -> None:  # noqa: N802
                self._handle_req()

            def do_PATCH(self) -> None:  # noqa: N802
                self._handle_req()

            def do_DELETE(self) -> None:  # noqa: N802
                self._handle_req()

            def do_OPTIONS(self) -> None:  # noqa: N802
                self._send_body(204, b"", content_type="text/plain; charset=utf-8")

        try:
            server = _NFMockServer((host, port), _NFMockHandler)
        except OSError as exc:
            raise RuntimeErrorNF(f"mock_http_server_start failed on {host}:{port}: {exc}") from exc

        actual_host = server.server_address[0]
        actual_port = int(server.server_address[1])
        base_url = f"http://{actual_host}:{actual_port}"
        meta = {
            "name": server_name,
            "host": actual_host,
            "port": actual_port,
            "base_url": base_url,
            "running": True,
            "routes": route_specs,
            "request_count": 0,
            "requests": [],
            "started_at": self._now_iso(),
            "config": {"allow_cors": allow_cors, "host": host},
        }
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, name=f"nf-mock-http-{server_name}", daemon=True)
        with self._lock:
            self.mock_http_servers[server_name] = meta
            self._mock_http_server_handles[server_name] = {"server": server, "thread": thread}
            tick_now = self.tick_count
        thread.start()
        self._append_event({"tick": tick_now, "event": "mock_http_server_start", "name": server_name, "port": actual_port})
        return copy.deepcopy(meta)

    def _mock_http_server_stop(self, name: str, timeout_sec: float = 2.0) -> Dict[str, Any]:
        server_name = str(name)
        with self._lock:
            handle = self._mock_http_server_handles.pop(server_name, None)
            meta = self.mock_http_servers.get(server_name)
            tick_now = self.tick_count
        if not handle:
            existed = isinstance(meta, dict)
            if existed:
                with self._lock:
                    self.mock_http_servers[server_name]["running"] = False
                    self.mock_http_servers[server_name]["stopped_at"] = self._now_iso()
            result = {"ok": False, "name": server_name, "existed": existed, "reason": "not_running"}
            self._append_event({"tick": tick_now, "event": "mock_http_server_stop", **result})
            return result

        server = handle.get("server")
        thread = handle.get("thread")
        err: Optional[str] = None
        try:
            if server is not None:
                server.shutdown()
                server.server_close()
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        if isinstance(thread, threading.Thread):
            thread.join(timeout=max(0.1, float(timeout_sec)))
        with self._lock:
            if server_name in self.mock_http_servers:
                self.mock_http_servers[server_name]["running"] = False
                self.mock_http_servers[server_name]["stopped_at"] = self._now_iso()
        result = {"ok": err is None, "name": server_name, "error": err}
        self._append_event({"tick": tick_now, "event": "mock_http_server_stop", **result})
        return result

    def _run_subprocess(self, cmd: List[str], *, stdin_text: Optional[str] = None, timeout_sec: float = 30.0) -> Dict[str, Any]:
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            result = {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "ok": False,
                "returncode": None,
                "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                "timeout": True,
            }
        result["duration_ms"] = round((time.time() - started) * 1000, 2)
        return result

    def _resolve_module_path(self, path_value: str) -> Path:
        p = Path(path_value)
        if p.is_absolute():
            return p.resolve()
        base_candidate = (self.base_dir / p).resolve()
        if base_candidate.exists():
            return base_candidate
        return (self.out_dir / p).resolve()

    def _record_lang_module_run(self, rec: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(rec)}
        with self._lock:
            self.lang_module_runs.append(payload)
            cap = self._history_cap("lang_runs_max", 100)
            self.lang_module_runs = self.lang_module_runs[-cap:]

    def _register_lang_module(self, name: str, language: str, path_value: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mod_name = str(name)
        lang = str(language or "python").lower().strip()
        src_path = self._resolve_module_path(str(path_value))
        config = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        meta: Dict[str, Any] = {
            "name": mod_name,
            "language": lang,
            "path": str(src_path),
            "exists": src_path.exists(),
            "config": config,
            "registered_at": self._now_iso(),
        }
        if lang in {"cpp", "cxx", "c++"}:
            meta["language"] = "cpp"
            meta["kind"] = "cpp_source"
            if str(src_path).lower().endswith((".exe", ".out", ".bin")):
                meta["kind"] = "cpp_binary"
                meta["binary_path"] = str(src_path)
        elif lang in {"py", "python"}:
            meta["language"] = "python"
            meta["kind"] = "script"
            meta["runtime"] = str(config.get("runtime", sys.executable or "python"))
        elif lang in {"js", "javascript", "node"}:
            meta["language"] = "javascript"
            meta["kind"] = "script"
            meta["runtime"] = str(config.get("runtime", "node"))
        elif lang in {"cs", "csharp", "c#"}:
            meta["language"] = "csharp"
            lower_path = str(src_path).lower()
            if lower_path.endswith(".csproj"):
                meta["kind"] = "csharp_project"
            elif lower_path.endswith(".dll"):
                meta["kind"] = "csharp_assembly"
                meta["binary_path"] = str(src_path)
            elif lower_path.endswith(".exe"):
                meta["kind"] = "csharp_binary"
                meta["binary_path"] = str(src_path)
            else:
                meta["kind"] = "csharp_source"
        elif lang in {"rs", "rust"}:
            meta["language"] = "rust"
            lower_path = str(src_path).lower()
            if src_path.is_dir() or lower_path.endswith("cargo.toml"):
                meta["kind"] = "rust_project"
            elif lower_path.endswith((".exe", ".out", ".bin")):
                meta["kind"] = "rust_binary"
                meta["binary_path"] = str(src_path)
            else:
                meta["kind"] = "rust_source"
        elif lang in {"ps1", "powershell"}:
            meta["language"] = "powershell"
            meta["kind"] = "script"
        else:
            meta["kind"] = "script"
        with self._lock:
            prev = self.lang_modules.get(mod_name, {})
            if isinstance(prev, dict):
                if "run_count" in prev:
                    meta["run_count"] = prev.get("run_count", 0)
                if "last_run" in prev:
                    meta["last_run"] = copy.deepcopy(prev.get("last_run"))
                if "build" in prev:
                    meta["build"] = copy.deepcopy(prev.get("build"))
                if "binary_path" in prev and "binary_path" not in meta:
                    meta["binary_path"] = prev.get("binary_path")
            self.lang_modules[mod_name] = meta
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "lang_module_register", "name": mod_name, "language": meta["language"]})
        return copy.deepcopy(meta)

    def _cpp_compiler_info(self) -> Optional[Dict[str, Any]]:
        for cmd in ("g++", "clang++", "c++"):
            path = shutil.which(cmd)
            if path:
                return {"flavor": "gcc_like", "command": cmd, "path": path}
        cl_path = shutil.which("cl")
        if cl_path:
            return {"flavor": "msvc", "command": "cl", "path": cl_path}
        return None

    def _csharp_compiler_info(self) -> Optional[Dict[str, Any]]:
        dotnet_path = shutil.which("dotnet")
        if dotnet_path:
            return {"flavor": "dotnet", "command": "dotnet", "path": dotnet_path}
        for cmd in ("csc", "mcs"):
            path = shutil.which(cmd)
            if path:
                return {"flavor": "compiler", "command": cmd, "path": path}
        return None

    def _rust_toolchain_info(self) -> Optional[Dict[str, Any]]:
        cargo_path = shutil.which("cargo")
        rustc_path = shutil.which("rustc")
        if cargo_path or rustc_path:
            return {
                "flavor": "cargo" if cargo_path else "rustc",
                "cargo": cargo_path,
                "rustc": rustc_path,
                "path": cargo_path or rustc_path,
            }
        return None

    def _rust_build_module(self, name: str, out_path: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mod_name = str(name)
        with self._lock:
            meta = copy.deepcopy(self.lang_modules.get(mod_name))
            tick_now = self.tick_count
        if not isinstance(meta, dict):
            raise RuntimeErrorNF(f"Unknown language module: {mod_name}")
        if meta.get("language") != "rust":
            raise RuntimeErrorNF(f"Module {mod_name} is not a rust module")
        src = Path(str(meta.get("path", "")))
        if not src.exists():
            result = {"ok": False, "name": mod_name, "reason": "source_missing", "path": str(src)}
            with self._lock:
                self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
            self._append_event({"tick": tick_now, "event": "rust_build", **result})
            return result

        tool = self._rust_toolchain_info()
        if tool is None:
            result = {"ok": False, "name": mod_name, "reason": "toolchain_missing"}
            with self._lock:
                self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
            self._append_event({"tick": tick_now, "event": "rust_build", **result})
            return result

        cfg_map: Dict[str, Any] = {}
        if isinstance(meta.get("config"), dict):
            cfg_map.update(copy.deepcopy(meta.get("config")))  # type: ignore[arg-type]
        if isinstance(cfg, dict):
            cfg_map.update(copy.deepcopy(cfg))
        timeout_sec = float(cfg_map.get("timeout_sec", 120.0))
        kind = str(meta.get("kind", "rust_source"))
        command_used: Optional[List[str]] = None
        binary_path: Optional[str] = None

        if kind == "rust_project":
            cargo_path = tool.get("cargo")
            if not cargo_path:
                result = {"ok": False, "name": mod_name, "reason": "cargo_missing"}
                with self._lock:
                    self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
                self._append_event({"tick": tick_now, "event": "rust_build", **result})
                return result
            manifest_path = src if src.is_file() else (src / "Cargo.toml")
            if not manifest_path.exists():
                result = {"ok": False, "name": mod_name, "reason": "cargo_manifest_missing", "path": str(manifest_path)}
                with self._lock:
                    self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
                self._append_event({"tick": tick_now, "event": "rust_build", **result})
                return result
            cmd = [str(cargo_path), "build", "--manifest-path", str(manifest_path)]
            if bool(cfg_map.get("release", True)):
                cmd.append("--release")
            extra_args = cfg_map.get("cargo_args")
            if isinstance(extra_args, list):
                cmd.extend(str(x) for x in extra_args)
            command_used = cmd
            build_result = self._run_subprocess(cmd, timeout_sec=timeout_sec)
            if build_result.get("ok"):
                target_dir = manifest_path.parent / "target" / ("release" if bool(cfg_map.get("release", True)) else "debug")
                bin_name = str(cfg_map.get("bin_name", src.name if src.is_dir() else manifest_path.parent.name))
                ext = ".exe" if os.name == "nt" else ""
                candidate = target_dir / (bin_name + ext)
                if candidate.exists():
                    binary_path = str(candidate)
                else:
                    candidates = sorted(target_dir.glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                    for c in candidates:
                        if c.is_file() and (os.name == "nt" and c.suffix == ".exe" or os.name != "nt"):
                            binary_path = str(c)
                            break
        else:
            rustc_path = tool.get("rustc")
            if not rustc_path:
                result = {"ok": False, "name": mod_name, "reason": "rustc_missing"}
                with self._lock:
                    self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
                self._append_event({"tick": tick_now, "event": "rust_build", **result})
                return result
            if out_path:
                out = self._resolve_output_path(out_path)
            else:
                ext = ".exe" if os.name == "nt" else ""
                out = self._resolve_output_path(f"rust_modules/{mod_name}{ext}")
            flags = [str(x) for x in (cfg_map.get("flags") or [])] if isinstance(cfg_map.get("flags"), list) else []
            cmd = [str(rustc_path), str(src), "-o", str(out)]
            if bool(cfg_map.get("optimize", True)):
                cmd.extend(["-C", "opt-level=2"])
            cmd.extend(flags)
            command_used = cmd
            build_result = self._run_subprocess(cmd, timeout_sec=timeout_sec)
            if out.exists():
                binary_path = str(out)

        result2 = {
            "ok": bool(build_result.get("ok")) and (binary_path is not None and Path(binary_path).exists()),
            "name": mod_name,
            "toolchain": tool,
            "path": str(src),
            "binary_path": binary_path,
            "command": command_used,
            **build_result,
        }
        with self._lock:
            self.lang_modules[mod_name]["build"] = copy.deepcopy(result2)
            if result2["ok"] and binary_path:
                self.lang_modules[mod_name]["binary_path"] = str(binary_path)
                self.lang_modules[mod_name]["kind"] = "rust_binary"
                self.lang_modules[mod_name]["exists"] = src.exists()
        self._append_event({"tick": tick_now, "event": "rust_build", "name": mod_name, "ok": bool(result2.get("ok"))})
        return result2

    def _csharp_build_module(self, name: str, out_path: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mod_name = str(name)
        with self._lock:
            meta = copy.deepcopy(self.lang_modules.get(mod_name))
            tick_now = self.tick_count
        if not isinstance(meta, dict):
            raise RuntimeErrorNF(f"Unknown language module: {mod_name}")
        if meta.get("language") != "csharp":
            raise RuntimeErrorNF(f"Module {mod_name} is not a csharp module")
        src = Path(str(meta.get("path", "")))
        if not src.exists():
            result = {"ok": False, "name": mod_name, "reason": "source_missing", "path": str(src)}
            with self._lock:
                self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
            self._append_event({"tick": tick_now, "event": "csharp_build", **result})
            return result

        tool = self._csharp_compiler_info()
        if tool is None:
            result = {"ok": False, "name": mod_name, "reason": "compiler_missing"}
            with self._lock:
                self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
            self._append_event({"tick": tick_now, "event": "csharp_build", **result})
            return result

        cfg_map: Dict[str, Any] = {}
        if isinstance(meta.get("config"), dict):
            cfg_map.update(copy.deepcopy(meta.get("config")))  # type: ignore[arg-type]
        if isinstance(cfg, dict):
            cfg_map.update(copy.deepcopy(cfg))

        timeout_sec = float(cfg_map.get("timeout_sec", 90.0))
        kind = str(meta.get("kind", "csharp_source"))
        build_result: Dict[str, Any]
        binary_path: Optional[str] = None
        command_used: Optional[List[str]] = None

        if kind == "csharp_project" and tool["flavor"] == "dotnet":
            configuration = str(cfg_map.get("configuration", "Release"))
            framework = str(cfg_map.get("framework", "")).strip()
            cmd = [str(tool["path"]), "build", str(src), "-c", configuration, "--nologo"]
            if framework:
                cmd.extend(["-f", framework])
            if out_path:
                out_dir = self._resolve_output_path(str(out_path))
                if out_dir.suffix:
                    out_dir = out_dir.parent
                cmd.extend(["-o", str(out_dir)])
            command_used = cmd
            build_result = self._run_subprocess(cmd, timeout_sec=timeout_sec)
            # Discover binary outputs (prefer dll for dotnet run)
            search_root = (self._resolve_output_path(str(out_path)).parent if out_path and Path(str(out_path)).suffix else self.out_dir)
            if build_result.get("ok"):
                candidates = sorted(search_root.rglob("*.dll"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                if not candidates:
                    candidates = sorted(search_root.rglob("*.exe"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                if candidates:
                    binary_path = str(candidates[0])
        else:
            # Single-file compilation via csc/mcs (or dotnet SDK if csc exists on PATH)
            ext = ".exe" if os.name == "nt" else ".exe"
            if out_path:
                out = self._resolve_output_path(str(out_path))
                if not out.suffix:
                    out = out.with_suffix(ext)
            else:
                out = self._resolve_output_path(f"csharp_modules/{mod_name}{ext}")
            extra_flags = [str(x) for x in (cfg_map.get("flags") or [])] if isinstance(cfg_map.get("flags"), list) else []
            cmd = [str(tool["path"])]
            if tool["command"] == "csc":
                cmd.extend(["/nologo", f"/out:{str(out)}", str(src)])
                cmd.extend(extra_flags)
            elif tool["command"] == "mcs":
                cmd.extend(["-out:" + str(out), str(src)])
                cmd.extend(extra_flags)
            elif tool["command"] == "dotnet":
                # No direct single-file compile path via dotnet CLI without a project.
                result = {"ok": False, "name": mod_name, "reason": "dotnet_requires_csproj", "path": str(src)}
                with self._lock:
                    self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
                self._append_event({"tick": tick_now, "event": "csharp_build", **result})
                return result
            command_used = cmd
            build_result = self._run_subprocess(cmd, timeout_sec=timeout_sec)
            if out.exists():
                binary_path = str(out)

        result = {
            "ok": bool(build_result.get("ok")) and (binary_path is None or Path(binary_path).exists()),
            "name": mod_name,
            "compiler": tool,
            "path": str(src),
            "binary_path": binary_path,
            "command": command_used,
            **build_result,
        }
        with self._lock:
            self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
            if result["ok"] and binary_path:
                self.lang_modules[mod_name]["binary_path"] = str(binary_path)
                bpath = str(binary_path).lower()
                if bpath.endswith(".dll"):
                    self.lang_modules[mod_name]["kind"] = "csharp_assembly"
                elif bpath.endswith(".exe"):
                    self.lang_modules[mod_name]["kind"] = "csharp_binary"
        self._append_event({"tick": tick_now, "event": "csharp_build", "name": mod_name, "ok": bool(result.get("ok"))})
        return result

    def _cpp_build_module(self, name: str, out_path: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mod_name = str(name)
        with self._lock:
            meta = copy.deepcopy(self.lang_modules.get(mod_name))
            tick_now = self.tick_count
        if not isinstance(meta, dict):
            raise RuntimeErrorNF(f"Unknown language module: {mod_name}")
        if meta.get("language") != "cpp":
            raise RuntimeErrorNF(f"Module {mod_name} is not a cpp module")
        src = Path(str(meta.get("path", "")))
        if not src.exists():
            result = {"ok": False, "name": mod_name, "reason": "source_missing", "path": str(src)}
            with self._lock:
                self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
            self._append_event({"tick": tick_now, "event": "cpp_build", **result})
            return result

        compiler = self._cpp_compiler_info()
        if compiler is None:
            result = {"ok": False, "name": mod_name, "reason": "compiler_missing"}
            with self._lock:
                self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
            self._append_event({"tick": tick_now, "event": "cpp_build", **result})
            return result

        cfg_map: Dict[str, Any] = {}
        if isinstance(meta.get("config"), dict):
            cfg_map.update(copy.deepcopy(meta.get("config")))  # type: ignore[arg-type]
        if isinstance(cfg, dict):
            cfg_map.update(copy.deepcopy(cfg))

        if out_path:
            out = self._resolve_output_path(str(out_path))
        else:
            ext = ".exe" if os.name == "nt" else ""
            out = self._resolve_output_path(f"cpp_modules/{mod_name}{ext}")
        flags = cfg_map.get("flags")
        extra_flags = [str(x) for x in flags] if isinstance(flags, list) else []
        std_flag = str(cfg_map.get("std", "c++17"))
        optimize = bool(cfg_map.get("optimize", True))

        if compiler["flavor"] == "gcc_like":
            cmd = [str(compiler["path"])]
            if std_flag:
                cmd.append(f"-std={std_flag}")
            if optimize:
                cmd.append("-O2")
            cmd.extend(["-o", str(out), str(src)])
            cmd.extend(extra_flags)
            build_result = self._run_subprocess(cmd, timeout_sec=float(cfg_map.get("timeout_sec", 60.0)))
        else:
            # Minimal MSVC path (requires Developer Command Prompt / environment initialized)
            std_part = f"/std:{std_flag}" if std_flag else ""
            opt_part = "/O2" if optimize else "/Od"
            cmd_str = " ".join(
                part
                for part in [
                    "cl",
                    "/nologo",
                    "/EHsc",
                    std_part,
                    opt_part,
                    f'/Fe:"{str(out)}"',
                    f'"{str(src)}"',
                    " ".join(extra_flags),
                ]
                if part
            )
            build_result = self._run_subprocess(["cmd", "/c", cmd_str], timeout_sec=float(cfg_map.get("timeout_sec", 60.0)))

        result = {
            "ok": bool(build_result.get("ok")) and out.exists(),
            "name": mod_name,
            "compiler": compiler,
            "path": str(src),
            "binary_path": str(out),
            **build_result,
        }
        with self._lock:
            self.lang_modules[mod_name]["build"] = copy.deepcopy(result)
            if result["ok"]:
                self.lang_modules[mod_name]["binary_path"] = str(out)
                self.lang_modules[mod_name]["kind"] = "cpp_binary"
                self.lang_modules[mod_name]["exists"] = src.exists()
        self._append_event({"tick": tick_now, "event": "cpp_build", "name": mod_name, "ok": result["ok"]})
        return result

    def _lang_module_run(
        self,
        name: str,
        args: Optional[Any] = None,
        *,
        timeout_sec: float = 30.0,
        stdin_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        mod_name = str(name)
        with self._lock:
            meta = copy.deepcopy(self.lang_modules.get(mod_name))
            tick_now = self.tick_count
        if not isinstance(meta, dict):
            raise RuntimeErrorNF(f"Unknown language module: {mod_name}")

        arg_list: List[str] = []
        if isinstance(args, list):
            arg_list = [str(x) for x in args]
        elif args is None:
            arg_list = []
        else:
            arg_list = [str(args)]

        lang = str(meta.get("language", "")).lower()
        cmd: Optional[List[str]] = None
        reason: Optional[str] = None

        if lang == "python":
            runtime = str((meta.get("config") or {}).get("runtime", meta.get("runtime", sys.executable or "python")))
            runtime_path = shutil.which(runtime) or (runtime if Path(runtime).exists() else None)
            if not runtime_path:
                reason = f"runtime_missing:{runtime}"
            else:
                cmd = [str(runtime_path), str(meta.get("path", "")), *arg_list]
        elif lang == "javascript":
            runtime = str((meta.get("config") or {}).get("runtime", meta.get("runtime", "node")))
            runtime_path = shutil.which(runtime) or (runtime if Path(runtime).exists() else None)
            if not runtime_path:
                reason = f"runtime_missing:{runtime}"
            else:
                cmd = [str(runtime_path), str(meta.get("path", "")), *arg_list]
        elif lang == "powershell":
            if not self._is_windows_host:
                reason = "not_windows"
            else:
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(meta.get("path", "")),
                    *arg_list,
                ]
        elif lang == "csharp":
            kind = str(meta.get("kind", ""))
            target_path = str(meta.get("path", ""))
            binary_path = meta.get("binary_path")
            if kind in {"csharp_project"} or (str(target_path).lower().endswith(".csproj") and not binary_path):
                build = self._csharp_build_module(mod_name)
                if not build.get("ok"):
                    reason = str(build.get("reason") or "csharp_build_failed")
                else:
                    binary_path = build.get("binary_path")
                    kind = "csharp_assembly" if str(binary_path or "").lower().endswith(".dll") else "csharp_binary"
            elif (not binary_path) and kind in {"csharp_source"}:
                build = self._csharp_build_module(mod_name)
                if not build.get("ok"):
                    reason = str(build.get("reason") or "csharp_build_failed")
                else:
                    binary_path = build.get("binary_path")
                    kind = "csharp_assembly" if str(binary_path or "").lower().endswith(".dll") else "csharp_binary"
            if reason is None:
                run_target = str(binary_path or target_path)
                lower_run = run_target.lower()
                if lower_run.endswith(".dll"):
                    dotnet_path = shutil.which("dotnet")
                    if not dotnet_path:
                        reason = "runtime_missing:dotnet"
                    else:
                        cmd = [str(dotnet_path), run_target, *arg_list]
                elif lower_run.endswith(".exe"):
                    cmd = [run_target, *arg_list]
                else:
                    # Allow direct `dotnet run --project` for csproj if provided and build wasn't required
                    if str(target_path).lower().endswith(".csproj"):
                        dotnet_path = shutil.which("dotnet")
                        if not dotnet_path:
                            reason = "runtime_missing:dotnet"
                        else:
                            cmd = [str(dotnet_path), "run", "--project", target_path, "--no-build", "--", *arg_list]
                    else:
                        reason = "csharp_no_runnable_target"
        elif lang == "rust":
            target_path = str(meta.get("path", ""))
            binary_path = meta.get("binary_path")
            kind = str(meta.get("kind", ""))
            if kind in {"rust_project", "rust_source"} or not (binary_path and Path(str(binary_path)).exists()):
                build = self._rust_build_module(mod_name)
                if not build.get("ok"):
                    reason = str(build.get("reason") or "rust_build_failed")
                else:
                    binary_path = build.get("binary_path")
            if reason is None and binary_path and Path(str(binary_path)).exists():
                cmd = [str(binary_path), *arg_list]
            elif reason is None and target_path and Path(target_path).exists() and kind == "rust_binary":
                cmd = [str(target_path), *arg_list]
            elif reason is None:
                reason = "rust_no_runnable_target"
        elif lang == "cpp":
            binary_path = meta.get("binary_path")
            if not binary_path or not Path(str(binary_path)).exists():
                build = self._cpp_build_module(mod_name)
                if not build.get("ok"):
                    reason = str(build.get("reason") or "cpp_build_failed")
                else:
                    binary_path = build.get("binary_path")
            if binary_path and Path(str(binary_path)).exists():
                cmd = [str(binary_path), *arg_list]
        else:
            # Generic script/binary fallback: try direct execution.
            p = Path(str(meta.get("path", "")))
            if p.exists():
                cmd = [str(p), *arg_list]
            else:
                reason = "path_missing"

        if cmd is None:
            result = {
                "ok": False,
                "name": mod_name,
                "language": lang,
                "args": arg_list,
                "reason": reason or "no_command",
                "duration_ms": 0.0,
            }
        else:
            sp = self._run_subprocess(cmd, stdin_text=stdin_text, timeout_sec=timeout_sec)
            result = {
                "name": mod_name,
                "language": lang,
                "args": arg_list,
                "command": cmd,
                **sp,
            }

        with self._lock:
            cur = self.lang_modules.get(mod_name, {})
            if isinstance(cur, dict):
                cur["run_count"] = int(cur.get("run_count", 0) or 0) + 1
                cur["last_run"] = copy.deepcopy({k: v for k, v in result.items() if k != "command" or isinstance(v, list)})
                self.lang_modules[mod_name] = cur
        self._record_lang_module_run({"tick": tick_now, **result})
        self._append_event({"tick": tick_now, "event": "lang_run", "name": mod_name, "ok": bool(result.get("ok"))})
        return result

    def _run_powershell(self, script: str, timeout_sec: float = 30.0, stdin_text: Optional[str] = None) -> Dict[str, Any]:
        if not self._is_windows_host:
            result = {"ok": False, "reason": "not_windows", "stdout": "", "stderr": "Host is not Windows"}
            self._record_windows_op({"type": "powershell", "script": script, **result})
            return result
        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ]
        result = self._run_subprocess(cmd, stdin_text=stdin_text, timeout_sec=timeout_sec)
        self._record_windows_op({"type": "powershell", "script": script, **result})
        return result

    def _http_request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, Any]] = None,
        auth_preset: Optional[str] = None,
        body: Optional[bytes] = None,
        timeout_sec: float = 15.0,
    ) -> Dict[str, Any]:
        method_up = str(method or "GET").upper()
        req_headers = self._resolve_http_headers(headers=headers, auth_preset=auth_preset)
        started = time.time()
        try:
            req = urllib.request.Request(url, data=body, headers=req_headers, method=method_up)
            with urllib.request.urlopen(req, timeout=max(0.1, float(timeout_sec))) as resp:
                raw = resp.read()
                ctype = getattr(resp, "headers", {}).get("Content-Type", "") if getattr(resp, "headers", None) else ""
                payload = {
                    "ok": True,
                    "status": getattr(resp, "status", 200),
                    "url": url,
                    "method": method_up,
                    "content_type": ctype,
                    "bytes": len(raw),
                    "body_bytes": raw,
                    "headers": dict(resp.headers.items()) if getattr(resp, "headers", None) else {},
                }
        except urllib.error.HTTPError as exc:
            err_bytes = exc.read() if hasattr(exc, "read") else b""
            payload = {
                "ok": False,
                "status": exc.code,
                "url": url,
                "method": method_up,
                "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
                "bytes": len(err_bytes),
                "body_bytes": err_bytes,
                "headers": dict(exc.headers.items()) if exc.headers else {},
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            payload = {
                "ok": False,
                "status": None,
                "url": url,
                "method": method_up,
                "content_type": "",
                "bytes": 0,
                "body_bytes": b"",
                "headers": {},
                "error": str(exc),
            }
        payload["duration_ms"] = round((time.time() - started) * 1000, 2)
        rec = {k: v for k, v in payload.items() if k != "body_bytes"}
        if auth_preset:
            rec["auth_preset"] = auth_preset
        preview = (payload.get("body_bytes") or b"")[:4000]
        try:
            rec["preview"] = preview.decode("utf-8", errors="replace")
        except Exception:
            rec["preview"] = ""
        self._record_http_op(rec)
        return payload

    def _resolve_runtime_path(self, rel_path: str) -> Path:
        p = Path(rel_path)
        return p if p.is_absolute() else (self.out_dir / p).resolve()

    def _safe_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _deep_get(self, value: Any, path: str, default: Any = None) -> Any:
        cur = value
        for part in str(path).split("."):
            if part == "":
                continue
            if isinstance(cur, dict):
                if part not in cur:
                    return default
                cur = cur[part]
                continue
            if isinstance(cur, list):
                try:
                    idx = int(part)
                except Exception:
                    return default
                if idx < 0 or idx >= len(cur):
                    return default
                cur = cur[idx]
                continue
            return default
        return cur

    def _write_text(self, rel_path: str, text: Any, append: bool = False) -> Path:
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with out.open(mode, encoding="utf-8") as fh:
            fh.write(self._safe_text(text))
        return out

    def _record_process_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        with self._lock:
            self.process_state["ops"].append(payload)
            cap = self._history_cap("process_ops_max", 120)
            self.process_state["ops"] = self.process_state["ops"][-cap:]

    def _record_wifi_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        with self._lock:
            self.wifi_state["ops"].append(payload)
            cap = self._history_cap("wifi_ops_max", 80)
            self.wifi_state["ops"] = self.wifi_state["ops"][-cap:]

    def _record_npu_probe(self, probe: Dict[str, Any]) -> None:
        payload = copy.deepcopy(probe)
        payload.setdefault("time", self._now_iso())
        with self._lock:
            self.npu_state["last_probe"] = copy.deepcopy(payload)
            probes = self.npu_state.setdefault("probes", [])
            probes.append(payload)
            cap = self._history_cap("npu_probes_max", 30)
            self.npu_state["probes"] = probes[-cap:]

    def _record_npu_op(self, op: Dict[str, Any], *, bucket: Optional[str] = None, last_key: Optional[str] = None) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        with self._lock:
            ops = self.npu_state.setdefault("ops", [])
            ops.append(payload)
            cap = self._history_cap("npu_probes_max", 30)
            self.npu_state["ops"] = ops[-cap:]
            if bucket:
                arr = self.npu_state.setdefault(str(bucket), [])
                if isinstance(arr, list):
                    arr.append(copy.deepcopy(payload))
                    self.npu_state[str(bucket)] = arr[-cap:]
            if last_key:
                self.npu_state[str(last_key)] = copy.deepcopy(payload)

    def _record_photo_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        with self._lock:
            self.photo_state["ops"].append(payload)
            cap = self._history_cap("photo_ops_max", 80)
            self.photo_state["ops"] = self.photo_state["ops"][-cap:]

    def _record_graph_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        with self._lock:
            self.graph_state["ops"].append(payload)
            cap = self._history_cap("graph_ops_max", 120)
            self.graph_state["ops"] = self.graph_state["ops"][-cap:]

    def _record_convert_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        with self._lock:
            self.convert_state["ops"].append(payload)
            cap = self._history_cap("convert_ops_max", 120)
            self.convert_state["ops"] = self.convert_state["ops"][-cap:]

    def _record_exe_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        key = str(op.get("out") or op.get("path") or op.get("artifact") or "")
        with self._lock:
            self.exe_state["ops"].append(payload)
            cap = self._history_cap("exe_ops_max", 80)
            self.exe_state["ops"] = self.exe_state["ops"][-cap:]
            self.exe_state["last"] = copy.deepcopy(payload)
            artifacts = self.exe_state.setdefault("artifacts", {})
            if key:
                info = artifacts.get(key, {"path": key, "ops": 0})
                info["ops"] = _as_int(info.get("ops", 0)) + 1
                info["last_op"] = payload.get("op")
                info["last_status"] = "ok" if payload.get("ok", True) else "error"
                for field in {
                    "name",
                    "tool",
                    "mode",
                    "dry_run",
                    "source",
                    "module",
                    "language",
                    "builder",
                    "reason",
                    "command",
                    "exists",
                    "bytes",
                }:
                    if field in payload:
                        info[field] = copy.deepcopy(payload.get(field))
                artifacts[key] = info

    def _record_github_local_op(self, op: Dict[str, Any]) -> None:
        payload = {"time": self._now_iso(), **copy.deepcopy(op)}
        with self._lock:
            self.github_local_state["ops"].append(payload)
            cap = self._history_cap("github_ops_max", 60)
            self.github_local_state["ops"] = self.github_local_state["ops"][-cap:]

    def _normalize_arg_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]

    def _normalize_env_dict(self, env_value: Any, inherit: bool = True) -> Dict[str, str]:
        env: Dict[str, str] = dict(os.environ) if inherit else {}
        if isinstance(env_value, dict):
            for k, v in env_value.items():
                if v is None:
                    env.pop(str(k), None)
                else:
                    env[str(k)] = str(v)
        return env

    def _proc_profile_register(self, name: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        profile_name = str(name)
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        inherit_env = bool(cfg_map.get("inherit_env", True))
        env_cfg = cfg_map.get("env")
        cwd_value = cfg_map.get("cwd")
        virtualized = bool(cfg_map.get("virtualized", False) or cfg_map.get("sandbox", False))
        sandbox_dir: Optional[Path] = None
        if virtualized:
            sandbox_hint = str(cfg_map.get("sandbox_dir", "")).strip()
            if sandbox_hint:
                sandbox_dir = self._resolve_runtime_path(sandbox_hint)
            else:
                sandbox_dir = self._resolve_output_path(f"proc_sandboxes/{self._web_slugify(profile_name)}")
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            if not cwd_value:
                cwd_value = str(sandbox_dir)
        cwd_path: Optional[str] = None
        if isinstance(cwd_value, str) and cwd_value.strip():
            cwd_path = str(self._resolve_runtime_path(cwd_value).resolve())
            Path(cwd_path).mkdir(parents=True, exist_ok=True)
        profile = {
            "name": profile_name,
            "cwd": cwd_path,
            "virtualized": virtualized,
            "sandbox_dir": str(sandbox_dir) if sandbox_dir else None,
            "inherit_env": inherit_env,
            "env": {str(k): (None if v is None else str(v)) for k, v in (env_cfg.items() if isinstance(env_cfg, dict) else [])},
            "shell": bool(cfg_map.get("shell", False)),
            "timeout_sec": float(cfg_map.get("timeout_sec", 30.0)),
            "created_at": self._now_iso(),
            "config": cfg_map,
        }
        with self._lock:
            prev = self.process_state["profiles"].get(profile_name, {})
            if isinstance(prev, dict) and "run_count" in prev:
                profile["run_count"] = int(prev.get("run_count", 0))
            else:
                profile["run_count"] = 0
            self.process_state["profiles"][profile_name] = profile
            tick_now = self.tick_count
        self._record_process_op({"type": "proc_profile", "name": profile_name, "virtualized": virtualized, "cwd": cwd_path})
        self._append_event({"tick": tick_now, "event": "proc_profile", "name": profile_name})
        return copy.deepcopy(profile)

    def _proc_merged_cfg(self, profile_name: Optional[str], cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if profile_name:
            with self._lock:
                base = copy.deepcopy(self.process_state.get("profiles", {}).get(profile_name))
            if isinstance(base, dict):
                merged.update(copy.deepcopy(base.get("config", {})) if isinstance(base.get("config"), dict) else {})
                if base.get("cwd"):
                    merged["cwd"] = base.get("cwd")
                merged["shell"] = bool(base.get("shell", False))
                merged["timeout_sec"] = float(base.get("timeout_sec", 30.0))
                merged["inherit_env"] = bool(base.get("inherit_env", True))
                if isinstance(base.get("env"), dict):
                    merged["env"] = copy.deepcopy(base.get("env"))
                merged["virtualized"] = bool(base.get("virtualized", False))
                if base.get("sandbox_dir"):
                    merged["sandbox_dir"] = base.get("sandbox_dir")
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if k == "env" and isinstance(v, dict):
                    env_cur = merged.get("env", {})
                    if not isinstance(env_cur, dict):
                        env_cur = {}
                    env_cur = {**env_cur, **v}
                    merged["env"] = env_cur
                else:
                    merged[k] = v
        return merged

    def _proc_exec(self, command: str, args: Any = None, cfg: Optional[Dict[str, Any]] = None, profile_name: Optional[str] = None) -> Dict[str, Any]:
        cfg_map = self._proc_merged_cfg(profile_name, cfg)
        arg_list = self._normalize_arg_list(args)
        shell = bool(cfg_map.get("shell", False))
        timeout_sec = float(cfg_map.get("timeout_sec", 30.0))
        cwd_value = cfg_map.get("cwd")
        cwd = str(cwd_value) if cwd_value else None
        env = self._normalize_env_dict(cfg_map.get("env"), inherit=bool(cfg_map.get("inherit_env", True)))
        cmd_list = [str(command), *arg_list]
        started = time.time()
        try:
            proc = subprocess.run(
                cmd_list if not shell else " ".join(cmd_list),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                encoding="utf-8",
                errors="replace",
                check=False,
                cwd=cwd,
                env=env,
                shell=shell,
            )
            result = {
                "ok": proc.returncode == 0,
                "returncode": int(proc.returncode),
                "stdout": (proc.stdout or "")[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "ok": False,
                "returncode": None,
                "timeout": True,
                "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            }
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "returncode": None, "error": str(exc), "stdout": "", "stderr": ""}
        result["duration_ms"] = round((time.time() - started) * 1000, 2)
        result["command"] = cmd_list
        if profile_name:
            result["profile"] = str(profile_name)
        if cwd:
            result["cwd"] = cwd
        self._record_process_op({"type": "proc_exec", **{k: v for k, v in result.items() if k not in {"stdout", "stderr"}}})
        with self._lock:
            tick_now = self.tick_count
            if profile_name and profile_name in self.process_state.get("profiles", {}):
                self.process_state["profiles"][str(profile_name)]["run_count"] = int(
                    self.process_state["profiles"][str(profile_name)].get("run_count", 0)
                ) + 1
        self._append_event({"tick": tick_now, "event": "proc_exec", "ok": bool(result.get("ok"))})
        return result

    def _proc_spawn(self, name: str, command: str, args: Any = None, cfg: Optional[Dict[str, Any]] = None, profile_name: Optional[str] = None) -> Dict[str, Any]:
        proc_name = str(name)
        cfg_map = self._proc_merged_cfg(profile_name, cfg)
        arg_list = self._normalize_arg_list(args)
        shell = bool(cfg_map.get("shell", False))
        cwd_value = cfg_map.get("cwd")
        cwd = str(cwd_value) if cwd_value else None
        env = self._normalize_env_dict(cfg_map.get("env"), inherit=bool(cfg_map.get("inherit_env", True)))
        cmd_list = [str(command), *arg_list]
        try:
            popen_obj = subprocess.Popen(
                cmd_list if not shell else " ".join(cmd_list),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                env=env,
                shell=shell,
            )
            result = {
                "ok": True,
                "name": proc_name,
                "pid": int(popen_obj.pid),
                "running": True,
                "command": cmd_list,
                "profile": profile_name,
                "cwd": cwd,
            }
            with self._lock:
                self._managed_process_handles[proc_name] = popen_obj
                self.process_state["managed"][proc_name] = {
                    "name": proc_name,
                    "pid": int(popen_obj.pid),
                    "command": cmd_list,
                    "profile": profile_name,
                    "cwd": cwd,
                    "started_at": self._now_iso(),
                    "running": True,
                    "returncode": None,
                }
                tick_now = self.tick_count
                if profile_name and profile_name in self.process_state.get("profiles", {}):
                    self.process_state["profiles"][str(profile_name)]["run_count"] = int(
                        self.process_state["profiles"][str(profile_name)].get("run_count", 0)
                    ) + 1
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "name": proc_name, "error": str(exc), "command": cmd_list}
            with self._lock:
                tick_now = self.tick_count
        self._record_process_op({"type": "proc_spawn", **result})
        self._append_event({"tick": tick_now, "event": "proc_spawn", "name": proc_name, "ok": bool(result.get("ok"))})
        return result

    def _proc_wait(self, name: str, timeout_sec: float = 30.0) -> Dict[str, Any]:
        proc_name = str(name)
        with self._lock:
            handle = self._managed_process_handles.get(proc_name)
            tick_now = self.tick_count
        if handle is None:
            return {"ok": False, "name": proc_name, "reason": "not_found"}
        try:
            stdout, stderr = handle.communicate(timeout=max(0.1, float(timeout_sec)))
            rc = handle.returncode
            result = {
                "ok": rc == 0,
                "name": proc_name,
                "returncode": int(rc) if rc is not None else None,
                "stdout": (stdout or "")[-4000:],
                "stderr": (stderr or "")[-4000:],
                "running": False,
            }
            with self._lock:
                self._managed_process_handles.pop(proc_name, None)
                if proc_name in self.process_state.get("managed", {}):
                    self.process_state["managed"][proc_name]["running"] = False
                    self.process_state["managed"][proc_name]["returncode"] = result["returncode"]
                    self.process_state["managed"][proc_name]["ended_at"] = self._now_iso()
        except subprocess.TimeoutExpired as exc:
            result = {
                "ok": False,
                "name": proc_name,
                "reason": "timeout",
                "timeout": True,
                "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                "running": True,
            }
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "name": proc_name, "error": str(exc)}
        self._record_process_op({"type": "proc_wait", **{k: v for k, v in result.items() if k not in {"stdout", "stderr"}}})
        self._append_event({"tick": tick_now, "event": "proc_wait", "name": proc_name, "ok": bool(result.get("ok"))})
        return result

    def _proc_kill(self, name: str, force: bool = False) -> Dict[str, Any]:
        proc_name = str(name)
        with self._lock:
            handle = self._managed_process_handles.get(proc_name)
            tick_now = self.tick_count
        if handle is None:
            return {"ok": False, "name": proc_name, "reason": "not_found"}
        try:
            if force:
                handle.kill()
            else:
                handle.terminate()
            try:
                stdout, stderr = handle.communicate(timeout=2.0)
            except Exception:
                stdout, stderr = ("", "")
            rc = handle.returncode
            result = {
                "ok": True,
                "name": proc_name,
                "returncode": int(rc) if rc is not None else None,
                "stdout": (stdout or "")[-4000:],
                "stderr": (stderr or "")[-4000:],
                "force": bool(force),
            }
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "name": proc_name, "error": str(exc), "force": bool(force)}
        with self._lock:
            self._managed_process_handles.pop(proc_name, None)
            if proc_name in self.process_state.get("managed", {}):
                self.process_state["managed"][proc_name]["running"] = False
                self.process_state["managed"][proc_name]["returncode"] = result.get("returncode")
                self.process_state["managed"][proc_name]["ended_at"] = self._now_iso()
        self._record_process_op({"type": "proc_kill", **{k: v for k, v in result.items() if k not in {"stdout", "stderr"}}})
        self._append_event({"tick": tick_now, "event": "proc_kill", "name": proc_name, "ok": bool(result.get("ok"))})
        return result

    def _proc_history_json(self, rel_path: str, limit: Optional[int] = None) -> Path:
        with self._lock:
            ops = copy.deepcopy(self.process_state.get("ops", []))
        if isinstance(limit, int) and limit > 0:
            ops = ops[-limit:]
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps({"ops": ops}, indent=2), encoding="utf-8")
        return out

    def _proc_managed_json(self, rel_path: str) -> Path:
        with self._lock:
            managed = copy.deepcopy(self.process_state.get("managed", {}))
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps({"managed": managed}, indent=2), encoding="utf-8")
        return out

    def _wifi_netsh(self, args: List[str], timeout_sec: float = 20.0) -> Dict[str, Any]:
        if not self._is_windows_host:
            return {"ok": False, "reason": "not_windows", "stdout": "", "stderr": "Host is not Windows", "args": list(args)}
        if shutil.which("netsh") is None:
            return {"ok": False, "reason": "netsh_missing", "stdout": "", "stderr": "netsh not found", "args": list(args)}
        result = self._run_subprocess(["netsh", "wlan", *args], timeout_sec=timeout_sec)
        result["args"] = list(args)
        return result

    def _wifi_parse_interfaces(self, text: str) -> List[Dict[str, Any]]:
        interfaces: List[Dict[str, Any]] = []
        cur: Dict[str, Any] = {}
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("-") or ":" not in line:
                continue
            key, value = [p.strip() for p in line.split(":", 1)]
            key_norm = key.lower().replace(" ", "_")
            if key_norm == "name" and cur:
                interfaces.append(cur)
                cur = {}
            cur[key_norm] = value
        if cur:
            interfaces.append(cur)
        return interfaces

    def _wifi_parse_profiles(self, text: str) -> List[str]:
        profiles: List[str] = []
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if "All User Profile" in line and ":" in line:
                profiles.append(line.split(":", 1)[1].strip())
        return sorted(dict.fromkeys(profiles))

    def _wifi_parse_scan(self, text: str) -> Dict[str, Any]:
        networks: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            m_ssid = re.match(r"SSID\s+\d+\s*:\s*(.*)$", line, flags=re.IGNORECASE)
            if m_ssid:
                if cur:
                    networks.append(cur)
                cur = {"ssid": m_ssid.group(1).strip()}
                continue
            if cur is None or ":" not in line:
                continue
            k, v = [p.strip() for p in line.split(":", 1)]
            k_norm = k.lower().replace(" ", "_")
            if k_norm in {"signal", "authentication", "encryption"}:
                alias = {"authentication": "auth"}.get(k_norm, k_norm)
                cur[alias] = v
            elif k_norm.startswith("bssid"):
                bssids = cur.setdefault("bssids", [])
                if isinstance(bssids, list):
                    bssids.append(v)
            else:
                cur[k_norm] = v
        if cur:
            networks.append(cur)
        return {"networks": networks, "count": len(networks)}

    def _wifi_interfaces(self) -> Dict[str, Any]:
        result = self._wifi_netsh(["show", "interfaces"])
        if result.get("ok"):
            items = self._wifi_parse_interfaces(str(result.get("stdout", "")))
            payload = {"ok": True, "interfaces": items, "count": len(items)}
            with self._lock:
                self.wifi_state["interfaces"] = copy.deepcopy(items)
        else:
            payload = {"ok": False, "reason": result.get("reason"), "error": result.get("stderr") or result.get("error"), "interfaces": [], "count": 0}
        self._record_wifi_op({"type": "wifi_interfaces", "ok": payload["ok"], "count": payload.get("count", 0), "reason": payload.get("reason")})
        return payload

    def _wifi_profiles(self) -> Dict[str, Any]:
        result = self._wifi_netsh(["show", "profiles"])
        if result.get("ok"):
            profiles = self._wifi_parse_profiles(str(result.get("stdout", "")))
            payload = {"ok": True, "profiles": profiles, "count": len(profiles)}
            with self._lock:
                self.wifi_state["profiles"] = copy.deepcopy(profiles)
        else:
            payload = {"ok": False, "reason": result.get("reason"), "error": result.get("stderr") or result.get("error"), "profiles": [], "count": 0}
        self._record_wifi_op({"type": "wifi_profiles", "ok": payload["ok"], "count": payload.get("count", 0), "reason": payload.get("reason")})
        return payload

    def _wifi_scan(self) -> Dict[str, Any]:
        result = self._wifi_netsh(["show", "networks", "mode=bssid"])
        if result.get("ok"):
            payload = {"ok": True, **self._wifi_parse_scan(str(result.get("stdout", "")))}
        else:
            payload = {"ok": False, "reason": result.get("reason"), "error": result.get("stderr") or result.get("error"), "networks": [], "count": 0}
        with self._lock:
            self.wifi_state["last_scan"] = copy.deepcopy(payload)
        self._record_wifi_op({"type": "wifi_scan", "ok": payload["ok"], "count": payload.get("count", 0), "reason": payload.get("reason")})
        return payload

    def _wifi_connect(self, profile: str, interface: Optional[str] = None) -> Dict[str, Any]:
        args = ["connect", f'name="{str(profile)}"']
        if interface:
            args.append(f'interface="{str(interface)}"')
        result = self._wifi_netsh(args)
        payload = {
            "ok": bool(result.get("ok")),
            "profile": str(profile),
            "interface": interface,
            "reason": result.get("reason"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
        self._record_wifi_op({"type": "wifi_connect", "ok": payload["ok"], "profile": payload["profile"], "interface": payload["interface"], "reason": payload.get("reason")})
        return payload

    def _wifi_disconnect(self, interface: Optional[str] = None) -> Dict[str, Any]:
        args = ["disconnect"]
        if interface:
            args.append(f'interface="{str(interface)}"')
        result = self._wifi_netsh(args)
        payload = {
            "ok": bool(result.get("ok")),
            "interface": interface,
            "reason": result.get("reason"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
        self._record_wifi_op({"type": "wifi_disconnect", "ok": payload["ok"], "interface": payload["interface"], "reason": payload.get("reason")})
        return payload

    def _wifi_export_json(self, kind: str, rel_path: str) -> Path:
        k = str(kind).lower()
        if k == "interfaces":
            payload = self._wifi_interfaces()
        elif k == "profiles":
            payload = self._wifi_profiles()
        elif k == "scan":
            payload = self._wifi_scan()
        else:
            raise RuntimeErrorNF(f"Unknown wifi export kind: {kind}")
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def _accelerator_probe(self, refresh: bool = False, deep: bool = False) -> Dict[str, Any]:
        cache_key = f"accelerator_probe:{'deep' if deep else 'shallow'}"
        if not refresh:
            cached = self._cache_get(cache_key, ttl_ms=5000)
            if isinstance(cached, dict):
                return cached
        probe: Dict[str, Any] = {
            "host": self._host_info(),
            "cpu_count": os.cpu_count() or 1,
            "pytorch_available": TORCH_AVAILABLE,
            "torch": {
                "available": TORCH_AVAILABLE,
                "cuda": {"available": False, "count": 0, "devices": []},
                "mps": {"available": False},
                "xpu": {"available": False, "count": 0},
            },
            "npu": {"available": False, "detected": [], "env_hints": []},
            "recommended_device": "cpu",
            "deep": bool(deep),
            "probed_at": self._now_iso(),
        }
        if TORCH_AVAILABLE and torch is not None:
            try:
                cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
            except Exception:
                cuda_available = False
            cuda_devices: List[str] = []
            cuda_count = 0
            if cuda_available:
                try:
                    cuda_count = int(torch.cuda.device_count())
                except Exception:
                    cuda_count = 0
                for i in range(max(0, cuda_count)):
                    try:
                        cuda_devices.append(str(torch.cuda.get_device_name(i)))
                    except Exception:
                        cuda_devices.append(f"cuda:{i}")
            probe["torch"]["cuda"] = {"available": cuda_available, "count": cuda_count, "devices": cuda_devices}
            try:
                mps_backend = getattr(torch.backends, "mps", None)
                mps_avail = bool(mps_backend and mps_backend.is_available())
            except Exception:
                mps_avail = False
            probe["torch"]["mps"] = {"available": mps_avail}
            try:
                xpu_mod = getattr(torch, "xpu", None)
                xpu_avail = bool(xpu_mod and xpu_mod.is_available())
                xpu_count = int(xpu_mod.device_count()) if xpu_avail and hasattr(xpu_mod, "device_count") else 0
            except Exception:
                xpu_avail = False
                xpu_count = 0
            probe["torch"]["xpu"] = {"available": xpu_avail, "count": xpu_count}
            if cuda_available:
                probe["recommended_device"] = "cuda"
            elif mps_avail:
                probe["recommended_device"] = "mps"
            elif xpu_avail:
                probe["recommended_device"] = "xpu"

        env_hints = []
        for key in os.environ.keys():
            ku = str(key).upper()
            if ku.startswith(("NPU_", "NEURON_", "HAILO_", "OPENVINO_", "QNN_", "RKNN_")):
                env_hints.append(str(key))
        probe["npu"]["env_hints"] = sorted(env_hints)[:40]

        detected_npu: List[str] = []
        if deep and self._is_windows_host:
            ps_cmd = (
                "Get-CimInstance Win32_PnPEntity | "
                "Where-Object { $_.Name -match 'NPU|Neural|AI Boost|Hexagon' } | "
                "Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
            )
            ps = self._run_subprocess(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd], timeout_sec=8.0)
            if ps.get("ok"):
                txt = str(ps.get("stdout", "")).strip()
                if txt:
                    try:
                        parsed = json.loads(txt)
                        if isinstance(parsed, list):
                            detected_npu.extend(str(x) for x in parsed)
                        elif isinstance(parsed, str):
                            detected_npu.append(parsed)
                    except Exception:
                        for line in txt.splitlines():
                            line = line.strip().strip('"')
                            if line:
                                detected_npu.append(line)
        elif deep and shutil.which("lspci"):
            pci = self._run_subprocess(["lspci"], timeout_sec=4.0)
            if pci.get("ok"):
                for line in str(pci.get("stdout", "")).splitlines():
                    if re.search(r"NPU|Neural|AI accelerator|VPU", line, flags=re.IGNORECASE):
                        detected_npu.append(line.strip())
        probe["npu"]["detected"] = detected_npu[:30]
        if probe["npu"]["detected"] or probe["npu"]["env_hints"]:
            probe["npu"]["available"] = True
            if probe["recommended_device"] == "cpu":
                probe["recommended_device"] = "npu"
        probe["npu"]["providers"] = self._npu_detect_providers(probe)
        self._record_npu_probe(probe)
        self._cache_set(cache_key, probe, ttl_ms=5000)
        return copy.deepcopy(probe)

    def _npu_probe_json(self, rel_path: str, deep: bool = False) -> Path:
        payload = self._accelerator_probe(refresh=True, deep=deep)
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def _npu_detect_providers(self, probe: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        probe_map = probe if isinstance(probe, dict) else self._accelerator_probe(refresh=False, deep=False)
        env_hints = [str(x) for x in ((probe_map.get("npu") or {}).get("env_hints") or [])]
        env_upper = [x.upper() for x in env_hints]
        providers: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def add(name: str, *, source: str, available: bool = True, detail: Optional[str] = None) -> None:
            key = str(name).lower()
            if key in seen:
                return
            seen.add(key)
            rec: Dict[str, Any] = {"name": str(name), "available": bool(available), "source": str(source)}
            if detail:
                rec["detail"] = str(detail)
            providers.append(rec)

        # Tool/runtime hints
        tool_checks = [
            ("openvino", ["benchmark_app", "mo"]),
            ("qnn", ["qnn-net-run", "qnn-context-binary-generator"]),
            ("rknn", ["rknn_toolkit2"]),
            ("hailo", ["hailortcli", "hailo"]),
            ("neuron", ["neuron-ls", "neuron-top"]),
            ("xnnpack", []),
            ("onnxruntime", []),
        ]
        for name, cmds in tool_checks:
            found_cmd = None
            for cmd in cmds:
                p = shutil.which(cmd)
                if p:
                    found_cmd = p
                    break
            if found_cmd:
                add(name, source="tool", available=True, detail=found_cmd)

        # Environment-driven hints
        env_map = [
            ("OPENVINO_", "openvino"),
            ("QNN_", "qnn"),
            ("RKNN_", "rknn"),
            ("HAILO_", "hailo"),
            ("NEURON_", "neuron"),
            ("NPU_", "generic_npu"),
        ]
        for prefix, provider in env_map:
            if any(k.startswith(prefix) for k in env_upper):
                add(provider, source="env", available=True, detail=prefix)

        # Torch-backed accelerators can be used as NPU-adjacent runtime paths
        torch_info = probe_map.get("torch") if isinstance(probe_map, dict) else {}
        if isinstance(torch_info, dict):
            if bool(((torch_info.get("xpu") or {}).get("available"))):
                add("torch_xpu", source="torch", available=True)
            if bool(((torch_info.get("cuda") or {}).get("available"))):
                add("torch_cuda", source="torch", available=True)
            if bool(((torch_info.get("mps") or {}).get("available"))):
                add("torch_mps", source="torch", available=True)
        # Windows AI Boost / generic NPU device strings
        for d in ((probe_map.get("npu") or {}).get("detected") or []):
            ds = str(d)
            if "AI Boost" in ds or "NPU" in ds or "Neural" in ds:
                add("windows_npu_device", source="probe", available=True, detail=ds)
                break

        if not providers:
            add("cpu_fallback", source="fallback", available=True)
        return providers

    def _npu_profile_set(self, name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(cfg, dict):
            raise RuntimeErrorNF("npu_profile(name, config) expects dict config")
        profile_name = str(name).strip()
        if not profile_name:
            raise RuntimeErrorNF("npu_profile requires non-empty profile name")
        profile = {
            "name": profile_name,
            "precision": str(cfg.get("precision", "auto")),
            "optimize_for": str(cfg.get("optimize_for", cfg.get("mode", "balanced"))),
            "preferred_provider": cfg.get("preferred_provider"),
            "preferred_device": cfg.get("preferred_device"),
            "batch_size": cfg.get("batch_size"),
            "quantize": bool(cfg.get("quantize", False)),
            "threads": cfg.get("threads"),
            "notes": cfg.get("notes"),
            "updated_at": self._now_iso(),
        }
        with self._lock:
            profiles = self.npu_state.setdefault("profiles", {})
            profiles[profile_name] = copy.deepcopy(profile)
            tick_now = self.tick_count
        self._record_npu_op({"type": "npu_profile_set", "ok": True, "profile": profile_name})
        self._append_event({"tick": tick_now, "event": "npu_profile_set", "profile": profile_name})
        return profile

    def _npu_profile_get(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy((self.npu_state.get("profiles") or {}).get(str(name)))

    def _npu_select_execution_device(self, requested: Optional[str] = None, probe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        probe_map = probe if isinstance(probe, dict) else self._accelerator_probe(refresh=False, deep=False)
        req = str(requested or "auto").strip().lower()
        torch_info = (probe_map.get("torch") or {}) if isinstance(probe_map, dict) else {}
        xpu_avail = bool(((torch_info.get("xpu") or {}).get("available"))) if isinstance(torch_info, dict) else False
        cuda_avail = bool(((torch_info.get("cuda") or {}).get("available"))) if isinstance(torch_info, dict) else False
        mps_avail = bool(((torch_info.get("mps") or {}).get("available"))) if isinstance(torch_info, dict) else False
        npu_avail = bool(((probe_map.get("npu") or {}).get("available"))) if isinstance(probe_map, dict) else False
        rec = str(probe_map.get("recommended_device", "cpu")) if isinstance(probe_map, dict) else "cpu"

        if req in {"xpu", "intel_xpu"}:
            return {"requested": req, "execution_device": "xpu" if xpu_avail else "cpu", "used_fallback": not xpu_avail, "fallback_reason": None if xpu_avail else "xpu_unavailable"}
        if req in {"cuda", "gpu"}:
            return {"requested": req, "execution_device": "cuda" if cuda_avail else "cpu", "used_fallback": not cuda_avail, "fallback_reason": None if cuda_avail else "cuda_unavailable"}
        if req in {"mps"}:
            return {"requested": req, "execution_device": "mps" if mps_avail else "cpu", "used_fallback": not mps_avail, "fallback_reason": None if mps_avail else "mps_unavailable"}
        if req in {"npu"}:
            if xpu_avail:
                return {"requested": req, "execution_device": "xpu", "logical_device": "npu", "used_fallback": False, "fallback_reason": None}
            if npu_avail:
                return {"requested": req, "execution_device": "cpu", "logical_device": "npu", "used_fallback": True, "fallback_reason": "npu_runtime_not_bound_to_torch"}
            return {"requested": req, "execution_device": "cpu", "logical_device": "npu", "used_fallback": True, "fallback_reason": "npu_unavailable"}
        if req in {"cpu"}:
            return {"requested": req, "execution_device": "cpu", "used_fallback": False, "fallback_reason": None}

        # auto path: prefer real accelerators, but respect probe recommendation
        if rec == "cuda" and cuda_avail:
            return {"requested": req, "execution_device": "cuda", "used_fallback": False, "fallback_reason": None}
        if rec == "mps" and mps_avail:
            return {"requested": req, "execution_device": "mps", "used_fallback": False, "fallback_reason": None}
        if rec == "xpu" and xpu_avail:
            return {"requested": req, "execution_device": "xpu", "used_fallback": False, "fallback_reason": None}
        if rec == "npu":
            if xpu_avail:
                return {"requested": req, "execution_device": "xpu", "logical_device": "npu", "used_fallback": False, "fallback_reason": None}
            return {"requested": req, "execution_device": "cpu", "logical_device": "npu", "used_fallback": True, "fallback_reason": "npu_runtime_not_bound_to_torch"}
        if xpu_avail:
            return {"requested": req, "execution_device": "xpu", "used_fallback": False, "fallback_reason": None}
        if cuda_avail:
            return {"requested": req, "execution_device": "cuda", "used_fallback": False, "fallback_reason": None}
        if mps_avail:
            return {"requested": req, "execution_device": "mps", "used_fallback": False, "fallback_reason": None}
        return {"requested": req, "execution_device": "cpu", "used_fallback": False, "fallback_reason": None}

    def _npu_workload_from_ref(self, workload_or_model: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        def _cfg_or_default(key: str, default: Any) -> Any:
            val = cfg_map.get(key, default)
            return default if val is None else val
        if isinstance(workload_or_model, dict):
            base = copy.deepcopy(workload_or_model)
            base.setdefault("name", str(base.get("name", "workload")))
            return base
        name = str(workload_or_model)
        model_spec = copy.deepcopy(self.models.get(name, {})) if name in self.models else {}
        dataset_ref = str(cfg_map.get("dataset", ""))
        dataset_spec = copy.deepcopy(self.datasets.get(dataset_ref, {})) if dataset_ref in self.datasets else {}
        dataset_batch_default = dataset_spec.get("batch_size", 16)
        if dataset_batch_default is None:
            dataset_batch_default = 16
        inputs = _as_int(model_spec.get("inputs", _cfg_or_default("inputs", dataset_spec.get("inputs", 8))))
        outputs = _as_int(model_spec.get("outputs", _cfg_or_default("outputs", dataset_spec.get("outputs", 1))))
        hidden = _as_int(model_spec.get("hidden", _cfg_or_default("hidden", model_spec.get("width", 16))))
        layers = _as_int(model_spec.get("layers", _cfg_or_default("layers", 2)))
        batch_size = _as_int(_cfg_or_default("batch_size", dataset_batch_default))
        task = str(_cfg_or_default("task", dataset_spec.get("task", model_spec.get("task", "inference"))))
        params_est = max(1, (inputs * hidden) + (hidden * hidden * max(0, layers - 1)) + (hidden * outputs))
        return {
            "name": name,
            "kind": "model_ref" if model_spec else "workload_ref",
            "model_spec": model_spec,
            "dataset_spec": dataset_spec,
            "inputs": max(1, inputs),
            "outputs": max(1, outputs),
            "hidden": max(1, hidden),
            "layers": max(1, layers),
            "batch_size": max(1, batch_size),
            "task": task,
            "params_estimate": int(params_est),
        }

    def _npu_plan(self, workload_or_model: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        profile_name = cfg_map.get("profile")
        profile = self._npu_profile_get(str(profile_name)) if profile_name is not None else None
        merged_cfg = {}
        if isinstance(profile, dict):
            merged_cfg.update(copy.deepcopy(profile))
        merged_cfg.update(cfg_map)
        def _merged_or_default(key: str, default: Any) -> Any:
            v = merged_cfg.get(key, default)
            return default if v is None else v
        deep_probe = bool(merged_cfg.get("deep_probe", False))
        refresh = bool(merged_cfg.get("refresh_probe", False))
        probe = self._accelerator_probe(refresh=refresh, deep=deep_probe)
        providers = self._npu_detect_providers(probe)
        workload = self._npu_workload_from_ref(workload_or_model, merged_cfg)

        selected_provider = None
        preferred_provider = merged_cfg.get("preferred_provider")
        if preferred_provider:
            pref = str(preferred_provider).lower()
            for p in providers:
                if str(p.get("name", "")).lower() == pref:
                    selected_provider = p
                    break
        if selected_provider is None:
            # Prefer real NPU-oriented providers, then torch_xpu
            provider_order = ["openvino", "qnn", "rknn", "hailo", "neuron", "windows_npu_device", "torch_xpu", "torch_cuda", "torch_mps", "cpu_fallback"]
            for key in provider_order:
                found = next((p for p in providers if str(p.get("name")) == key), None)
                if found:
                    selected_provider = found
                    break
        if selected_provider is None:
            selected_provider = {"name": "cpu_fallback", "available": True, "source": "fallback"}

        preferred_device = merged_cfg.get("preferred_device")
        sel = self._npu_select_execution_device(str(preferred_device) if preferred_device is not None else "auto", probe=probe)
        precision = str(merged_cfg.get("precision", "auto")).lower()
        if precision == "auto":
            precision = "int8" if ("openvino" in str(selected_provider.get("name")) or bool(merged_cfg.get("quantize"))) else "fp16" if sel.get("execution_device") in {"cuda", "xpu", "mps"} else "fp32"
        optimize_for = str(merged_cfg.get("optimize_for", "balanced")).lower()
        batch_size = max(1, _as_int(_merged_or_default("batch_size", workload.get("batch_size", 1))))
        params_est = max(1, _as_int(workload.get("params_estimate", 1)))
        bytes_per_param = 1 if precision == "int8" else 2 if precision in {"fp16", "bf16"} else 4
        weights_mem = params_est * bytes_per_param
        activation_mem = batch_size * max(1, _as_int(workload.get("hidden", workload.get("inputs", 8)))) * bytes_per_param * max(1, _as_int(workload.get("layers", 1)))
        estimated_mem = int(weights_mem + activation_mem)
        throughput_score = 1.0
        if str(selected_provider.get("name")) in {"openvino", "qnn", "rknn", "hailo", "neuron", "windows_npu_device"}:
            throughput_score = 3.0
        elif str(selected_provider.get("name")) in {"torch_xpu", "torch_cuda"}:
            throughput_score = 2.0
        elif str(selected_provider.get("name")) == "torch_mps":
            throughput_score = 1.6
        if optimize_for == "latency":
            throughput_score *= 0.9
            batch_size = min(batch_size, 4)
        elif optimize_for == "throughput":
            throughput_score *= 1.2
            batch_size = max(batch_size, 8)
        fallback_chain = [sel.get("execution_device", "cpu")]
        for d in ["xpu", "cuda", "mps", "cpu"]:
            if d not in fallback_chain:
                fallback_chain.append(d)
        plan = {
            "ok": True,
            "name": str(merged_cfg.get("name", workload.get("name", "npu_workload"))),
            "workload": workload,
            "profile": profile_name if profile_name is not None else None,
            "provider": copy.deepcopy(selected_provider),
            "providers": copy.deepcopy(providers),
            "probe_summary": {
                "recommended_device": probe.get("recommended_device"),
                "npu_available": bool(((probe.get("npu") or {}).get("available"))),
                "torch": copy.deepcopy(probe.get("torch")),
            },
            "execution_device": sel.get("execution_device", "cpu"),
            "logical_device": sel.get("logical_device", "npu" if str(selected_provider.get("name")) in {"openvino", "qnn", "rknn", "hailo", "neuron", "windows_npu_device"} else sel.get("execution_device", "cpu")),
            "used_fallback": bool(sel.get("used_fallback", False)),
            "fallback_reason": sel.get("fallback_reason"),
            "fallback_chain": fallback_chain,
            "precision": precision,
            "quantize": bool(merged_cfg.get("quantize", precision == "int8")),
            "optimize_for": optimize_for,
            "batch_size": batch_size,
            "threads": int(_merged_or_default("threads", max(1, min(8, os.cpu_count() or 1)))),
            "estimated_memory_bytes": estimated_mem,
            "estimated_throughput_score": round(float(throughput_score), 4),
            "created_at": self._now_iso(),
        }
        self._record_npu_op({"type": "npu_plan", **copy.deepcopy(plan)}, bucket="plans", last_key="last_plan")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "npu_plan", "device": str(plan.get("execution_device")), "provider": str((plan.get("provider") or {}).get("name"))})
        return plan

    def _npu_plan_json(self, rel_path: str, workload_or_model: Any, cfg: Optional[Dict[str, Any]] = None) -> Path:
        plan = self._npu_plan(workload_or_model, cfg=cfg)
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "npu_plan_json", "path": str(out)})
        return out

    def _torch_sync_device(self, device_name: str) -> None:
        if not TORCH_AVAILABLE or torch is None:
            return
        dn = str(device_name).lower()
        try:
            if dn.startswith("cuda") and hasattr(torch, "cuda"):
                torch.cuda.synchronize()
            elif dn.startswith("xpu") and hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
            elif dn.startswith("mps") and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            return

    def _npu_benchmark(self, workload_or_model: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        plan = self._npu_plan(workload_or_model, cfg=cfg_map)
        iterations = max(1, _as_int(cfg_map.get("iterations", 6)))
        size = max(8, _as_int(cfg_map.get("size", cfg_map.get("matrix_size", 96))))
        warmup = max(0, _as_int(cfg_map.get("warmup", 1)))
        real_torch = bool(cfg_map.get("real_torch", True))
        execution_device = str(plan.get("execution_device", "cpu"))
        benchmark: Dict[str, Any] = {
            "ok": True,
            "plan": copy.deepcopy(plan),
            "iterations": iterations,
            "size": size,
            "device": execution_device,
            "real_run": False,
            "latency_ms_avg": None,
            "throughput_ops_per_s": None,
            "samples": [],
            "created_at": self._now_iso(),
        }
        if real_torch and TORCH_AVAILABLE and torch is not None and execution_device in {"cpu", "cuda", "mps", "xpu"}:
            try:
                dev = torch.device(execution_device)
                x = torch.randn(size, size, device=dev)
                y = torch.randn(size, size, device=dev)
                for _ in range(warmup):
                    _ = x @ y
                self._torch_sync_device(execution_device)
                samples: List[float] = []
                for _ in range(iterations):
                    t0 = time.time()
                    _ = x @ y
                    self._torch_sync_device(execution_device)
                    samples.append((time.time() - t0) * 1000.0)
                avg_ms = sum(samples) / max(1, len(samples))
                flops_est = 2.0 * (size ** 3)
                benchmark["samples"] = [round(v, 4) for v in samples]
                benchmark["latency_ms_avg"] = round(avg_ms, 4)
                benchmark["throughput_ops_per_s"] = round((flops_est / max(1e-9, avg_ms / 1000.0)), 2)
                benchmark["real_run"] = True
            except Exception as exc:
                benchmark["real_run"] = False
                benchmark["fallback_reason"] = f"torch_benchmark_failed:{exc}"
        if not benchmark["real_run"]:
            score = float(plan.get("estimated_throughput_score", 1.0) or 1.0)
            base_ms = (size * size) / 2200.0
            if str(plan.get("optimize_for")) == "latency":
                base_ms *= 0.85
            jitter = [1.0 + (0.03 * math.sin(i + size)) for i in range(iterations)]
            samples2 = [base_ms * j / max(0.25, score) for j in jitter]
            avg_ms = sum(samples2) / max(1, len(samples2))
            benchmark["samples"] = [round(v, 4) for v in samples2]
            benchmark["latency_ms_avg"] = round(avg_ms, 4)
            benchmark["throughput_ops_per_s"] = round((2.0 * (size ** 3)) / max(1e-9, avg_ms / 1000.0), 2)
        self._record_npu_op({"type": "npu_benchmark", **copy.deepcopy(benchmark)}, bucket="benchmarks", last_key="last_benchmark")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "npu_benchmark", "device": execution_device, "real": bool(benchmark.get("real_run"))})
        return benchmark

    def _npu_benchmark_json(self, rel_path: str, workload_or_model: Any, cfg: Optional[Dict[str, Any]] = None) -> Path:
        payload = self._npu_benchmark(workload_or_model, cfg=cfg)
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "npu_benchmark_json", "path": str(out)})
        return out

    def _npu_runtime_export_json(self, rel_path: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        probe = self._accelerator_probe(refresh=bool(cfg_map.get("refresh_probe", False)), deep=bool(cfg_map.get("deep_probe", False)))
        payload = {
            "project": self.project.name,
            "generated_at": self._now_iso(),
            "probe": probe,
            "providers": self._npu_detect_providers(probe),
            "profiles": copy.deepcopy((self.npu_state.get("profiles") or {})),
            "runtime_hints": {
                "recommended_device": probe.get("recommended_device"),
                "env": {k: os.environ.get(k) for k in sorted(os.environ.keys()) if str(k).upper().startswith(("NPU_", "NEURON_", "HAILO_", "OPENVINO_", "QNN_", "RKNN_"))},
            },
        }
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._record_npu_op({"type": "npu_runtime_export_json", "ok": True, "path": str(out)})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "npu_runtime_export_json", "path": str(out)})
        return out

    def _npu_torch_train(self, model_name: str, dataset_name: str, epochs: int = 1, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        if not TORCH_AVAILABLE:
            reason = TORCH_IMPORT_ERROR or "torch_missing"
            self._record_unsupported_step("npu_torch_train", [model_name, dataset_name, epochs, cfg_map], reason)
            result = {"ok": False, "reason": reason, "backend": "pytorch", "model": model_name, "dataset": dataset_name}
            self._record_npu_op({"type": "npu_torch_train", **result}, bucket="runs", last_key="last_run")
            return result
        plan_cfg = copy.deepcopy(cfg_map)
        plan_cfg.setdefault("task", "training")
        plan_cfg.setdefault("dataset", dataset_name)
        plan = self._npu_plan(model_name, cfg=plan_cfg)
        device_pref = str(cfg_map.get("device", plan.get("execution_device", "cpu")))
        lr = float(cfg_map.get("lr", self.models.get(model_name, {}).get("lr", 1e-3)))
        batch_size = _as_int(cfg_map.get("batch_size", self.datasets.get(dataset_name, {}).get("batch_size", 0))) if cfg_map.get("batch_size", self.datasets.get(dataset_name, {}).get("batch_size")) is not None else None
        try:
            self._torch_train(model_name, dataset_name, epochs=max(1, int(epochs)), lr=lr, batch_size=batch_size, device_pref=device_pref)
            success = True
            error_text = None
        except Exception as exc:
            success = False
            error_text = str(exc)
        train_meta = None
        with self._lock:
            if self.training_runs:
                train_meta = copy.deepcopy(self.training_runs[-1])
            tick_now = self.tick_count
        result = {
            "ok": bool(success),
            "type": "npu_torch_train",
            "model": model_name,
            "dataset": dataset_name,
            "epochs": max(1, int(epochs)),
            "device_requested": device_pref,
            "plan": copy.deepcopy(plan),
            "used_fallback": bool(plan.get("used_fallback", False)),
            "backend": "pytorch",
            "training_run": train_meta,
        }
        if error_text:
            result["error"] = error_text
        self._record_npu_op({"type": "npu_torch_train", **result}, bucket="runs", last_key="last_run")
        self._append_event({"tick": tick_now, "event": "npu_torch_train", "ok": bool(success), "device": str(device_pref)})
        return result

    def _npu_plan_json_step_arg(self, workload_or_model: Any) -> Any:
        # helper exists only to keep pipeline dispatch simple and AST trace side-effect free
        return workload_or_model

    def _photo_seed_rng(self, seed_hint: Any) -> random.Random:
        seed_src = f"{self._base_seed}|{seed_hint}"
        digest = hashlib.sha256(seed_src.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") & 0xFFFFFFFF
        return random.Random(seed)

    def _photo_default_palette(self, rng: random.Random) -> List[str]:
        palettes = [
            ["#0ea5e9", "#22d3ee", "#a7f3d0", "#fef08a"],
            ["#fb7185", "#f59e0b", "#facc15", "#84cc16"],
            ["#60a5fa", "#a78bfa", "#f472b6", "#fde68a"],
            ["#34d399", "#14b8a6", "#0ea5e9", "#38bdf8"],
        ]
        return list(palettes[rng.randrange(len(palettes))])

    def _photo_generate(self, rel_path: str, width: int, height: int, prompt_or_cfg: Any = None) -> Dict[str, Any]:
        out = self._resolve_output_path(rel_path)
        w = max(16, min(4096, int(width)))
        h = max(16, min(4096, int(height)))
        cfg: Dict[str, Any] = {}
        prompt = None
        if isinstance(prompt_or_cfg, dict):
            cfg = copy.deepcopy(prompt_or_cfg)
            prompt = str(cfg.get("prompt", cfg.get("text", ""))).strip() or None
        elif prompt_or_cfg is not None:
            prompt = str(prompt_or_cfg)
        seed_hint = cfg.get("seed", f"{out.name}|{w}x{h}|{prompt or ''}")
        rng = self._photo_seed_rng(seed_hint)
        palette = [str(c) for c in (cfg.get("palette") or self._photo_default_palette(rng))]
        style = str(cfg.get("style", "abstract")).lower()
        fmt = out.suffix.lower().lstrip(".") or "svg"
        if fmt not in {"svg", "ppm"}:
            fmt = "svg"
            out = out.with_suffix(".svg")

        if fmt == "ppm":
            rows: List[str] = [f"P3\n{w} {h}\n255"]
            for y in range(h):
                vals = []
                for x in range(w):
                    nx = x / max(1, w - 1)
                    ny = y / max(1, h - 1)
                    wave = math.sin((nx * 7.3 + ny * 4.9 + rng.random() * 0.3) * math.pi)
                    r = int(max(0, min(255, 40 + 180 * nx + 30 * wave)))
                    g = int(max(0, min(255, 50 + 150 * ny + 25 * math.cos((nx + ny) * math.pi * 3))))
                    b = int(max(0, min(255, 80 + 120 * (1 - nx) + 35 * math.sin(ny * math.pi * 6))))
                    vals.extend([str(r), str(g), str(b)])
                rows.append(" ".join(vals))
            out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        else:
            bg1 = palette[0] if palette else "#0ea5e9"
            bg2 = palette[1] if len(palette) > 1 else "#22d3ee"
            bg3 = palette[2] if len(palette) > 2 else "#a7f3d0"
            shapes_n = max(4, min(48, _as_int(cfg.get("shapes", 14))))
            caption = py_html.escape(str(cfg.get("caption", prompt or "NexusFlow Photo Tool")))
            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                "<defs>",
                f'<linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{bg1}"/><stop offset="55%" stop-color="{bg2}"/><stop offset="100%" stop-color="{bg3}"/></linearGradient>',
                "</defs>",
                f'<rect width="{w}" height="{h}" fill="url(#g)"/>',
            ]
            for i in range(shapes_n):
                x = rng.uniform(0, w)
                y = rng.uniform(0, h)
                rx = rng.uniform(max(8, w * 0.03), max(12, w * 0.18))
                ry = rng.uniform(max(8, h * 0.03), max(12, h * 0.18))
                rot = rng.uniform(0, 360)
                col = palette[i % len(palette)] if palette else "#ffffff"
                opacity = 0.08 + (i % 5) * 0.05
                if style == "geometric" or (style == "abstract" and i % 2 == 0):
                    parts.append(
                        f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{col}" opacity="{opacity:.2f}" transform="rotate({rot:.1f} {x:.1f} {y:.1f})"/>'
                    )
                else:
                    x2 = rng.uniform(0, w)
                    y2 = rng.uniform(0, h)
                    sw = rng.uniform(1.0, 4.0)
                    parts.append(
                        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{sw:.1f}" opacity="{opacity:.2f}" stroke-linecap="round"/>'
                    )
            parts.append(f'<rect x="{w*0.03:.1f}" y="{h*0.78:.1f}" width="{w*0.94:.1f}" height="{h*0.17:.1f}" rx="12" fill="rgba(0,0,0,0.28)"/>')
            parts.append(f'<text x="{w*0.05:.1f}" y="{h*0.88:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="{max(12, int(min(w,h)*0.045))}" fill="#ffffff">{caption}</text>')
            if prompt:
                prompt_txt = py_html.escape(prompt)[:220]
                parts.append(f'<text x="{w*0.05:.1f}" y="{h*0.94:.1f}" font-family="Consolas, monospace" font-size="{max(10, int(min(w,h)*0.027))}" fill="#e2e8f0">{prompt_txt}</text>')
            parts.append("</svg>")
            out.write_text("\n".join(parts), encoding="utf-8")

        meta = {
            "path": str(out),
            "name": out.name,
            "format": fmt,
            "width": w,
            "height": h,
            "prompt": prompt,
            "style": style,
            "palette": palette,
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.photo_state["images"][str(out)] = copy.deepcopy(meta)
        self._record_photo_op({"type": "photo_generate", "path": str(out), "format": fmt, "width": w, "height": h})
        return meta

    def _photo_find_meta(self, path_or_name: Any) -> Optional[Dict[str, Any]]:
        key = str(path_or_name)
        with self._lock:
            imgs = copy.deepcopy(self.photo_state.get("images", {}))
        if key in imgs:
            return imgs[key]
        for pth, meta in imgs.items():
            if Path(str(pth)).name == key:
                return meta
        return None

    def _photo_batch_find(self, name_or_path: Any) -> Optional[Dict[str, Any]]:
        key = str(name_or_path)
        with self._lock:
            batches = copy.deepcopy(self.photo_state.get("batches", {}))
        if key in batches:
            return batches[key]
        for pth, meta in batches.items():
            if Path(str(pth)).name == key or str(meta.get("name", "")) == key:
                return meta
        return None

    def _photo_data_uri(self, path_value: str | Path) -> Optional[str]:
        p = Path(path_value)
        if not p.exists() or not p.is_file():
            return None
        sfx = p.suffix.lower()
        if sfx not in {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return None
        mime = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(sfx, "application/octet-stream")
        try:
            payload = base64.b64encode(p.read_bytes()).decode("ascii")
        except Exception:
            return None
        return f"data:{mime};base64,{payload}"

    def _photo_normalize_sources(self, items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            raise RuntimeErrorNF("photo_collage/contact_sheet expects a list of photo refs")
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                ref = item.get("path", item.get("file", item.get("ref", item.get("name", f"item_{idx+1}"))))
                title = item.get("title", item.get("label"))
                subtitle = item.get("subtitle")
                normalized.append({"ref": str(ref), "title": None if title is None else str(title), "subtitle": None if subtitle is None else str(subtitle)})
            else:
                normalized.append({"ref": str(item), "title": None, "subtitle": None})
        return normalized

    def _photo_batch_generate(self, dir_rel: str, width: int, height: int, prompts: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not isinstance(prompts, list):
            raise RuntimeErrorNF("photo_batch_generate(dir, w, h, prompts[, cfg]) expects prompts list")
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        out_dir = self._resolve_output_path(dir_rel)
        out_dir.mkdir(parents=True, exist_ok=True)
        w = max(16, min(4096, int(width)))
        h = max(16, min(4096, int(height)))
        fmt = str(cfg_map.get("format", "svg")).lower().strip(".")
        if fmt not in {"svg", "ppm"}:
            fmt = "svg"
        prefix = str(cfg_map.get("prefix", "photo"))
        zero_pad = max(1, min(6, _as_int(cfg_map.get("zero_pad", 2))))
        start_index = max(0, _as_int(cfg_map.get("start_index", 1)))
        style_cycle = [str(s) for s in cfg_map.get("style_cycle", [])] if isinstance(cfg_map.get("style_cycle"), list) else []
        batch_name = str(cfg_map.get("name", out_dir.name))

        created: List[Dict[str, Any]] = []
        for i, prompt_item in enumerate(prompts):
            prompt_cfg: Dict[str, Any] = {}
            prompt_text: Optional[str] = None
            filename_stub: Optional[str] = None
            if isinstance(prompt_item, dict):
                prompt_cfg = copy.deepcopy(prompt_item)
                prompt_text = None if prompt_cfg.get("prompt") is None and prompt_cfg.get("text") is None else str(prompt_cfg.get("prompt", prompt_cfg.get("text", "")))
                filename_stub = str(prompt_cfg.get("name", "")).strip() or None
            elif prompt_item is not None:
                prompt_text = str(prompt_item)
            merged = {k: copy.deepcopy(v) for k, v in cfg_map.items() if k not in {"name", "prefix", "zero_pad", "start_index", "format", "style_cycle"}}
            merged.update(prompt_cfg)
            if style_cycle and "style" not in merged:
                merged["style"] = style_cycle[i % len(style_cycle)]
            if prompt_text and "prompt" not in merged and "text" not in merged:
                merged["prompt"] = prompt_text
            index_num = start_index + i
            default_stub = f"{prefix}_{index_num:0{zero_pad}d}"
            if filename_stub:
                default_stub = self._web_slugify(filename_stub) or default_stub
            out_path = out_dir / f"{default_stub}.{fmt}"
            meta = self._photo_generate(str(out_path), w, h, merged)
            created.append(copy.deepcopy(meta))

        batch_meta = {
            "name": batch_name,
            "dir": str(out_dir),
            "count": len(created),
            "width": w,
            "height": h,
            "format": fmt,
            "items": created,
            "created_at": self._now_iso(),
        }
        with self._lock:
            batches = self.photo_state.setdefault("batches", {})
            batches[str(out_dir)] = copy.deepcopy(batch_meta)
        self._record_photo_op({"type": "photo_batch_generate", "dir": str(out_dir), "count": len(created), "format": fmt})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "photo_batch_generate", "dir": str(out_dir), "count": len(created)})
        return batch_meta

    def _photo_poster(self, rel_path: str, width: int, height: int, title_or_cfg: Any = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        poster_cfg: Dict[str, Any] = {}
        if isinstance(cfg, dict):
            poster_cfg.update(copy.deepcopy(cfg))
        if isinstance(title_or_cfg, dict):
            poster_cfg.update(copy.deepcopy(title_or_cfg))
        elif title_or_cfg is not None:
            poster_cfg["title"] = str(title_or_cfg)
        title = str(poster_cfg.get("title", poster_cfg.get("caption", "NexusFlow Poster")))
        subtitle = str(poster_cfg.get("subtitle", poster_cfg.get("prompt", ""))).strip() or None
        style = str(poster_cfg.get("style", "geometric"))
        generated = self._photo_generate(
            rel_path,
            width,
            height,
            {
                **poster_cfg,
                "style": style,
                "caption": title,
                "prompt": subtitle or poster_cfg.get("prompt") or "",
                "shapes": poster_cfg.get("shapes", 22),
            },
        )
        meta = self._photo_find_meta(generated.get("path", rel_path)) or generated
        if isinstance(meta, dict):
            meta2 = copy.deepcopy(meta)
            meta2["kind"] = "poster"
            meta2["title"] = title
            if subtitle:
                meta2["subtitle"] = subtitle
            with self._lock:
                self.photo_state["images"][str(meta2["path"])] = copy.deepcopy(meta2)
            self._record_photo_op({"type": "photo_poster", "path": str(meta2["path"]), "width": width, "height": height})
            return meta2
        return generated

    def _photo_collage(self, rel_path: str, photos: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        items = self._photo_normalize_sources(photos)
        if not items:
            raise RuntimeErrorNF("photo_collage/contact_sheet requires at least one item")
        out = self._resolve_output_path(rel_path)
        if out.suffix.lower() != ".svg":
            out = out.with_suffix(".svg")
        columns = max(1, _as_int(cfg_map.get("columns", math.ceil(math.sqrt(len(items))))))
        gap = max(0, _as_int(cfg_map.get("gap", 16)))
        pad = max(0, _as_int(cfg_map.get("padding", 20)))
        tile_w = max(80, _as_int(cfg_map.get("tile_width", cfg_map.get("thumb_width", 220))))
        tile_h = max(80, _as_int(cfg_map.get("tile_height", cfg_map.get("thumb_height", 150))))
        label_h = max(24, _as_int(cfg_map.get("label_height", 44)))
        rows = max(1, math.ceil(len(items) / columns))
        w = max(tile_w + pad * 2, _as_int(cfg_map.get("width", pad * 2 + columns * tile_w + (columns - 1) * gap)))
        h = max(tile_h + label_h + pad * 2, _as_int(cfg_map.get("height", pad * 2 + rows * (tile_h + label_h) + (rows - 1) * gap)))
        bg = str(cfg_map.get("background", "#0b1020"))
        title = py_html.escape(str(cfg_map.get("title", "NexusFlow Photo Collage")))
        subtitle = py_html.escape(str(cfg_map.get("subtitle", f"{len(items)} items")))

        parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            "<defs>",
            "<filter id='shadow' x='-20%' y='-20%' width='140%' height='140%'>"
            "<feDropShadow dx='0' dy='3' stdDeviation='4' flood-color='#000000' flood-opacity='0.35'/>"
            "</filter>",
            "</defs>",
            f"<rect width='{w}' height='{h}' fill='{py_html.escape(bg)}'/>",
            f"<text x='{pad}' y='{pad}' fill='#e2e8f0' font-family='Segoe UI, sans-serif' font-size='16' dominant-baseline='hanging'>{title}</text>",
            f"<text x='{pad}' y='{pad+20}' fill='#94a3b8' font-family='Segoe UI, sans-serif' font-size='11' dominant-baseline='hanging'>{subtitle}</text>",
        ]

        source_refs: List[str] = []
        for idx, item in enumerate(items):
            col = idx % columns
            row = idx // columns
            x = pad + col * (tile_w + gap)
            y = pad + 40 + row * (tile_h + label_h + gap)
            ref = str(item.get("ref", ""))
            source_refs.append(ref)
            p = self._resolve_runtime_path(ref)
            meta = self._photo_find_meta(ref)
            title_txt = item.get("title") or (meta.get("name") if isinstance(meta, dict) else p.name)
            subtitle_txt = item.get("subtitle") or (meta.get("prompt") if isinstance(meta, dict) else None) or (meta.get("style") if isinstance(meta, dict) else None) or ""
            palette = (meta.get("palette") if isinstance(meta, dict) else None) or ["#334155", "#475569", "#64748b"]
            if not isinstance(palette, list) or not palette:
                palette = ["#334155", "#475569", "#64748b"]
            c1 = str(palette[0])
            c2 = str(palette[1] if len(palette) > 1 else palette[0])
            c3 = str(palette[2] if len(palette) > 2 else palette[-1])
            tile_id = f"g{idx}"
            parts.append(f"<defs><linearGradient id='{tile_id}' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='{c1}'/><stop offset='55%' stop-color='{c2}'/><stop offset='100%' stop-color='{c3}'/></linearGradient></defs>")
            parts.append(f"<g filter='url(#shadow)'><rect x='{x}' y='{y}' width='{tile_w}' height='{tile_h+label_h}' rx='12' fill='#111827' stroke='rgba(255,255,255,0.08)'/></g>")
            parts.append(f"<rect x='{x+8}' y='{y+8}' width='{tile_w-16}' height='{tile_h-16}' rx='8' fill='url(#{tile_id})'/>")

            data_uri = self._photo_data_uri(p)
            if data_uri and p.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                parts.append(
                    f"<image x='{x+8}' y='{y+8}' width='{tile_w-16}' height='{tile_h-16}' preserveAspectRatio='xMidYMid slice' href='{data_uri}' clip-path='inset(0 round 8px)'/>"
                )
            else:
                # decorative placeholder overlays when the format is unsupported for embedding (e.g., ppm)
                for j in range(6):
                    ox = x + 12 + (j * (tile_w - 24) / 6.0)
                    oy = y + 12 + ((j % 3) * (tile_h - 28) / 3.0)
                    parts.append(f"<circle cx='{ox:.1f}' cy='{oy:.1f}' r='{6 + (j%3)*3}' fill='{py_html.escape(str(palette[j % len(palette)]))}' opacity='0.32'/>")
                parts.append(f"<text x='{x + tile_w/2:.1f}' y='{y + tile_h/2:.1f}' text-anchor='middle' fill='#e5e7eb' font-family='Consolas, monospace' font-size='12'>{py_html.escape(p.suffix.lower() or '.img')}</text>")
            parts.append(f"<text x='{x+10}' y='{y+tile_h+16}' fill='#e5e7eb' font-family='Segoe UI, sans-serif' font-size='12' dominant-baseline='hanging'>{py_html.escape(str(title_txt or p.name))[:42]}</text>")
            if subtitle_txt:
                parts.append(f"<text x='{x+10}' y='{y+tile_h+31}' fill='#94a3b8' font-family='Consolas, monospace' font-size='10' dominant-baseline='hanging'>{py_html.escape(str(subtitle_txt))[:50]}</text>")

        parts.append("</svg>")
        out.write_text("\n".join(parts), encoding="utf-8")
        meta_out = {
            "path": str(out),
            "name": out.name,
            "kind": "collage",
            "format": "svg",
            "width": int(w),
            "height": int(h),
            "count": len(items),
            "columns": columns,
            "rows": rows,
            "sources": source_refs,
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.photo_state["images"][str(out)] = copy.deepcopy(meta_out)
        self._record_photo_op({"type": "photo_collage", "path": str(out), "count": len(items), "columns": columns})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "photo_collage", "path": str(out), "count": len(items)})
        return meta_out

    def _photo_parse_ppm_p3(self, path_value: str | Path) -> Dict[str, Any]:
        p = Path(path_value)
        text = p.read_text(encoding="utf-8", errors="replace")
        cleaned = re.sub(r"#.*", "", text)
        toks = cleaned.split()
        if len(toks) < 4 or toks[0] != "P3":
            raise RuntimeErrorNF(f"photo_filter only supports ASCII PPM P3 for pixel filters: {p}")
        try:
            w = int(toks[1]); h = int(toks[2]); maxv = int(toks[3])
        except Exception as exc:
            raise RuntimeErrorNF(f"Invalid PPM header: {p}") from exc
        if w <= 0 or h <= 0:
            raise RuntimeErrorNF(f"Invalid PPM dimensions: {p}")
        vals = []
        for tok in toks[4:]:
            try:
                vals.append(int(tok))
            except Exception:
                vals.append(0)
        need = w * h * 3
        if len(vals) < need:
            vals.extend([0] * (need - len(vals)))
        vals = vals[:need]
        return {"width": w, "height": h, "maxv": max(1, maxv), "pixels": vals}

    def _photo_write_ppm_p3(self, path_value: str | Path, width: int, height: int, pixels: List[int], maxv: int = 255) -> None:
        p = Path(path_value)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [f"P3\n{int(width)} {int(height)}\n{int(max(1, maxv))}"]
        line_parts: List[str] = []
        for i, v in enumerate(pixels):
            line_parts.append(str(int(max(0, min(maxv, v)))))
            if len(line_parts) >= 18:
                rows.append(" ".join(line_parts))
                line_parts = []
        if line_parts:
            rows.append(" ".join(line_parts))
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _photo_filter(self, src_rel: str, dst_rel: str, mode: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        src = self._resolve_runtime_path(src_rel)
        dst = self._resolve_output_path(dst_rel)
        if not src.exists() or not src.is_file():
            raise RuntimeErrorNF(f"photo_filter source missing: {src}")
        mode_l = str(mode or "grayscale").strip().lower()
        if src.suffix.lower() == ".ppm" and dst.suffix.lower() in {".ppm", ""}:
            ppm = self._photo_parse_ppm_p3(src)
            px = list(ppm["pixels"])
            maxv = int(ppm["maxv"])
            factor = float(cfg_map.get("factor", cfg_map.get("amount", 1.2)))
            contrast = float(cfg_map.get("contrast", cfg_map.get("factor", 1.2)))
            levels = max(2, min(64, _as_int(cfg_map.get("levels", 4))))
            for i in range(0, len(px), 3):
                r, g, b = px[i], px[i + 1], px[i + 2]
                if mode_l in {"invert", "negative"}:
                    r, g, b = maxv - r, maxv - g, maxv - b
                elif mode_l in {"grayscale", "greyscale"}:
                    y = int(round(0.299 * r + 0.587 * g + 0.114 * b))
                    r = g = b = y
                elif mode_l == "sepia":
                    nr = int(round(0.393 * r + 0.769 * g + 0.189 * b))
                    ng = int(round(0.349 * r + 0.686 * g + 0.168 * b))
                    nb = int(round(0.272 * r + 0.534 * g + 0.131 * b))
                    r, g, b = nr, ng, nb
                elif mode_l == "brightness":
                    r, g, b = int(r * factor), int(g * factor), int(b * factor)
                elif mode_l == "contrast":
                    def _adj(v: int) -> int:
                        return int((v - 128) * contrast + 128)
                    r, g, b = _adj(r), _adj(g), _adj(b)
                elif mode_l == "posterize":
                    step = max(1, maxv // max(1, levels - 1))
                    def _post(v: int) -> int:
                        return int(round(v / step) * step)
                    r, g, b = _post(r), _post(g), _post(b)
                elif mode_l == "threshold":
                    thr = max(0, min(maxv, _as_int(cfg_map.get("threshold", maxv // 2))))
                    y = int(round(0.299 * r + 0.587 * g + 0.114 * b))
                    r = g = b = maxv if y >= thr else 0
                else:
                    raise RuntimeErrorNF(f"Unsupported photo_filter mode for PPM: {mode_l}")
                px[i], px[i + 1], px[i + 2] = [max(0, min(maxv, int(r))), max(0, min(maxv, int(g))), max(0, min(maxv, int(b)))]
            if dst.suffix == "":
                dst = dst.with_suffix(".ppm")
            self._photo_write_ppm_p3(dst, int(ppm["width"]), int(ppm["height"]), px, maxv=maxv)
            out_meta = {
                "path": str(dst),
                "name": dst.name,
                "format": "ppm",
                "width": int(ppm["width"]),
                "height": int(ppm["height"]),
                "kind": "photo_filter",
                "filter": mode_l,
                "source": str(src),
                "created_at": self._now_iso(),
            }
        else:
            # SVG (or other embeddable) wrapper-filter output for portability
            if dst.suffix.lower() != ".svg":
                dst = dst.with_suffix(".svg")
            meta_src = self._photo_find_meta(str(src)) or self._photo_find_meta(src.name)
            w = _as_int(cfg_map.get("width", (meta_src or {}).get("width", 640)))
            h = _as_int(cfg_map.get("height", (meta_src or {}).get("height", 360)))
            uri = self._photo_data_uri(src)
            if not uri:
                # fallback: wrapper with textual placeholder if source can't be embedded
                uri = None
            filter_css = {
                "grayscale": "grayscale(1)",
                "greyscale": "grayscale(1)",
                "invert": "invert(1)",
                "negative": "invert(1)",
                "sepia": "sepia(1)",
                "brightness": f"brightness({float(cfg_map.get('factor', 1.2))})",
                "contrast": f"contrast({float(cfg_map.get('factor', 1.25))})",
                "saturate": f"saturate({float(cfg_map.get('factor', 1.6))})",
                "blur": f"blur({float(cfg_map.get('px', cfg_map.get('radius', 2.0)))}px)",
            }.get(mode_l)
            if filter_css is None:
                raise RuntimeErrorNF(f"Unsupported photo_filter mode for wrapper SVG: {mode_l}")
            title = py_html.escape(str(cfg_map.get("title", f"{mode_l.title()} Filter")))
            src_label = py_html.escape(src.name)
            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                f"<rect width='{w}' height='{h}' fill='#0b1020'/>",
            ]
            if uri:
                parts.append(f"<image x='0' y='0' width='{w}' height='{h}' href='{uri}' preserveAspectRatio='xMidYMid slice' style='filter:{filter_css};'/>")
            else:
                parts.append(f"<rect x='12' y='12' width='{w-24}' height='{h-24}' rx='10' fill='#1f2937' stroke='#334155'/>")
                parts.append(f"<text x='{w/2:.1f}' y='{h/2-8:.1f}' text-anchor='middle' fill='#e5e7eb' font-family='Segoe UI, sans-serif' font-size='14'>Filtered Preview Placeholder</text>")
                parts.append(f"<text x='{w/2:.1f}' y='{h/2+14:.1f}' text-anchor='middle' fill='#94a3b8' font-family='Consolas, monospace' font-size='11'>{src_label}</text>")
            parts.append(f"<rect x='12' y='{h-40}' width='{w-24}' height='28' rx='8' fill='rgba(2,6,23,.55)'/>")
            parts.append(f"<text x='20' y='{h-22}' fill='#e2e8f0' font-family='Segoe UI, sans-serif' font-size='12'>{title} | {py_html.escape(mode_l)} | {src_label}</text>")
            parts.append("</svg>")
            dst.write_text("\n".join(parts), encoding="utf-8")
            out_meta = {
                "path": str(dst),
                "name": dst.name,
                "format": "svg",
                "width": int(w),
                "height": int(h),
                "kind": "photo_filter",
                "filter": mode_l,
                "source": str(src),
                "created_at": self._now_iso(),
            }
        with self._lock:
            self.photo_state["images"][str(dst)] = copy.deepcopy(out_meta)
        self._record_photo_op({"type": "photo_filter", "source": str(src), "path": str(dst), "mode": mode_l})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "photo_filter", "path": str(dst), "mode": mode_l})
        return out_meta

    def _data_chart_svg(self, rel_path: str, values: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = self._resolve_output_path(rel_path)
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        vals: List[float] = []
        labels: List[str] = []
        if isinstance(values, dict) and isinstance(values.get("values"), list):
            values = values.get("values")
        if not isinstance(values, list):
            raise RuntimeErrorNF("data_chart_svg(path, values[, cfg]) requires a list of values")
        for i, v in enumerate(values):
            if isinstance(v, dict):
                vals.append(float(v.get("y", 0)))
                labels.append(str(v.get("x", i)))
            else:
                vals.append(float(v))
                labels.append(str(i))
        w = max(240, min(2200, _as_int(cfg_map.get("width", 760))))
        h = max(180, min(1600, _as_int(cfg_map.get("height", 360))))
        mode = str(cfg_map.get("mode", "line")).lower()
        title = py_html.escape(str(cfg_map.get("title", "NexusFlow Data Chart")))
        if not vals:
            vals = [0.0]
            labels = ["0"]
        vmin = min(vals)
        vmax = max(vals)
        if math.isclose(vmin, vmax):
            vmax = vmin + 1.0
        pad_l, pad_r, pad_t, pad_b = 52, 18, 26, 42
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)

        def sx(i: int) -> float:
            if len(vals) <= 1:
                return pad_l + plot_w / 2
            return pad_l + (i * plot_w / (len(vals) - 1))

        def sy(v: float) -> float:
            return pad_t + (vmax - v) * plot_h / (vmax - vmin)

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            "<defs><linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>"
            "<stop offset='0%' stop-color='#0f172a'/><stop offset='100%' stop-color='#1e293b'/></linearGradient></defs>",
            f"<rect width='{w}' height='{h}' fill='url(#bg)' rx='14'/>",
            f"<text x='18' y='18' fill='#e2e8f0' font-family='Segoe UI, sans-serif' font-size='13' dominant-baseline='hanging'>{title}</text>",
        ]
        for t in range(5):
            v = vmin + (vmax - vmin) * (t / 4)
            y = sy(v)
            parts.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{w-pad_r}' y2='{y:.1f}' stroke='#334155' stroke-width='1'/>")
            parts.append(f"<text x='{pad_l-6}' y='{y:.1f}' text-anchor='end' fill='#94a3b8' font-family='Consolas, monospace' font-size='10' dominant-baseline='middle'>{v:.2f}</text>")
        parts.append(f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{h-pad_b}' stroke='#64748b' stroke-width='1.2'/>")
        parts.append(f"<line x1='{pad_l}' y1='{h-pad_b}' x2='{w-pad_r}' y2='{h-pad_b}' stroke='#64748b' stroke-width='1.2'/>")

        if mode == "bar":
            bw = max(6, plot_w / max(1, len(vals)) * 0.72)
            for i, v in enumerate(vals):
                x = pad_l + (i + 0.5) * plot_w / max(1, len(vals)) - bw / 2
                y = sy(v)
                bh = (h - pad_b) - y
                parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw:.1f}' height='{max(0.0,bh):.1f}' fill='#38bdf8' opacity='0.9' rx='4'/>")
        else:
            pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
            parts.append(f"<polyline points='{pts}' fill='none' stroke='#22d3ee' stroke-width='2.5' stroke-linejoin='round' stroke-linecap='round'/>")
            for i, v in enumerate(vals):
                parts.append(f"<circle cx='{sx(i):.1f}' cy='{sy(v):.1f}' r='2.8' fill='#f8fafc' stroke='#38bdf8' stroke-width='1.2'/>")
        step = max(1, len(labels) // 8)
        for i, lab in enumerate(labels):
            if i % step != 0 and i != len(labels) - 1:
                continue
            parts.append(f"<text x='{sx(i):.1f}' y='{h-14}' text-anchor='middle' fill='#94a3b8' font-family='Consolas, monospace' font-size='10'>{py_html.escape(str(lab))[:12]}</text>")
        parts.append("</svg>")
        out.write_text("\n".join(parts), encoding="utf-8")
        meta = {"path": str(out), "type": "data_chart_svg", "mode": mode, "points": len(vals), "width": w, "height": h}
        self._record_graph_op(meta)
        return meta

    def _graph_normalize(self, graph: Any, *, name: Optional[str] = None, directed_default: bool = False) -> Dict[str, Any]:
        if isinstance(graph, str):
            with self._lock:
                stored = copy.deepcopy(self.graph_state.get("graphs", {}).get(str(graph)))
            if not isinstance(stored, dict):
                raise RuntimeErrorNF(f"Unknown graph: {graph}")
            return stored
        if isinstance(graph, dict):
            directed = bool(graph.get("directed", directed_default))
            nodes_in = graph.get("nodes")
            edges_in = graph.get("edges", [])
            gname = str(graph.get("name", name or "graph"))
        else:
            directed = directed_default
            nodes_in = None
            edges_in = graph
            gname = str(name or "graph")
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        if isinstance(nodes_in, list):
            for n in nodes_in:
                if isinstance(n, dict):
                    nid = str(n.get("id", ""))
                    if not nid:
                        continue
                    nodes[nid] = {"id": nid, **{k: v for k, v in n.items() if k != "id"}}
                else:
                    nid = str(n)
                    nodes[nid] = {"id": nid}
        if not isinstance(edges_in, list):
            raise RuntimeErrorNF("Graph edges must be a list")
        for item in edges_in:
            src = None
            dst = None
            weight = 1.0
            meta: Dict[str, Any] = {}
            if isinstance(item, dict):
                src = item.get("source", item.get("src", item.get("from")))
                dst = item.get("target", item.get("dst", item.get("to")))
                if "weight" in item:
                    try:
                        weight = float(item.get("weight", 1.0))
                    except Exception:
                        weight = 1.0
                meta = {k: v for k, v in item.items() if k not in {"source", "src", "from", "target", "dst", "to", "weight"}}
            elif isinstance(item, list) and len(item) >= 2:
                src = item[0]
                dst = item[1]
                if len(item) >= 3:
                    try:
                        weight = float(item[2])
                    except Exception:
                        weight = 1.0
            else:
                continue
            if src is None or dst is None:
                continue
            s = str(src)
            t = str(dst)
            nodes.setdefault(s, {"id": s})
            nodes.setdefault(t, {"id": t})
            edges.append({"source": s, "target": t, "weight": weight, **meta})
        return {"name": gname, "directed": directed, "nodes": list(nodes.values()), "edges": edges}

    def _graph_store(self, name: str, graph: Any, directed: bool = False) -> Dict[str, Any]:
        normalized = self._graph_normalize(graph, name=name, directed_default=directed)
        with self._lock:
            self.graph_state["graphs"][str(name)] = copy.deepcopy(normalized)
        self._record_graph_op({"type": "graph_create", "name": str(name), "nodes": len(normalized["nodes"]), "edges": len(normalized["edges"])})
        return normalized

    def _graph_adj(self, g: Dict[str, Any]) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {}
        for n in g.get("nodes", []):
            if isinstance(n, dict) and "id" in n:
                adj.setdefault(str(n["id"]), [])
        directed = bool(g.get("directed", False))
        for e in g.get("edges", []):
            if not isinstance(e, dict):
                continue
            s = str(e.get("source"))
            t = str(e.get("target"))
            adj.setdefault(s, []).append(t)
            adj.setdefault(t, adj.get(t, []))
            if not directed:
                adj.setdefault(t, []).append(s)
        return adj

    def _graph_stats(self, graph: Any) -> Dict[str, Any]:
        g = self._graph_normalize(graph)
        adj = self._graph_adj(g)
        degrees = {node: len(nei) for node, nei in adj.items()}
        comps = self._graph_components(g)
        return {
            "name": g.get("name"),
            "directed": bool(g.get("directed", False)),
            "nodes": len(g.get("nodes", [])),
            "edges": len(g.get("edges", [])),
            "degree_min": min(degrees.values()) if degrees else 0,
            "degree_max": max(degrees.values()) if degrees else 0,
            "degree_avg": (sum(degrees.values()) / len(degrees)) if degrees else 0.0,
            "components": len(comps.get("components", [])),
            "isolated_nodes": [n for n, d in degrees.items() if d == 0],
        }

    def _graph_degrees(self, graph: Any) -> Dict[str, int]:
        g = self._graph_normalize(graph)
        adj = self._graph_adj(g)
        return {node: len(nei) for node, nei in adj.items()}

    def _graph_shortest_path(self, graph: Any, source: Any, target: Any) -> Dict[str, Any]:
        g = self._graph_normalize(graph)
        src = str(source)
        dst = str(target)
        adj = self._graph_adj(g)
        if src not in adj or dst not in adj:
            return {"ok": False, "reason": "node_missing", "source": src, "target": dst, "path": []}
        q: deque[str] = deque([src])
        prev: Dict[str, Optional[str]] = {src: None}
        while q:
            node = q.popleft()
            if node == dst:
                break
            for nei in adj.get(node, []):
                if nei not in prev:
                    prev[nei] = node
                    q.append(nei)
        if dst not in prev:
            return {"ok": False, "reason": "no_path", "source": src, "target": dst, "path": []}
        path: List[str] = []
        cur: Optional[str] = dst
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return {"ok": True, "source": src, "target": dst, "path": path, "hops": max(0, len(path) - 1)}

    def _graph_components(self, graph: Any) -> Dict[str, Any]:
        g = self._graph_normalize(graph)
        adj = self._graph_adj(g)
        seen: set[str] = set()
        components: List[List[str]] = []
        for node in adj.keys():
            if node in seen:
                continue
            comp: List[str] = []
            q: deque[str] = deque([node])
            seen.add(node)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nei in adj.get(cur, []):
                    if nei not in seen:
                        seen.add(nei)
                        q.append(nei)
            components.append(sorted(comp))
        components.sort(key=lambda c: (-len(c), c))
        return {"name": g.get("name"), "components": components}

    def _graph_export_svg(self, graph_name: str, rel_path: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        g = self._graph_normalize(str(graph_name))
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        out = self._resolve_output_path(rel_path)
        w = max(240, min(2400, _as_int(cfg_map.get("width", 900))))
        h = max(220, min(1800, _as_int(cfg_map.get("height", 620))))
        title = py_html.escape(str(cfg_map.get("title", g.get("name", graph_name))))
        nodes = [str((n or {}).get("id")) for n in g.get("nodes", []) if isinstance(n, dict) and "id" in n]
        n_count = len(nodes)
        cx = w / 2
        cy = h / 2 + 10
        radius = max(60.0, min(w, h) * 0.34)
        pos: Dict[str, tuple[float, float]] = {}
        for i, node in enumerate(nodes):
            angle = (2 * math.pi * i / max(1, n_count)) - math.pi / 2
            pos[node] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        deg = self._graph_degrees(g)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            f"<rect width='{w}' height='{h}' fill='#020617'/>",
            f"<text x='16' y='20' fill='#e2e8f0' font-family='Segoe UI, sans-serif' font-size='14'>{title}</text>",
        ]
        for e in g.get("edges", []):
            if not isinstance(e, dict):
                continue
            s = str(e.get("source"))
            t = str(e.get("target"))
            if s not in pos or t not in pos:
                continue
            x1, y1 = pos[s]
            x2, y2 = pos[t]
            parts.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#334155' stroke-width='1.4' opacity='0.9'/>")
        for node in nodes:
            x, y = pos[node]
            d = deg.get(node, 0)
            r = 8 + min(18, d * 1.8)
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='#38bdf8' stroke='#e0f2fe' stroke-width='1.2'/>")
            parts.append(f"<text x='{x:.1f}' y='{y + r + 14:.1f}' text-anchor='middle' fill='#cbd5e1' font-family='Consolas, monospace' font-size='10'>{py_html.escape(node)[:18]}</text>")
        parts.append("</svg>")
        out.write_text("\n".join(parts), encoding="utf-8")
        meta = {"type": "graph_export_svg", "name": str(graph_name), "path": str(out), "nodes": n_count, "edges": len(g.get("edges", []))}
        self._record_graph_op(meta)
        return meta

    def _graph_from_csv(self, name: str, rel_csv: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        src_col = str(cfg_map.get("source_col", "source"))
        dst_col = str(cfg_map.get("target_col", "target"))
        weight_col = cfg_map.get("weight_col")
        has_header = bool(cfg_map.get("has_header", True))
        delimiter = str(cfg_map.get("delimiter", ","))[:1] or ","
        p = self._resolve_runtime_path(rel_csv)
        if not p.exists():
            raise RuntimeErrorNF(f"CSV graph source missing: {p}")
        edges: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8", newline="") as fh:
            if has_header:
                reader = csv.DictReader(fh, delimiter=delimiter)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    src = row.get(src_col)
                    dst = row.get(dst_col)
                    if src is None or dst is None:
                        continue
                    edge: Dict[str, Any] = {"source": src, "target": dst}
                    if isinstance(weight_col, str) and weight_col in row and row.get(weight_col) not in {None, ""}:
                        try:
                            edge["weight"] = float(row.get(weight_col))
                        except Exception:
                            pass
                    edges.append(edge)
            else:
                reader2 = csv.reader(fh, delimiter=delimiter)
                for row in reader2:
                    if len(row) < 2:
                        continue
                    edge2: Dict[str, Any] = {"source": row[0], "target": row[1]}
                    if len(row) >= 3 and row[2] != "":
                        try:
                            edge2["weight"] = float(row[2])
                        except Exception:
                            pass
                    edges.append(edge2)
        return self._graph_store(str(name), {"name": str(name), "directed": bool(cfg_map.get("directed", False)), "edges": edges})

    def _graph_metrics_json(self, graph_name: str, rel_path: str) -> Path:
        payload = {
            "graph": str(graph_name),
            "stats": self._graph_stats(str(graph_name)),
            "degrees": self._graph_degrees(str(graph_name)),
            "components": self._graph_components(str(graph_name)),
        }
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._record_graph_op({"type": "graph_metrics_json", "name": str(graph_name), "path": str(out)})
        return out

    def _numeric_series(self, value: Any) -> List[float]:
        if not isinstance(value, list):
            raise RuntimeErrorNF("Expected list of numbers")
        vals: List[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                vals.append(float(item))
            elif isinstance(item, dict) and isinstance(item.get("y"), (int, float)):
                vals.append(float(item.get("y")))
            else:
                raise RuntimeErrorNF(f"Non-numeric item in list: {item!r}")
        return vals

    def _stats_summary(self, value: Any) -> Dict[str, Any]:
        vals = self._numeric_series(value)
        n = len(vals)
        if n == 0:
            return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "variance": None, "stddev": None}
        sorted_vals = sorted(vals)
        mid = n // 2
        median = sorted_vals[mid] if n % 2 == 1 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return {
            "count": n,
            "min": min(vals),
            "max": max(vals),
            "sum": sum(vals),
            "mean": mean,
            "median": median,
            "variance": var,
            "stddev": math.sqrt(var),
        }

    def _differentiate(self, xs: Any, ys: Any) -> List[float]:
        xvals = self._numeric_series(xs)
        yvals = self._numeric_series(ys)
        n = min(len(xvals), len(yvals))
        if n < 2:
            return []
        out: List[float] = []
        for i in range(1, n):
            dx = xvals[i] - xvals[i - 1]
            dy = yvals[i] - yvals[i - 1]
            out.append(0.0 if math.isclose(dx, 0.0) else dy / dx)
        return out

    def _integrate_trapz(self, ys: Any, xs_or_dx: Any = None) -> float:
        yvals = self._numeric_series(ys)
        if len(yvals) < 2:
            return 0.0
        if isinstance(xs_or_dx, list):
            xvals = self._numeric_series(xs_or_dx)
            n = min(len(xvals), len(yvals))
            if n < 2:
                return 0.0
            area = 0.0
            for i in range(1, n):
                dx = xvals[i] - xvals[i - 1]
                area += 0.5 * (yvals[i] + yvals[i - 1]) * dx
            return area
        dx = float(xs_or_dx) if isinstance(xs_or_dx, (int, float)) else 1.0
        return sum(0.5 * (yvals[i] + yvals[i - 1]) * dx for i in range(1, len(yvals)))

    def _poly_eval(self, coeffs: Any, x: Any) -> float:
        c = self._numeric_series(coeffs)
        xv = float(x)
        total = 0.0
        for i, coef in enumerate(c):
            total += coef * (xv ** i)
        return total

    def _linear_fit(self, xs: Any, ys: Any) -> Dict[str, Any]:
        xvals = self._numeric_series(xs)
        yvals = self._numeric_series(ys)
        n = min(len(xvals), len(yvals))
        if n == 0:
            return {"ok": False, "reason": "empty"}
        xvals = xvals[:n]
        yvals = yvals[:n]
        mx = sum(xvals) / n
        my = sum(yvals) / n
        sxx = sum((x - mx) ** 2 for x in xvals)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xvals, yvals))
        slope = 0.0 if math.isclose(sxx, 0.0) else sxy / sxx
        intercept = my - slope * mx
        ss_tot = sum((y - my) ** 2 for y in yvals)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xvals, yvals))
        r2 = 1.0 if math.isclose(ss_tot, 0.0) else max(-1.0, min(1.0, 1.0 - ss_res / ss_tot))
        return {"ok": True, "slope": slope, "intercept": intercept, "r2": r2, "count": n}

    def _convert_modes(self) -> List[str]:
        return [
            "copy",
            "json_pretty",
            "csv_to_json",
            "tsv_to_json",
            "json_to_csv",
            "json_to_tsv",
            "jsonl_to_json",
            "json_to_jsonl",
            "text_to_base64",
            "base64_to_text",
            "text_to_hex",
            "hex_to_text",
        ]

    def _convert_infer_mode(self, src: Path, dst: Path) -> str:
        sfx = src.suffix.lower()
        dsfx = dst.suffix.lower()
        src_name = src.name.lower()
        dst_name = dst.name.lower()
        if sfx == ".json" and dsfx == ".csv":
            return "json_to_csv"
        if sfx == ".json" and dsfx == ".tsv":
            return "json_to_tsv"
        if sfx == ".csv" and dsfx == ".json":
            return "csv_to_json"
        if sfx == ".tsv" and dsfx == ".json":
            return "tsv_to_json"
        if src_name.endswith(".jsonl") and dsfx == ".json":
            return "jsonl_to_json"
        if sfx == ".json" and dst_name.endswith(".jsonl"):
            return "json_to_jsonl"
        if src_name.endswith(".b64") or src_name.endswith(".base64"):
            return "base64_to_text"
        if dst_name.endswith(".b64") or dst_name.endswith(".base64"):
            return "text_to_base64"
        return "copy"

    def _file_convert(self, src_path: str, dst_path: str, mode: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        src = self._resolve_runtime_path(src_path)
        dst = self._resolve_output_path(dst_path)
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        if not src.exists():
            raise RuntimeErrorNF(f"Conversion source not found: {src}")
        selected = str(mode or "").strip().lower() or self._convert_infer_mode(src, dst)
        payload: Dict[str, Any] = {"ok": True, "src": str(src), "dst": str(dst), "mode": selected}

        if selected == "copy":
            shutil.copy2(src, dst)
        elif selected == "json_pretty":
            data = json.loads(src.read_text(encoding="utf-8"))
            dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
        elif selected in {"csv_to_json", "tsv_to_json"}:
            delim = "\t" if selected == "tsv_to_json" else ","
            with src.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh, delimiter=delim))
            dst.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            payload["rows"] = len(rows)
        elif selected in {"json_to_csv", "json_to_tsv"}:
            delim = "\t" if selected == "json_to_tsv" else ","
            data = json.loads(src.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise RuntimeErrorNF(f"{selected} expects top-level JSON list")
            rows = [row if isinstance(row, dict) else {"value": row} for row in data]
            fields: List[str] = []
            for row in rows:
                for k in row.keys():
                    if k not in fields:
                        fields.append(str(k))
            with dst.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields or ["value"], delimiter=delim)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k) for k in (fields or ["value"])})
            payload["rows"] = len(rows)
            payload["columns"] = len(fields)
        elif selected == "jsonl_to_json":
            rows2 = []
            for line in src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rows2.append(json.loads(line))
            dst.write_text(json.dumps(rows2, indent=2), encoding="utf-8")
            payload["rows"] = len(rows2)
        elif selected == "json_to_jsonl":
            data2 = json.loads(src.read_text(encoding="utf-8"))
            if not isinstance(data2, list):
                raise RuntimeErrorNF("json_to_jsonl expects top-level JSON list")
            text = "\n".join(json.dumps(item, ensure_ascii=False) for item in data2) + ("\n" if data2 else "")
            dst.write_text(text, encoding="utf-8")
            payload["rows"] = len(data2)
        elif selected == "text_to_base64":
            raw = src.read_bytes()
            dst.write_text(base64.b64encode(raw).decode("ascii"), encoding="utf-8")
        elif selected == "base64_to_text":
            raw_txt = src.read_text(encoding="utf-8").strip()
            decoded = base64.b64decode(raw_txt.encode("ascii"))
            try:
                dst.write_text(decoded.decode(str(cfg_map.get("encoding", "utf-8")), errors="replace"), encoding="utf-8")
            except Exception:
                dst.write_bytes(decoded)
        elif selected == "text_to_hex":
            raw = src.read_bytes()
            dst.write_text(raw.hex(), encoding="utf-8")
        elif selected == "hex_to_text":
            hex_txt = re.sub(r"\s+", "", src.read_text(encoding="utf-8"))
            raw = bytes.fromhex(hex_txt)
            try:
                dst.write_text(raw.decode(str(cfg_map.get("encoding", "utf-8")), errors="replace"), encoding="utf-8")
            except Exception:
                dst.write_bytes(raw)
        else:
            raise RuntimeErrorNF(f"Unsupported conversion mode: {selected}")

        try:
            payload["bytes"] = dst.stat().st_size
        except Exception:
            pass
        self._record_convert_op(payload)
        return payload

    def _exe_tool_info(self, refresh: bool = False) -> Dict[str, Any]:
        if not refresh:
            with self._lock:
                cached = copy.deepcopy(self.exe_state.get("tools"))
            if isinstance(cached, dict) and cached.get("checked_at"):
                return cached
        pyinstaller_path = shutil.which("pyinstaller")
        nuitka_path = shutil.which("nuitka")
        python_path = sys.executable or shutil.which("python") or shutil.which("python3")
        cpp_info = self._cpp_compiler_info()
        cs_info = self._csharp_compiler_info()
        rust_info = self._rust_toolchain_info()
        selected_python = "pyinstaller" if pyinstaller_path else ("nuitka" if nuitka_path else None)
        info = {
            "checked_at": self._now_iso(),
            "available": bool(pyinstaller_path or nuitka_path),
            "selected": {
                "python": selected_python,
                "cpp": (cpp_info or {}).get("command") if isinstance(cpp_info, dict) else None,
                "csharp": (cs_info or {}).get("command") if isinstance(cs_info, dict) else None,
                "rust": (rust_info or {}).get("flavor") if isinstance(rust_info, dict) else None,
            },
            "tools": {
                "python": python_path,
                "pyinstaller": pyinstaller_path,
                "nuitka": nuitka_path,
                "cpp": cpp_info,
                "csharp": cs_info,
                "rust": rust_info,
            },
            "host": {
                "is_windows": self._is_windows_host,
                "platform": py_platform.platform(),
            },
        }
        with self._lock:
            self.exe_state["tools"] = copy.deepcopy(info)
        return copy.deepcopy(info)

    def _exe_build_python_script(self, source: Path, out: Path, name: Optional[str], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        tool_info = self._exe_tool_info(refresh=bool(cfg_map.get("refresh_tools", False)))
        builder = str(cfg_map.get("builder") or cfg_map.get("tool") or "auto").lower()
        onefile = bool(cfg_map.get("onefile", True))
        windowed = bool(cfg_map.get("windowed", False))
        timeout_sec = float(cfg_map.get("timeout_sec", 300.0))
        dry_run = bool(cfg_map.get("dry_run", False))
        exe_name = str(cfg_map.get("name") or name or out.stem or source.stem)
        flags = [str(x) for x in (cfg_map.get("flags") or [])] if isinstance(cfg_map.get("flags"), list) else []

        pyinstaller_path = ((tool_info.get("tools") or {}).get("pyinstaller") if isinstance(tool_info, dict) else None) or None
        nuitka_path = ((tool_info.get("tools") or {}).get("nuitka") if isinstance(tool_info, dict) else None) or None
        if builder == "auto":
            builder = str((tool_info.get("selected") or {}).get("python") or "")
        if builder not in {"pyinstaller", "nuitka"}:
            builder = "pyinstaller" if pyinstaller_path else ("nuitka" if nuitka_path else "")
        if dry_run and builder not in {"pyinstaller", "nuitka"}:
            builder = "pyinstaller"

        ext = ".exe" if self._is_windows_host else ""
        if out.suffix.lower() != ext.lower() and ext:
            out = out.with_suffix(ext)

        command: List[str]
        expected_out = out
        if builder == "pyinstaller":
            if not pyinstaller_path and not dry_run:
                return {
                    "ok": False,
                    "reason": "tool_missing:pyinstaller",
                    "builder": "pyinstaller",
                    "source": str(source),
                    "out": str(out),
                }
            work_root = self._resolve_output_path(str(cfg_map.get("work_root", f"_exe_build/{self._web_slugify(exe_name)}")))
            build_dir = work_root / "build"
            spec_dir = work_root / "spec"
            build_dir.mkdir(parents=True, exist_ok=True)
            spec_dir.mkdir(parents=True, exist_ok=True)
            command = [str(pyinstaller_path or "pyinstaller"), "--noconfirm"]
            if bool(cfg_map.get("clean", True)):
                command.append("--clean")
            if onefile:
                command.append("--onefile")
            if windowed:
                command.append("--windowed")
            command.extend(["--name", exe_name, "--distpath", str(out.parent), "--workpath", str(build_dir), "--specpath", str(spec_dir)])
            command.extend(flags)
            command.append(str(source))
            expected_out = out.parent / (exe_name + ext)
        elif builder == "nuitka":
            py_cmd = (tool_info.get("tools") or {}).get("python") or sys.executable or "python"
            if not nuitka_path and not py_cmd and not dry_run:
                return {
                    "ok": False,
                    "reason": "tool_missing:nuitka",
                    "builder": "nuitka",
                    "source": str(source),
                    "out": str(out),
                }
            command = [str(py_cmd), "-m", "nuitka"]
            if onefile:
                command.append("--onefile")
            else:
                command.append("--standalone")
            if windowed:
                command.append("--windows-disable-console")
            command.extend([f"--output-dir={str(out.parent)}", f"--output-filename={out.name}"])
            command.extend(flags)
            command.append(str(source))
            expected_out = out
        else:
            return {
                "ok": False,
                "reason": "tool_missing",
                "builder": builder or None,
                "source": str(source),
                "out": str(out),
                "tools": tool_info,
            }

        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "builder": builder,
                "source": str(source),
                "out": str(out),
                "name": exe_name,
                "onefile": onefile,
                "windowed": windowed,
                "command": command,
                "exists": out.exists(),
                "mode": "python_script",
            }

        run = self._run_subprocess(command, timeout_sec=timeout_sec)
        built_path = expected_out if expected_out.exists() else (out if out.exists() else None)
        if built_path is None and out.parent.exists():
            for cand in sorted(out.parent.glob(f"{exe_name}*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
                if cand.is_file():
                    built_path = cand
                    break
        if built_path is not None and built_path.exists() and built_path.resolve() != out.resolve():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_path, out)
            built_path = out
        exists = out.exists() or (built_path is not None and built_path.exists())
        out_path_final = out if out.exists() else (built_path or out)
        size_bytes = 0
        if out_path_final.exists():
            try:
                size_bytes = int(out_path_final.stat().st_size)
            except Exception:
                size_bytes = 0
        result = {
            "ok": bool(run.get("ok")) and exists,
            "builder": builder,
            "source": str(source),
            "out": str(out_path_final),
            "name": exe_name,
            "onefile": onefile,
            "windowed": windowed,
            "command": command,
            "exists": exists,
            "bytes": size_bytes,
            "stdout": run.get("stdout", ""),
            "stderr": run.get("stderr", ""),
            "returncode": run.get("returncode"),
            "duration_ms": run.get("duration_ms"),
            "mode": "python_script",
        }
        if not bool(run.get("ok")):
            result["reason"] = f"build_failed:{builder}"
        elif not exists:
            result["ok"] = False
            result["reason"] = "output_missing"
        return result

    def _exe_build(self, source_or_module: str, out_rel: str, name: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        out = self._resolve_output_path(out_rel)
        if self._is_windows_host and not out.suffix:
            out = out.with_suffix(".exe")
        dry_run = bool(cfg_map.get("dry_run", False))
        target = str(source_or_module)

        with self._lock:
            module_meta = copy.deepcopy(self.lang_modules.get(target))

        if isinstance(module_meta, dict):
            lang = str(module_meta.get("language", "")).lower()
            result: Dict[str, Any]
            if dry_run:
                result = {
                    "ok": True,
                    "dry_run": True,
                    "mode": "module",
                    "module": target,
                    "language": lang,
                    "out": str(out),
                    "builder": f"module:{lang}",
                }
            elif lang == "cpp":
                build = self._cpp_build_module(target, out_path=str(out), cfg=cfg_map)
                result = {"mode": "module", "module": target, "language": lang, "builder": "cpp", "out": str(out), **build}
            elif lang == "csharp":
                build = self._csharp_build_module(target, out_path=str(out), cfg=cfg_map)
                result = {"mode": "module", "module": target, "language": lang, "builder": "csharp", "out": str(out), **build}
            elif lang == "rust":
                build = self._rust_build_module(target, out_path=str(out), cfg=cfg_map)
                result = {"mode": "module", "module": target, "language": lang, "builder": "rust", "out": str(out), **build}
            elif lang == "python":
                src = Path(str(module_meta.get("path", "")))
                result = self._exe_build_python_script(src, out, name=name or target, cfg=cfg_map)
                result.update({"module": target, "language": "python"})
            else:
                result = {
                    "ok": False,
                    "mode": "module",
                    "module": target,
                    "language": lang,
                    "out": str(out),
                    "reason": "unsupported_module_language",
                }
            if "out" not in result:
                result["out"] = str(out)
            if "module" not in result:
                result["module"] = target
            self._record_exe_op({"op": "build", **copy.deepcopy(result)})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "exe_build", "ok": bool(result.get("ok")), "target": target})
            return result

        source = self._resolve_runtime_path(target)
        if not source.exists():
            result2 = {"ok": False, "op": "build", "source": str(source), "out": str(out), "reason": "source_missing"}
            self._record_exe_op(result2)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "exe_build", "ok": False, "reason": "source_missing"})
            return {k: v for k, v in result2.items() if k != "op"}

        ext = source.suffix.lower()
        if ext == ".exe":
            if dry_run:
                result3 = {
                    "ok": True,
                    "dry_run": True,
                    "mode": "copy",
                    "source": str(source),
                    "out": str(out),
                    "builder": "copy",
                    "exists": out.exists(),
                }
            else:
                shutil.copy2(source, out)
                result3 = {
                    "ok": True,
                    "mode": "copy",
                    "source": str(source),
                    "out": str(out),
                    "builder": "copy",
                    "exists": out.exists(),
                    "bytes": (out.stat().st_size if out.exists() else 0),
                }
        elif ext == ".py":
            result3 = self._exe_build_python_script(source, out, name=name, cfg=cfg_map)
        else:
            result3 = {
                "ok": False,
                "mode": "path",
                "source": str(source),
                "out": str(out),
                "reason": "unsupported_source_type",
                "hint": "Register a rust/cpp/csharp module and call exe_build(module_name, ...), or use a .py script/.exe source.",
            }
        self._record_exe_op({"op": "build", **copy.deepcopy(result3)})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "exe_build", "ok": bool(result3.get("ok")), "source": str(source)})
        return result3

    def _dir_manifest(self, root_path: Path) -> Dict[str, Any]:
        if not root_path.exists():
            return {"ok": False, "path": str(root_path), "exists": False, "files": []}
        if root_path.is_file():
            return {
                "ok": True,
                "path": str(root_path),
                "exists": True,
                "is_file": True,
                "files": [{"path": root_path.name, "bytes": root_path.stat().st_size}],
                "total_bytes": root_path.stat().st_size,
            }
        entries = []
        total = 0
        for p in sorted(root_path.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(root_path)).replace("\\", "/")
                try:
                    size = int(p.stat().st_size)
                except Exception:
                    size = 0
                total += size
                entries.append({"path": rel, "bytes": size})
        return {"ok": True, "path": str(root_path), "exists": True, "files": entries, "count": len(entries), "total_bytes": total}

    def _iso_manifest_json(self, source_dir: str, rel_path: str) -> Path:
        src = self._resolve_runtime_path(source_dir)
        manifest = self._dir_manifest(src)
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._record_iso_op({"ok": True, "type": "iso_manifest", "source": str(src), "out": str(out), "count": manifest.get("count", 0)})
        return out

    def _github_local_load(self, path_value: str) -> Dict[str, Any]:
        p = self._resolve_runtime_path(path_value)
        if not p.exists():
            raise RuntimeErrorNF(f"GitHub metadata file not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        repos: List[Dict[str, Any]]
        if isinstance(raw, list):
            repos = [r for r in raw if isinstance(r, dict)]
        elif isinstance(raw, dict):
            found = None
            for key in ("repos", "repositories", "items", "data"):
                if isinstance(raw.get(key), list):
                    found = raw.get(key)
                    break
            if isinstance(found, list):
                repos = [r for r in found if isinstance(r, dict)]
            else:
                repos = [raw]
        else:
            repos = []
        return {"path": str(p), "repos": repos}

    def _github_local_summary(self, path_value: str = "repos_metadata.json") -> Dict[str, Any]:
        data = self._github_local_load(path_value)
        repos = data["repos"]
        lang_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}
        stars_total = 0
        forks_total = 0
        for repo in repos:
            lang = str(repo.get("language") or repo.get("primaryLanguage") or "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            stars_val = repo.get("stargazers_count", repo.get("stars", 0))
            forks_val = repo.get("forks_count", repo.get("forks", 0))
            if stars_val not in {None, ""}:
                stars_total += _as_int(stars_val)
            if forks_val not in {None, ""}:
                forks_total += _as_int(forks_val)
            topics = repo.get("topics") or repo.get("tags") or []
            if isinstance(topics, list):
                for t in topics:
                    topic_counts[str(t)] = topic_counts.get(str(t), 0) + 1
        top_langs = sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        top_topics = sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        top_starred = sorted(
            repos,
            key=lambda r: _as_int(r.get("stargazers_count", r.get("stars", 0))) if r.get("stargazers_count", r.get("stars", 0)) not in {None, ""} else 0,
            reverse=True,
        )[:10]
        summary = {
            "path": data["path"],
            "repo_count": len(repos),
            "stars_total": stars_total,
            "forks_total": forks_total,
            "top_languages": [{"language": k, "count": v} for k, v in top_langs],
            "top_topics": [{"topic": k, "count": v} for k, v in top_topics],
            "top_starred": [
                {
                    "name": str(r.get("name", "")),
                    "full_name": r.get("full_name", r.get("fullName")),
                    "stars": _as_int(r.get("stargazers_count", r.get("stars", 0))) if r.get("stargazers_count", r.get("stars", 0)) not in {None, ""} else 0,
                    "language": r.get("language", r.get("primaryLanguage")),
                    "description": r.get("description"),
                }
                for r in top_starred
            ],
        }
        self._record_github_local_op({"type": "github_local_summary", "path": summary["path"], "repo_count": summary["repo_count"]})
        return summary

    def _github_repo_find(self, term: str, path_value: str = "repos_metadata.json") -> List[Dict[str, Any]]:
        q = str(term).strip().lower()
        if not q:
            return []
        repos = self._github_local_load(path_value)["repos"]
        matches: List[Dict[str, Any]] = []
        for repo in repos:
            fields = [
                str(repo.get("name", "")),
                str(repo.get("full_name", repo.get("fullName", ""))),
                str(repo.get("description", "")),
                str(repo.get("language", repo.get("primaryLanguage", ""))),
                " ".join(str(t) for t in (repo.get("topics") or [])) if isinstance(repo.get("topics"), list) else "",
            ]
            hay = " | ".join(fields).lower()
            if q in hay:
                matches.append(
                    {
                        "name": str(repo.get("name", "")),
                        "full_name": repo.get("full_name", repo.get("fullName")),
                        "language": repo.get("language", repo.get("primaryLanguage")),
                        "stars": _as_int(repo.get("stargazers_count", repo.get("stars", 0))) if repo.get("stargazers_count", repo.get("stars", 0)) not in {None, ""} else 0,
                        "description": repo.get("description"),
                    }
                )
        matches.sort(key=lambda r: (-_as_int(r.get("stars", 0)), str(r.get("name", ""))))
        self._record_github_local_op({"type": "github_repo_find", "term": q, "matches": len(matches)})
        return matches[:50]

    def _idea_forge(self, theme: Optional[str] = None) -> Dict[str, Any]:
        theme_str = str(theme or "general").strip()
        accel = self._accelerator_probe(refresh=False, deep=False)
        rust_tool = self._rust_toolchain_info()
        iso_tool = self._iso_tool_info()
        try:
            github_summary = self._github_local_summary("repos_metadata.json")
        except Exception:
            github_summary = None
        dominant_langs = [x.get("language") for x in (github_summary or {}).get("top_languages", []) if isinstance(x, dict)]
        ideas = [
            {
                "name": "Steer Iso Forge",
                "category": "iso/build",
                "why": "Package curated tool bundles or offline labs into reproducible ISO images.",
                "nexusflow_steps": ["iso_manifest_json", "iso_build", "export_json"],
                "notes": "Falls back to manifest-only mode if no ISO tool is installed.",
            },
            {
                "name": "NPU Route Planner",
                "category": "hardware/ai",
                "why": "Probe host accelerators and route workloads to CUDA/MPS/NPU/CPU with deployment hints.",
                "nexusflow_steps": ["npu_probe_json", "torch_export", "write_text"],
                "notes": f"Current recommended device: {accel.get('recommended_device', 'cpu')}",
            },
            {
                "name": "Repo Graph Radar",
                "category": "github/graphs",
                "why": "Build dependency/topic graphs from local GitHub metadata and export SVG maps.",
                "nexusflow_steps": ["github_portfolio_report", "graph_create", "graph_export_svg"],
                "notes": f"Dominant languages seen: {', '.join(str(x) for x in dominant_langs[:3]) or 'unknown'}",
            },
            {
                "name": "Photo + Data Lab",
                "category": "creative/data",
                "why": "Generate procedural hero art and charts from runtime metrics in one pipeline.",
                "nexusflow_steps": ["photo_generate", "data_chart_svg", "export_html"],
                "notes": "Good for dashboards and README assets without external dependencies.",
            },
        ]
        if rust_tool:
            ideas.append(
                {
                    "name": "Rust Edge Module Bench",
                    "category": "rust/perf",
                    "why": "Compile Rust helpers as fast local modules and benchmark them from pipelines.",
                    "nexusflow_steps": ["rust_module", "rust_build", "rust_run", "bench"],
                    "notes": f"Rust toolchain detected at {rust_tool.get('path')}",
                }
            )
        if iso_tool:
            ideas[0]["notes"] = f"ISO tool available: {iso_tool.get('command')} ({iso_tool.get('path')})"
        payload = {"theme": theme_str, "generated_at": self._now_iso(), "ideas": ideas}
        self._record_github_local_op({"type": "idea_forge", "theme": theme_str, "ideas": len(ideas)})
        return payload

    def _idea_forge_json(self, rel_path: str, theme: Optional[str] = None) -> Path:
        payload = self._idea_forge(theme=theme)
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def _github_portfolio_report(self, meta_path: str, out_path: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        summary = self._github_local_summary(meta_path)
        top_langs = [x.get("language") for x in summary.get("top_languages", []) if isinstance(x, dict)]
        top_langs_l = [str(x).lower() for x in top_langs]
        suggestions: List[str] = []
        if "python" in top_langs_l:
            suggestions.append("Add more packaged CLI releases for Python-heavy repos (pipx-ready).")
        if "rust" not in top_langs_l:
            suggestions.append("Create one Rust utility module repo to improve performance-sensitive tooling stories.")
        suggestions.append("Generate repo relationship graphs (topics/languages/dependencies) and publish as SVG dashboards.")
        suggestions.append("Ship offline demo bundles/ISOs for standout projects to improve portability and showcasing.")
        suggestions.append("Add accelerator-aware examples (CUDA/NPU fallback plans) for AI repos.")
        report = {
            "summary": summary,
            "suggestions": suggestions[: max(1, _as_int(cfg_map.get("max_suggestions", 5)))],
            "idea_forge": self._idea_forge(theme="github_portfolio"),
        }
        out = self._resolve_output_path(out_path)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        with self._lock:
            self.github_local_state["last_report"] = {"path": str(out), "meta_path": summary.get("path"), "generated_at": self._now_iso()}
        self._record_github_local_op({"type": "github_portfolio_report", "meta_path": summary.get("path"), "out": str(out)})
        return {"ok": True, "path": str(out), "repo_count": summary.get("repo_count", 0)}

    def _export_csv(self, rel_path: str) -> Path:
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        snap = self.snapshot()
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["section", "name", "key", "value"])
            for k, v in (snap.get("state") or {}).items():
                writer.writerow(["state", "", k, self._safe_text(v)])
            for k, v in (snap.get("metrics") or {}).items():
                writer.writerow(["metric", "", k, self._safe_text(v)])
            for k, v in (snap.get("channels") or {}).items():
                writer.writerow(["channel", "", k, self._safe_text(v)])
            for agent_name, summary in (snap.get("agent_summary") or {}).items():
                for k, v in (summary or {}).items():
                    writer.writerow(["agent_summary", agent_name, k, self._safe_text(v)])
            for i, run in enumerate(snap.get("training_runs") or []):
                writer.writerow(["training_run", str(i), "backend", self._safe_text(run.get("backend"))])
                for k, v in run.items():
                    if k == "backend":
                        continue
                    writer.writerow(["training_run", str(i), k, self._safe_text(v)])
        return out

    def _save_state_file(self, rel_path: str) -> Path:
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "project": self.project.name,
            "tick": self.tick_count,
            "state": self.snapshot().get("state"),
            "channels": {k: list(v) for k, v in self.channels.items()},
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def _load_state_file(self, rel_path: str) -> Path:
        src = self._resolve_runtime_path(rel_path)
        data = json.loads(src.read_text(encoding="utf-8"))
        with self._lock:
            if isinstance(data.get("state"), dict):
                for k, v in data["state"].items():
                    self.state[k] = v
            if isinstance(data.get("channels"), dict):
                for k, v in data["channels"].items():
                    if isinstance(v, list):
                        self.channels[k] = list(v)
            if isinstance(data.get("tick"), int):
                self.tick_count = data["tick"]
        return src

    def _regex_flags(self, flags_value: Any) -> int:
        if flags_value is None:
            return 0
        if isinstance(flags_value, (int, float)):
            return int(flags_value)
        flags = 0
        for ch in str(flags_value):
            if ch in {"i", "I"}:
                flags |= re.IGNORECASE
            elif ch in {"m", "M"}:
                flags |= re.MULTILINE
            elif ch in {"s", "S"}:
                flags |= re.DOTALL
            elif ch in {"x", "X"}:
                flags |= re.VERBOSE
            elif ch in {"a", "A"}:
                flags |= re.ASCII
        return flags

    def _template_fill(self, template: str, data: Any, strict: bool = False) -> str:
        sentinel = object()

        def repl(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            if key in {"", "."}:
                val = data
            else:
                val = self._deep_get(data, key, sentinel) if isinstance(data, (dict, list)) else sentinel
            if val is sentinel:
                if strict:
                    raise RuntimeErrorNF(f"template_fill() missing key: {key}")
                return ""
            if isinstance(val, (dict, list)):
                return json.dumps(val, ensure_ascii=False)
            return self._safe_text(val)

        return re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", repl, str(template))

    def _sqlite_db_key(self, db_path: str) -> str:
        return ":memory:" if str(db_path) == ":memory:" else str(self._resolve_runtime_path(db_path))

    def _sqlite_connect(self, db_path: str) -> sqlite3.Connection:
        if str(db_path) == ":memory:":
            conn = sqlite3.connect(":memory:")
        else:
            p = self._resolve_runtime_path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(p))
        conn.row_factory = sqlite3.Row
        return conn

    def _sqlite_exec(self, db_path: str, sql: str, params: Any = None) -> Dict[str, Any]:
        db_key = self._sqlite_db_key(db_path)
        started = time.time()
        sql_text = str(sql)
        conn = None
        cur = None
        try:
            conn = self._sqlite_connect(db_path)
            cur = conn.cursor()
            if params is None:
                multi_stmt = sql_text.count(";") > 1 or "\n" in sql_text.strip().rstrip(";")
                if multi_stmt:
                    cur.executescript(sql_text)
                else:
                    cur.execute(sql_text)
            elif isinstance(params, (list, tuple)) and params and all(isinstance(x, (list, tuple, dict)) for x in params):
                cur.executemany(sql_text, params)
            else:
                bind_params = params if isinstance(params, (dict, list, tuple)) else (params,)
                cur.execute(sql_text, bind_params)
            conn.commit()
            result = {
                "ok": True,
                "db": db_key,
                "rowcount": int(cur.rowcount) if cur is not None and cur.rowcount is not None else -1,
                "lastrowid": int(cur.lastrowid) if cur is not None and cur.lastrowid is not None else None,
                "changes": int(conn.total_changes),
                "duration_ms": round((time.time() - started) * 1000, 2),
            }
            self._record_sqlite_op({"op": "exec", "ok": True, "db": db_key, "sql": sql_text, **result})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "sqlite_exec", "db": db_key, "ok": True})
            return result
        except Exception as exc:  # noqa: BLE001
            result = {
                "ok": False,
                "db": db_key,
                "error": str(exc),
                "duration_ms": round((time.time() - started) * 1000, 2),
            }
            self._record_sqlite_op({"op": "exec", "ok": False, "db": db_key, "sql": sql_text, "error": str(exc), "duration_ms": result["duration_ms"]})
            raise RuntimeErrorNF(f"sqlite_exec failed: {exc}") from exc
        finally:
            try:
                if cur is not None:
                    cur.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    def _sqlite_query(self, db_path: str, sql: str, params: Any = None) -> List[Dict[str, Any]]:
        db_key = self._sqlite_db_key(db_path)
        started = time.time()
        conn = None
        cur = None
        sql_text = str(sql)
        try:
            conn = self._sqlite_connect(db_path)
            cur = conn.cursor()
            if params is None:
                cur.execute(sql_text)
            else:
                bind_params = params if isinstance(params, (dict, list, tuple)) else (params,)
                cur.execute(sql_text, bind_params)
            rows = cur.fetchall()
            result = [dict(row) for row in rows]
            self._record_sqlite_op(
                {
                    "op": "query",
                    "ok": True,
                    "db": db_key,
                    "sql": sql_text,
                    "rows": len(result),
                    "duration_ms": round((time.time() - started) * 1000, 2),
                }
            )
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "sqlite_query", "db": db_key, "rows": len(result)})
            return result
        except Exception as exc:  # noqa: BLE001
            self._record_sqlite_op(
                {
                    "op": "query",
                    "ok": False,
                    "db": db_key,
                    "sql": sql_text,
                    "error": str(exc),
                    "duration_ms": round((time.time() - started) * 1000, 2),
                }
            )
            raise RuntimeErrorNF(f"sqlite_query failed: {exc}") from exc
        finally:
            try:
                if cur is not None:
                    cur.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    def _sqlite_scalar(self, db_path: str, sql: str, params: Any = None) -> Any:
        rows = self._sqlite_query(db_path, sql, params=params)
        if not rows:
            return None
        row = rows[0]
        if not isinstance(row, dict) or not row:
            return None
        first_key = next(iter(row.keys()))
        return row.get(first_key)

    def _sqlite_query_json(self, db_path: str, sql: str, rel_path: str, params: Any = None) -> Path:
        rows = self._sqlite_query(db_path, sql, params=params)
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows, "count": len(rows)}, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "sqlite_query_json", "db": self._sqlite_db_key(db_path), "rows": len(rows), "path": str(out)})
        return out

    def _sqlite_export_csv(self, db_path: str, sql: str, rel_path: str, params: Any = None) -> Path:
        rows = self._sqlite_query(db_path, sql, params=params)
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        headers: List[str] = []
        if rows and isinstance(rows[0], dict):
            headers = list(rows[0].keys())
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if headers:
                writer.writerow(headers)
                for row in rows:
                    writer.writerow([row.get(h) for h in headers])
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "sqlite_export_csv", "db": self._sqlite_db_key(db_path), "rows": len(rows), "path": str(out)})
        return out

    def _sqlite_quote_ident(self, name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def _sqlite_import_csv(self, db_path: str, table: str, csv_rel_path: str, header: bool = True, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = cfg or {}
        delimiter = str(cfg.get("delimiter", ","))
        if_exists = str(cfg.get("if_exists", "append")).lower()
        create_table = bool(cfg.get("create_table", True))
        src = self._resolve_runtime_path(csv_rel_path)
        if not src.exists():
            raise RuntimeErrorNF(f"sqlite_import_csv source missing: {src}")

        with src.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            rows_raw = list(reader)
        if not rows_raw:
            result = {"ok": True, "db": self._sqlite_db_key(db_path), "table": table, "inserted": 0, "columns": []}
            self._record_sqlite_op({"op": "import_csv", **result, "source": str(src)})
            return result

        if header:
            raw_cols = rows_raw[0]
            data_rows = rows_raw[1:]
        else:
            raw_cols = [f"c{i+1}" for i in range(len(rows_raw[0]))]
            data_rows = rows_raw
        cols = [str(c).strip() or f"c{i+1}" for i, c in enumerate(raw_cols)]
        width = len(cols)
        normalized_rows = [list(r[:width]) + [""] * max(0, width - len(r)) for r in data_rows]

        db_key = self._sqlite_db_key(db_path)
        conn = None
        cur = None
        started = time.time()
        try:
            conn = self._sqlite_connect(db_path)
            cur = conn.cursor()
            q_table = self._sqlite_quote_ident(table)
            q_cols = ", ".join(self._sqlite_quote_ident(c) for c in cols)
            if create_table:
                if if_exists == "replace":
                    cur.execute(f"DROP TABLE IF EXISTS {q_table}")
                col_defs = ", ".join(f"{self._sqlite_quote_ident(c)} TEXT" for c in cols)
                cur.execute(f"CREATE TABLE IF NOT EXISTS {q_table} ({col_defs})")
            if if_exists == "truncate":
                cur.execute(f"DELETE FROM {q_table}")
            placeholders = ", ".join(["?"] * len(cols))
            if normalized_rows:
                cur.executemany(
                    f"INSERT INTO {q_table} ({q_cols}) VALUES ({placeholders})",
                    normalized_rows,
                )
            conn.commit()
            result = {
                "ok": True,
                "db": db_key,
                "table": table,
                "inserted": len(normalized_rows),
                "columns": cols,
                "duration_ms": round((time.time() - started) * 1000, 2),
            }
            self._record_sqlite_op({"op": "import_csv", **result, "source": str(src)})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "sqlite_import_csv", "db": db_key, "table": table, "rows": len(normalized_rows)})
            return result
        except Exception as exc:  # noqa: BLE001
            self._record_sqlite_op({"op": "import_csv", "ok": False, "db": db_key, "table": table, "source": str(src), "error": str(exc)})
            raise RuntimeErrorNF(f"sqlite_import_csv failed: {exc}") from exc
        finally:
            try:
                if cur is not None:
                    cur.close()
            except Exception:
                pass
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    def _zip_pack(self, source_rel: str, archive_rel: str, include_root: bool = False) -> Path:
        source = self._resolve_runtime_path(source_rel)
        archive = self._resolve_runtime_path(archive_rel)
        if not source.exists():
            raise RuntimeErrorNF(f"zip_pack source missing: {source}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        entries = 0
        total_bytes = 0
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if source.is_file():
                zf.write(source, arcname=source.name)
                entries = 1
                total_bytes = source.stat().st_size
            else:
                root_prefix = source.name if include_root else ""
                for p in sorted(source.rglob("*")):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(source)
                    arcname = (Path(root_prefix) / rel) if root_prefix else rel
                    zf.write(p, arcname=str(arcname).replace("\\", "/"))
                    entries += 1
                    try:
                        total_bytes += p.stat().st_size
                    except Exception:
                        pass
        rec = {
            "op": "pack",
            "ok": True,
            "archive": str(archive),
            "source": str(source),
            "entries": entries,
            "bytes": total_bytes,
        }
        self._record_archive_op(rec)
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "zip_pack", "archive": str(archive), "entries": entries})
        return archive

    def _zip_unpack(self, archive_rel: str, dest_rel: str) -> Dict[str, Any]:
        archive = self._resolve_runtime_path(archive_rel)
        dest = self._resolve_runtime_path(dest_rel)
        if not archive.exists():
            raise RuntimeErrorNF(f"zip_unpack archive missing: {archive}")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            infos = zf.infolist()
            zf.extractall(dest)
        result = {
            "ok": True,
            "archive": str(archive),
            "dest": str(dest),
            "entries": len(infos),
            "bytes": sum(int(i.file_size) for i in infos),
        }
        self._record_archive_op({"op": "unpack", **result})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "zip_unpack", "archive": str(archive), "entries": len(infos)})
        return result

    def _hash_algo_name(self, algo: Optional[str]) -> str:
        raw = str(algo or "sha256").strip().lower().replace("-", "")
        aliases = {
            "sha": "sha1",
            "sha2": "sha256",
            "sha256sum": "sha256",
            "sha1sum": "sha1",
            "md5sum": "md5",
        }
        return aliases.get(raw, raw or "sha256")

    def _file_hash(self, rel_path: str, algo: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        src = self._resolve_runtime_path(rel_path)
        if not src.exists() or not src.is_file():
            raise RuntimeErrorNF(f"file_hash source missing: {src}")
        algo_name = self._hash_algo_name(str(cfg_map.get("algo", algo or "sha256")))
        chunk_size = max(1024, _as_int(cfg_map.get("chunk_size", 65536))) if cfg_map else 65536
        started = time.time()
        try:
            h = hashlib.new(algo_name)
        except Exception as exc:
            raise RuntimeErrorNF(f"Unsupported hash algorithm: {algo_name}") from exc
        size_bytes = 0
        with src.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                size_bytes += len(chunk)
                h.update(chunk)
        result = {
            "ok": True,
            "path": str(src),
            "algo": algo_name,
            "digest": h.hexdigest(),
            "bytes": int(size_bytes),
            "duration_ms": round((time.time() - started) * 1000, 2),
        }
        self._record_convert_op({"type": "file_hash", **result})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "file_hash", "path": str(src), "algo": algo_name})
        return result

    def _file_hash_verify(self, rel_path: str, expected: str, algo: Optional[str] = None) -> Dict[str, Any]:
        expected_str = str(expected).strip()
        algo_name = self._hash_algo_name(algo)
        expected_digest = expected_str
        if ":" in expected_str and not algo:
            maybe_algo, maybe_digest = expected_str.split(":", 1)
            if maybe_algo and maybe_digest:
                algo_name = self._hash_algo_name(maybe_algo)
                expected_digest = maybe_digest.strip()
        info = self._file_hash(rel_path, algo_name)
        ok = str(info.get("digest", "")).lower() == expected_digest.lower()
        payload = {
            **info,
            "expected": expected_digest,
            "match": bool(ok),
        }
        self._record_convert_op({"type": "file_hash_verify", "ok": True, "path": info["path"], "algo": info["algo"], "match": bool(ok)})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "file_hash_verify", "path": info["path"], "algo": info["algo"], "match": bool(ok)})
        return payload

    def _hash_file_json(self, rel_path: str, out_rel: str, algo: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Path:
        payload = self._file_hash(rel_path, algo=algo, cfg=cfg)
        out = self._resolve_runtime_path(out_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "hash_file_json", "path": payload.get("path"), "out": str(out)})
        return out

    def _tar_pack_mode(self, archive: Path, compression: Optional[str] = None) -> str:
        comp = str(compression or "").strip().lower()
        if comp in {"", "auto"}:
            name = archive.name.lower()
            if name.endswith((".tar.gz", ".tgz")):
                comp = "gz"
            elif name.endswith((".tar.bz2", ".tbz2", ".tbz")):
                comp = "bz2"
            elif name.endswith((".tar.xz", ".txz")):
                comp = "xz"
            else:
                comp = "none"
        if comp in {"none", "tar"}:
            return "w"
        if comp in {"gz", "gzip"}:
            return "w:gz"
        if comp in {"bz2", "bzip2"}:
            return "w:bz2"
        if comp in {"xz", "lzma"}:
            return "w:xz"
        raise RuntimeErrorNF(f"Unsupported tar compression: {compression}")

    def _tar_pack(self, source_rel: str, archive_rel: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        source = self._resolve_runtime_path(source_rel)
        archive = self._resolve_runtime_path(archive_rel)
        if not source.exists():
            raise RuntimeErrorNF(f"tar_pack source missing: {source}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        include_root = bool(cfg_map.get("include_root", False))
        mode = self._tar_pack_mode(archive, compression=cfg_map.get("compression"))
        entries = 0
        total_bytes = 0
        with tarfile.open(archive, mode) as tf:
            if source.is_file():
                tf.add(source, arcname=source.name)
                entries = 1
                try:
                    total_bytes = int(source.stat().st_size)
                except Exception:
                    total_bytes = 0
            else:
                root_prefix = source.name if include_root else ""
                for p in sorted(source.rglob("*")):
                    rel = p.relative_to(source)
                    arcname = (Path(root_prefix) / rel) if root_prefix else rel
                    tf.add(p, arcname=str(arcname).replace("\\", "/"), recursive=False)
                    if p.is_file():
                        entries += 1
                        try:
                            total_bytes += int(p.stat().st_size)
                        except Exception:
                            pass
        rec = {
            "op": "tar_pack",
            "ok": True,
            "archive": str(archive),
            "source": str(source),
            "entries": entries,
            "bytes": total_bytes,
            "mode": mode,
        }
        self._record_archive_op(rec)
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "tar_pack", "archive": str(archive), "entries": entries})
        return archive

    def _tar_unpack(self, archive_rel: str, dest_rel: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _ = cfg if isinstance(cfg, dict) else {}
        archive = self._resolve_runtime_path(archive_rel)
        dest = self._resolve_runtime_path(dest_rel)
        if not archive.exists():
            raise RuntimeErrorNF(f"tar_unpack archive missing: {archive}")
        dest.mkdir(parents=True, exist_ok=True)
        dest_root = dest.resolve()
        entries = 0
        total_bytes = 0
        with tarfile.open(archive, "r:*") as tf:
            members = tf.getmembers()
            safe_members = []
            for m in members:
                name = str(m.name or "")
                if not name or name.startswith("/") or name.startswith("\\"):
                    continue
                target = (dest_root / Path(name)).resolve()
                try:
                    if os.path.commonpath([str(dest_root), str(target)]) != str(dest_root):
                        continue
                except Exception:
                    continue
                safe_members.append(m)
                if m.isfile():
                    entries += 1
                    try:
                        total_bytes += int(m.size)
                    except Exception:
                        pass
            try:
                tf.extractall(dest_root, members=safe_members, filter="data")
            except TypeError:
                tf.extractall(dest_root, members=safe_members)
        result = {
            "ok": True,
            "archive": str(archive),
            "dest": str(dest_root),
            "entries": entries,
            "bytes": total_bytes,
        }
        self._record_archive_op({"op": "tar_unpack", **result})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "tar_unpack", "archive": str(archive), "entries": entries})
        return result

    def _dir_manifest_data(self, rel_path: str, recursive: bool = True, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        root = self._resolve_runtime_path(rel_path)
        include_hash = bool(cfg_map.get("hash", False) or cfg_map.get("include_hash", False))
        hash_algo = self._hash_algo_name(cfg_map.get("algo", "sha256")) if include_hash else None
        include_mtime = bool(cfg_map.get("mtime", True))
        limit = cfg_map.get("limit")
        limit_n = None
        if limit is not None:
            try:
                limit_n = max(1, int(limit))
            except Exception:
                limit_n = None
        started = time.time()
        if not root.exists():
            payload = {"ok": False, "path": str(root), "exists": False, "files": [], "count": 0, "bytes": 0}
            self._record_convert_op({"type": "dir_manifest", **payload})
            return payload
        if root.is_file():
            info = self._file_hash(str(root), algo=hash_algo) if include_hash else None
            try:
                st = root.stat()
                size_val = int(st.st_size)
                mtime_val = st.st_mtime
            except Exception:
                size_val = 0
                mtime_val = None
            file_rec: Dict[str, Any] = {"path": root.name, "bytes": size_val}
            if include_mtime and mtime_val is not None:
                file_rec["mtime"] = round(float(mtime_val), 6)
            if info:
                file_rec["hash"] = info.get("digest")
                file_rec["algo"] = info.get("algo")
            payload = {
                "ok": True,
                "path": str(root),
                "exists": True,
                "is_file": True,
                "files": [file_rec],
                "count": 1,
                "bytes": size_val,
                "hash": include_hash,
                "algo": hash_algo,
                "recursive": False,
                "duration_ms": round((time.time() - started) * 1000, 2),
            }
            self._record_convert_op({"type": "dir_manifest", **{k: v for k, v in payload.items() if k != "files"}})
            return payload
        files: List[Dict[str, Any]] = []
        total_bytes = 0
        total_seen = 0
        truncated = False
        it = root.rglob("*") if recursive else root.glob("*")
        for p in sorted(it):
            if not p.is_file():
                continue
            total_seen += 1
            try:
                st = p.stat()
                size_val = int(st.st_size)
                mtime_val = st.st_mtime
            except Exception:
                size_val = 0
                mtime_val = None
            total_bytes += size_val
            if limit_n is not None and len(files) >= limit_n:
                truncated = True
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            rec: Dict[str, Any] = {"path": rel, "bytes": size_val}
            if include_mtime and mtime_val is not None:
                rec["mtime"] = round(float(mtime_val), 6)
            if include_hash and hash_algo:
                h = hashlib.new(hash_algo)
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        h.update(chunk)
                rec["hash"] = h.hexdigest()
                rec["algo"] = hash_algo
            files.append(rec)
        payload = {
            "ok": True,
            "path": str(root),
            "exists": True,
            "is_file": False,
            "files": files,
            "count": total_seen,
            "bytes": total_bytes,
            "hash": include_hash,
            "algo": hash_algo,
            "recursive": bool(recursive),
            "truncated": bool(truncated),
            "duration_ms": round((time.time() - started) * 1000, 2),
        }
        self._record_convert_op({"type": "dir_manifest", **{k: v for k, v in payload.items() if k != "files"}})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "dir_manifest", "path": str(root), "count": total_seen})
        return payload

    def _dir_manifest_json(self, rel_path: str, out_rel: str, recursive: bool = True, cfg: Optional[Dict[str, Any]] = None) -> Path:
        payload = self._dir_manifest_data(rel_path, recursive=recursive, cfg=cfg)
        out = self._resolve_runtime_path(out_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "dir_manifest_json", "path": str(self._resolve_runtime_path(rel_path)), "out": str(out)})
        return out

    def _dir_diff(self, left_rel: str, right_rel: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        recursive = bool(cfg_map.get("recursive", True))
        include_hash = bool(cfg_map.get("hash", False) or cfg_map.get("include_hash", False))
        manifest_cfg = {"hash": include_hash, "algo": cfg_map.get("algo", "sha256"), "mtime": bool(cfg_map.get("mtime", True))}
        left = self._dir_manifest_data(left_rel, recursive=recursive, cfg=manifest_cfg)
        right = self._dir_manifest_data(right_rel, recursive=recursive, cfg=manifest_cfg)
        if not left.get("ok") or not right.get("ok"):
            result = {
                "ok": False,
                "left": left,
                "right": right,
                "reason": "manifest_failed",
            }
            self._record_convert_op({"type": "dir_diff", "ok": False, "reason": "manifest_failed"})
            return result
        left_map = {str(item.get("path")): item for item in left.get("files", []) if isinstance(item, dict)}
        right_map = {str(item.get("path")): item for item in right.get("files", []) if isinstance(item, dict)}
        added = sorted([p for p in right_map.keys() if p not in left_map])
        removed = sorted([p for p in left_map.keys() if p not in right_map])
        changed: List[Dict[str, Any]] = []
        same = 0
        for p in sorted(set(left_map.keys()) & set(right_map.keys())):
            a = left_map[p]
            b = right_map[p]
            a_size = a.get("bytes")
            b_size = b.get("bytes")
            a_hash = a.get("hash")
            b_hash = b.get("hash")
            if a_size != b_size or (include_hash and a_hash != b_hash):
                changed.append(
                    {
                        "path": p,
                        "left_bytes": a_size,
                        "right_bytes": b_size,
                        "left_hash": a_hash if include_hash else None,
                        "right_hash": b_hash if include_hash else None,
                    }
                )
            else:
                same += 1
        result = {
            "ok": True,
            "left_path": left.get("path"),
            "right_path": right.get("path"),
            "recursive": recursive,
            "hash": include_hash,
            "added": added,
            "removed": removed,
            "changed": changed,
            "same_count": same,
            "left_count": int(left.get("count", 0)),
            "right_count": int(right.get("count", 0)),
            "changed_count": len(changed),
            "added_count": len(added),
            "removed_count": len(removed),
        }
        self._record_convert_op({"type": "dir_diff", "ok": True, "added": len(added), "removed": len(removed), "changed": len(changed)})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "dir_diff", "added": len(added), "removed": len(removed), "changed": len(changed)})
        return result

    def _dir_diff_json(self, left_rel: str, right_rel: str, out_rel: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
        payload = self._dir_diff(left_rel, right_rel, cfg=cfg)
        out = self._resolve_runtime_path(out_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "dir_diff_json", "out": str(out), "ok": bool(payload.get("ok"))})
        return out

    def _iso_tool_info(self, refresh: bool = False) -> Dict[str, Any]:
        if not refresh:
            cached = self._cache_get("iso:tool_info", 15000)
            if isinstance(cached, dict):
                return cached
        groups = {
            "builders": ["oscdimg", "xorriso", "mkisofs", "genisoimage"],
            "listers": ["7z", "xorriso", "bsdtar", "isoinfo"],
            "extractors": ["7z", "xorriso", "bsdtar"],
        }
        info: Dict[str, Any] = {"checked_at": self._now_iso()}
        for group, names in groups.items():
            found: List[Dict[str, Any]] = []
            for name in names:
                path = shutil.which(name)
                if path:
                    found.append({"name": name, "path": path})
            info[group] = found
        info["selected"] = {
            "build": info["builders"][0]["name"] if info["builders"] else None,
            "list": info["listers"][0]["name"] if info["listers"] else None,
            "extract": info["extractors"][0]["name"] if info["extractors"] else None,
        }
        with self._lock:
            self.iso_state["tools"] = copy.deepcopy(info)
        self._cache_set("iso:tool_info", info, 15000)
        return info

    def _iso_pick_tool(self, capability: str, preferred: Optional[str] = None) -> Optional[Dict[str, Any]]:
        group = "builders" if capability == "build" else "listers" if capability == "list" else "extractors"
        items = self._iso_tool_info().get(group, [])
        if not isinstance(items, list):
            return None
        if preferred:
            pref = str(preferred).lower()
            for item in items:
                if str((item or {}).get("name", "")).lower() == pref:
                    return copy.deepcopy(item)
        return copy.deepcopy(items[0]) if items else None

    def _iso_normalize_label(self, label: Optional[str], source: Path) -> str:
        raw = str(label or source.name or "NEXUSFLOW").strip().upper()
        raw = re.sub(r"[^A-Z0-9_]+", "_", raw).strip("_")
        if not raw:
            raw = "NEXUSFLOW"
        return raw[:32]

    def _iso_scan_source_manifest(self, source: Path, limit: Optional[int] = None) -> Dict[str, Any]:
        if not source.exists():
            raise RuntimeErrorNF(f"iso source missing: {source}")
        cap = None if limit is None else max(1, int(limit))
        entries: List[Dict[str, Any]] = []
        file_count = 0
        dir_count = 0
        total_bytes = 0
        truncated = False
        if source.is_file():
            try:
                size_val = int(source.stat().st_size)
            except Exception:
                size_val = 0
            file_count = 1
            total_bytes = size_val
            if cap is None or len(entries) < cap:
                entries.append({"path": source.name, "bytes": size_val, "is_dir": False})
            else:
                truncated = True
            return {
                "entries": entries,
                "count": file_count,
                "file_count": file_count,
                "dir_count": 0,
                "bytes": total_bytes,
                "truncated": truncated,
            }
        for item in sorted(source.rglob("*")):
            try:
                if item.is_dir():
                    dir_count += 1
                    continue
                if not item.is_file():
                    continue
                rel = str(item.relative_to(source)).replace("\\", "/")
                size_val = int(item.stat().st_size)
            except Exception:
                rel = item.name
                try:
                    size_val = int(item.stat().st_size)
                except Exception:
                    size_val = 0
            file_count += 1
            total_bytes += size_val
            if cap is None or len(entries) < cap:
                entries.append({"path": rel, "bytes": size_val, "is_dir": False})
            else:
                truncated = True
        return {
            "entries": entries,
            "count": file_count,
            "file_count": file_count,
            "dir_count": dir_count,
            "bytes": total_bytes,
            "truncated": truncated,
        }

    def _iso_build_command(self, tool_name: str, source: Path, image: Path, label: str, cfg: Optional[Dict[str, Any]] = None) -> List[str]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        use_joliet = bool(cfg_map.get("joliet", True))
        use_rock_ridge = bool(cfg_map.get("rock_ridge", True))
        use_udf = bool(cfg_map.get("udf", False))
        extra_flags = [str(x) for x in cfg_map.get("flags", [])] if isinstance(cfg_map.get("flags"), list) else []
        if tool_name == "oscdimg":
            cmd = [tool_name, "-m", "-o"]
            if use_udf:
                cmd.extend(["-u2", "-udfver102"])
            elif use_joliet:
                cmd.append("-j1")
            if label:
                cmd.append(f"-l{label}")
            cmd.extend(extra_flags)
            cmd.extend([str(source), str(image)])
            return cmd
        if tool_name in {"mkisofs", "genisoimage"}:
            cmd = [tool_name, "-o", str(image)]
            if label:
                cmd.extend(["-V", label])
            if use_joliet:
                cmd.append("-J")
            if use_rock_ridge:
                cmd.append("-R")
            if use_udf:
                cmd.append("-udf")
            cmd.extend(extra_flags)
            cmd.append(str(source))
            return cmd
        if tool_name == "xorriso":
            cmd = [tool_name, "-as", "mkisofs", "-o", str(image)]
            if label:
                cmd.extend(["-V", label])
            if use_joliet:
                cmd.append("-J")
            if use_rock_ridge:
                cmd.append("-R")
            if use_udf:
                cmd.append("-udf")
            cmd.extend(extra_flags)
            cmd.append(str(source))
            return cmd
        raise RuntimeErrorNF(f"Unsupported ISO builder tool: {tool_name}")

    def _iso_list_command(self, tool_name: str, image: Path) -> List[str]:
        if tool_name == "7z":
            return [tool_name, "l", "-slt", str(image)]
        if tool_name == "bsdtar":
            return [tool_name, "-tf", str(image)]
        if tool_name == "xorriso":
            return [tool_name, "-indev", str(image), "-find", "/", "-print"]
        if tool_name == "isoinfo":
            return [tool_name, "-i", str(image), "-f"]
        raise RuntimeErrorNF(f"Unsupported ISO lister tool: {tool_name}")

    def _iso_extract_command(self, tool_name: str, image: Path, dest: Path) -> List[str]:
        if tool_name == "7z":
            return [tool_name, "x", "-y", f"-o{dest}", str(image)]
        if tool_name == "bsdtar":
            return [tool_name, "-xf", str(image), "-C", str(dest)]
        if tool_name == "xorriso":
            return [tool_name, "-osirrox", "on", "-indev", str(image), "-extract", "/", str(dest)]
        raise RuntimeErrorNF(f"Unsupported ISO extractor tool: {tool_name}")

    def _iso_parse_list_output(self, tool_name: str, image: Path, run_result: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
        stdout = str(run_result.get("stdout") or "")
        cap = None if limit is None else max(1, int(limit))
        entries: List[Dict[str, Any]] = []
        total_seen = 0
        truncated = False
        if tool_name == "7z":
            records: List[Dict[str, str]] = []
            cur: Dict[str, str] = {}
            for raw in stdout.splitlines():
                line = raw.strip()
                if not line:
                    if cur:
                        records.append(cur)
                        cur = {}
                    continue
                if " = " in line:
                    k, v = line.split(" = ", 1)
                    cur[k.strip()] = v.strip()
            if cur:
                records.append(cur)
            for rec in records:
                p = str(rec.get("Path", "")).strip()
                if not p:
                    continue
                typ = str(rec.get("Type", "")).strip().lower()
                if typ == "iso" and (p == str(image) or Path(p).name == image.name):
                    continue
                total_seen += 1
                is_dir = str(rec.get("Folder", "")).strip() == "+"
                size_raw = rec.get("Size")
                try:
                    size_val = int(size_raw) if size_raw not in {None, ""} else 0
                except Exception:
                    size_val = 0
                if cap is None or len(entries) < cap:
                    entries.append({"path": p.replace("\\", "/"), "bytes": size_val, "is_dir": bool(is_dir)})
                else:
                    truncated = True
            return {"entries": entries, "count": total_seen, "truncated": truncated}
        for raw in stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            if tool_name in {"xorriso", "isoinfo"} and not line.startswith("/"):
                continue
            total_seen += 1
            if cap is None or len(entries) < cap:
                entries.append({"path": line.replace("\\", "/"), "bytes": None, "is_dir": line.endswith("/")})
            else:
                truncated = True
        return {"entries": entries, "count": total_seen, "truncated": truncated}

    def _iso_build(self, source_rel: str, image_rel: str, label: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        source = self._resolve_runtime_path(source_rel)
        image = self._resolve_runtime_path(image_rel)
        if not source.exists():
            raise RuntimeErrorNF(f"iso_build source missing: {source}")
        image.parent.mkdir(parents=True, exist_ok=True)
        dry_run = bool(cfg_map.get("dry_run", False))
        timeout_sec = float(cfg_map.get("timeout_sec", 120.0))
        label_value = self._iso_normalize_label((cfg_map.get("label") if cfg_map.get("label") is not None else label), source)
        manifest = self._iso_scan_source_manifest(source, limit=None)
        manifest_preview = manifest["entries"][: min(200, len(manifest["entries"]))]
        tool_meta = self._iso_pick_tool("build", preferred=str(cfg_map.get("tool")) if cfg_map.get("tool") else None)
        tool_name = str(tool_meta["name"]) if isinstance(tool_meta, dict) and tool_meta.get("name") else None
        cmd = self._iso_build_command(tool_name, source, image, label_value, cfg_map) if tool_name else None
        if dry_run:
            result = {
                "ok": True,
                "dry_run": True,
                "image": str(image),
                "source": str(source),
                "label": label_value,
                "tool": tool_name,
                "command": cmd,
                "entries": int(manifest["count"]),
                "bytes": int(manifest["bytes"]),
                "mode": "dry_run",
            }
            self._record_iso_op(
                {
                    "op": "build",
                    **result,
                    "exists": image.exists(),
                    "manifest_preview": manifest_preview,
                    "manifest_preview_truncated": bool(manifest["count"] > len(manifest_preview)),
                    "manifest_count": int(manifest["count"]),
                    "source_file_count": int(manifest["file_count"]),
                    "source_dir_count": int(manifest["dir_count"]),
                    "source_bytes": int(manifest["bytes"]),
                }
            )
            self._record_archive_op({"op": "iso_build_dry_run", "ok": True, "archive": str(image), "entries": int(manifest["count"]), "bytes": int(manifest["bytes"])})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_build", "image": str(image), "dry_run": True})
            return result
        if not tool_name or not cmd:
            result = {
                "ok": False,
                "image": str(image),
                "source": str(source),
                "label": label_value,
                "reason": "tool_missing",
                "capability": "build",
                "tools": self._iso_tool_info(),
                "entries": int(manifest["count"]),
                "bytes": int(manifest["bytes"]),
            }
            self._record_iso_op(
                {
                    "op": "build",
                    **result,
                    "exists": image.exists(),
                    "manifest_preview": manifest_preview,
                    "manifest_preview_truncated": bool(manifest["count"] > len(manifest_preview)),
                    "manifest_count": int(manifest["count"]),
                    "source_file_count": int(manifest["file_count"]),
                    "source_dir_count": int(manifest["dir_count"]),
                    "source_bytes": int(manifest["bytes"]),
                }
            )
            self._record_archive_op({"op": "iso_build", "ok": False, "archive": str(image), "entries": int(manifest["count"]), "bytes": int(manifest["bytes"])})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_build", "image": str(image), "ok": False, "reason": "tool_missing"})
            return result
        run = self._run_subprocess(cmd, timeout_sec=timeout_sec)
        exists = image.exists()
        try:
            size_bytes = int(image.stat().st_size) if exists else 0
        except Exception:
            size_bytes = 0
        result = {
            "ok": bool(run.get("ok")) and exists,
            "image": str(image),
            "source": str(source),
            "label": label_value,
            "tool": tool_name,
            "command": cmd,
            "entries": int(manifest["count"]),
            "bytes": int(manifest["bytes"]),
            "size_bytes": size_bytes,
            "exists": exists,
            "stdout": run.get("stdout", ""),
            "stderr": run.get("stderr", ""),
            "returncode": run.get("returncode"),
            "duration_ms": run.get("duration_ms"),
            "mode": "tool",
        }
        if not bool(run.get("ok")):
            result["reason"] = "build_failed"
        elif not exists:
            result["ok"] = False
            result["reason"] = "output_missing"
        self._record_iso_op(
            {
                "op": "build",
                **result,
                "dry_run": False,
                "manifest_preview": manifest_preview,
                "manifest_preview_truncated": bool(manifest["count"] > len(manifest_preview)),
                "manifest_count": int(manifest["count"]),
                "source_file_count": int(manifest["file_count"]),
                "source_dir_count": int(manifest["dir_count"]),
                "source_bytes": int(manifest["bytes"]),
            }
        )
        self._record_archive_op({"op": "iso_build", "ok": bool(result["ok"]), "archive": str(image), "entries": int(manifest["count"]), "bytes": size_bytes})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "iso_build", "image": str(image), "ok": bool(result["ok"]), "tool": tool_name})
        return result

    def _iso_list(self, image_rel: str, limit: Optional[int] = None, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        image = self._resolve_runtime_path(image_rel)
        limit_i = None if limit is None else max(1, int(limit))
        allow_source_fallback = bool(cfg_map.get("allow_source_fallback", True))
        prefer_source_manifest = bool(cfg_map.get("prefer_source_manifest", False))
        meta = None
        with self._lock:
            images = (self.iso_state or {}).get("images", {})
            if isinstance(images, dict):
                meta = copy.deepcopy(images.get(str(image)))
                if meta is None:
                    for pth, item in images.items():
                        if Path(str(pth)).name == image.name:
                            meta = copy.deepcopy(item)
                            break
        source_path: Optional[Path] = None
        if isinstance(meta, dict) and meta.get("source"):
            try:
                p = Path(str(meta["source"]))
                if p.exists():
                    source_path = p
            except Exception:
                source_path = None
        if allow_source_fallback and source_path is not None and (prefer_source_manifest or not image.exists()):
            manifest = self._iso_scan_source_manifest(source_path, limit=limit_i)
            payload = {
                "ok": True,
                "image": str(image),
                "mode": "source_manifest",
                "tool": None,
                "entries": manifest["entries"],
                "count": int(manifest["count"]),
                "bytes": int(manifest["bytes"]),
                "truncated": bool(manifest["truncated"]),
            }
            self._record_iso_op({"op": "list", **payload})
            self._record_archive_op({"op": "iso_list", "ok": True, "archive": str(image), "entries": int(manifest["count"]), "bytes": int(manifest["bytes"])})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_list", "image": str(image), "mode": "source_manifest"})
            return payload
        if not image.exists():
            payload = {"ok": False, "image": str(image), "reason": "missing_image"}
            self._record_iso_op({"op": "list", **payload})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_list", "image": str(image), "ok": False, "reason": "missing_image"})
            return payload
        tool_meta = self._iso_pick_tool("list", preferred=str(cfg_map.get("tool")) if cfg_map.get("tool") else None)
        if not isinstance(tool_meta, dict):
            payload = {"ok": False, "image": str(image), "reason": "tool_missing", "capability": "list", "tools": self._iso_tool_info()}
            self._record_iso_op({"op": "list", **payload})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_list", "image": str(image), "ok": False, "reason": "tool_missing"})
            return payload
        tool_name = str(tool_meta["name"])
        run = self._run_subprocess(self._iso_list_command(tool_name, image), timeout_sec=float(cfg_map.get("timeout_sec", 45.0)))
        if not bool(run.get("ok")):
            payload = {
                "ok": False,
                "image": str(image),
                "reason": "list_failed",
                "tool": tool_name,
                "stdout": run.get("stdout", ""),
                "stderr": run.get("stderr", ""),
                "returncode": run.get("returncode"),
                "duration_ms": run.get("duration_ms"),
            }
            self._record_iso_op({"op": "list", **payload})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_list", "image": str(image), "ok": False, "reason": "list_failed"})
            return payload
        parsed = self._iso_parse_list_output(tool_name, image, run, limit=limit_i)
        payload = {
            "ok": True,
            "image": str(image),
            "mode": "tool",
            "tool": tool_name,
            "entries": parsed.get("entries", []),
            "count": int(parsed.get("count", 0)),
            "truncated": bool(parsed.get("truncated", False)),
            "duration_ms": run.get("duration_ms"),
        }
        self._record_iso_op({"op": "list", **payload})
        self._record_archive_op({"op": "iso_list", "ok": True, "archive": str(image), "entries": int(payload["count"])})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "iso_list", "image": str(image), "ok": True, "tool": tool_name})
        return payload

    def _iso_list_json(self, image_rel: str, out_rel: str, limit: Optional[int] = None, cfg: Optional[Dict[str, Any]] = None) -> Path:
        payload = self._iso_list(image_rel, limit=limit, cfg=cfg)
        out = self._resolve_runtime_path(out_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "iso_list_json", "path": str(out), "ok": bool(payload.get("ok"))})
        return out

    def _iso_extract(self, image_rel: str, dest_rel: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_map = cfg if isinstance(cfg, dict) else {}
        image = self._resolve_runtime_path(image_rel)
        dest = self._resolve_runtime_path(dest_rel)
        dry_run = bool(cfg_map.get("dry_run", False))
        overwrite = bool(cfg_map.get("overwrite", True))
        allow_source_fallback = bool(cfg_map.get("allow_source_fallback", True))
        if dry_run:
            payload = {"ok": True, "image": str(image), "dest": str(dest), "dry_run": True, "mode": "dry_run"}
            self._record_iso_op({"op": "extract", **payload, "extract_dest": str(dest)})
            self._record_archive_op({"op": "iso_extract_dry_run", "ok": True, "archive": str(image)})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_extract", "image": str(image), "dry_run": True})
            return payload
        meta = None
        with self._lock:
            images = (self.iso_state or {}).get("images", {})
            if isinstance(images, dict):
                meta = copy.deepcopy(images.get(str(image)))
                if meta is None:
                    for pth, item in images.items():
                        if Path(str(pth)).name == image.name:
                            meta = copy.deepcopy(item)
                            break
        source_fallback: Optional[Path] = None
        if isinstance(meta, dict) and meta.get("source"):
            try:
                p = Path(str(meta["source"]))
                if p.exists():
                    source_fallback = p
            except Exception:
                source_fallback = None
        if not image.exists() and allow_source_fallback and source_fallback is not None:
            if source_fallback.is_dir():
                shutil.copytree(source_fallback, dest, dirs_exist_ok=overwrite)
                man = self._iso_scan_source_manifest(source_fallback, limit=None)
                payload = {
                    "ok": True,
                    "image": str(image),
                    "dest": str(dest),
                    "mode": "source_fallback_copy",
                    "source": str(source_fallback),
                    "entries": int(man["count"]),
                    "bytes": int(man["bytes"]),
                }
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                target = (dest / source_fallback.name) if dest.suffix == "" else dest
                if target.exists() and not overwrite:
                    raise RuntimeErrorNF(f"iso_extract destination exists and overwrite=false: {target}")
                shutil.copy2(source_fallback, target)
                try:
                    size_val = int(target.stat().st_size)
                except Exception:
                    size_val = 0
                payload = {
                    "ok": True,
                    "image": str(image),
                    "dest": str(dest),
                    "mode": "source_fallback_copy",
                    "source": str(source_fallback),
                    "entries": 1,
                    "bytes": size_val,
                }
            self._record_iso_op({"op": "extract", **payload, "extract_dest": str(dest)})
            self._record_archive_op({"op": "iso_extract", "ok": True, "archive": str(image), "entries": int(payload.get("entries", 0)), "bytes": int(payload.get("bytes", 0))})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_extract", "image": str(image), "mode": "source_fallback_copy"})
            return payload
        if not image.exists():
            payload = {"ok": False, "image": str(image), "dest": str(dest), "reason": "missing_image"}
            self._record_iso_op({"op": "extract", **payload, "extract_dest": str(dest)})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_extract", "image": str(image), "ok": False, "reason": "missing_image"})
            return payload
        dest.mkdir(parents=True, exist_ok=True)
        tool_meta = self._iso_pick_tool("extract", preferred=str(cfg_map.get("tool")) if cfg_map.get("tool") else None)
        if not isinstance(tool_meta, dict):
            payload = {"ok": False, "image": str(image), "dest": str(dest), "reason": "tool_missing", "capability": "extract", "tools": self._iso_tool_info()}
            self._record_iso_op({"op": "extract", **payload, "extract_dest": str(dest)})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_extract", "image": str(image), "ok": False, "reason": "tool_missing"})
            return payload
        tool_name = str(tool_meta["name"])
        run = self._run_subprocess(self._iso_extract_command(tool_name, image, dest), timeout_sec=float(cfg_map.get("timeout_sec", 120.0)))
        dir_meta = self._dir_stats(str(dest), recursive=True)
        payload = {
            "ok": bool(run.get("ok")),
            "image": str(image),
            "dest": str(dest),
            "mode": "tool",
            "tool": tool_name,
            "entries": int(dir_meta.get("files", 0)),
            "bytes": int(dir_meta.get("bytes", 0)),
            "stdout": run.get("stdout", ""),
            "stderr": run.get("stderr", ""),
            "returncode": run.get("returncode"),
            "duration_ms": run.get("duration_ms"),
        }
        if not payload["ok"]:
            payload["reason"] = "extract_failed"
        self._record_iso_op({"op": "extract", **payload, "extract_dest": str(dest)})
        self._record_archive_op({"op": "iso_extract", "ok": bool(payload["ok"]), "archive": str(image), "entries": int(payload["entries"]), "bytes": int(payload["bytes"])})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "iso_extract", "image": str(image), "ok": bool(payload["ok"]), "tool": tool_name})
        return payload

    def _win_open(self, target: str) -> Dict[str, Any]:
        if not self._is_windows_host:
            opened = webbrowser.open(target)
            result = {"ok": bool(opened), "target": target, "mode": "webbrowser"}
            self._record_windows_op({"type": "win_open", **result})
            return result
        try:
            os.startfile(target)  # type: ignore[attr-defined]
            result = {"ok": True, "target": target, "mode": "startfile"}
        except Exception as exc:
            result = {"ok": False, "target": target, "error": str(exc)}
        self._record_windows_op({"type": "win_open", **result})
        return result

    def _win_reveal(self, target: str) -> Dict[str, Any]:
        if not self._is_windows_host:
            result = {"ok": False, "target": target, "reason": "not_windows"}
            self._record_windows_op({"type": "win_reveal", **result})
            return result
        p = str(self._resolve_runtime_path(target) if not Path(target).is_absolute() else Path(target))
        if Path(p).exists() and Path(p).is_dir():
            cmd = ["explorer", p]
        else:
            cmd = ["explorer", f"/select,{p}"]
        result = self._run_subprocess(cmd, timeout_sec=15.0)
        self._record_windows_op({"type": "win_reveal", "target": p, **result})
        return result

    def _win_notify(self, title: str, message: str, seconds: int = 4) -> Dict[str, Any]:
        if not self._is_windows_host:
            result = {"ok": False, "reason": "not_windows"}
            self._record_windows_op({"type": "win_notify", "title": title, "message": message, **result})
            return result
        safe_title = title.replace("'", "''")
        safe_msg = message.replace("'", "''")
        script = (
            "$wshell = New-Object -ComObject WScript.Shell; "
            f"$null = $wshell.Popup('{safe_msg}', {max(1, seconds)}, '{safe_title}', 0x40); "
            "Write-Output 'ok'"
        )
        result = self._run_powershell(script, timeout_sec=max(5.0, float(seconds) + 2.0))
        self._record_windows_op({"type": "win_notify", "title": title, "message": message, **result})
        return result

    def _win_clipboard_set(self, text: str) -> Dict[str, Any]:
        script = "Set-Clipboard -Value ([Console]::In.ReadToEnd()); Write-Output 'ok'"
        result = self._run_powershell(script, stdin_text=text, timeout_sec=10.0)
        self._record_windows_op({"type": "win_clipboard_set", "chars": len(text), **result})
        return result

    def _win_clipboard_get(self) -> str:
        result = self._run_powershell("Get-Clipboard -Raw", timeout_sec=10.0)
        return (result.get("stdout") or "").rstrip("\r\n")

    def _win_cmd(self, command: str, timeout_sec: float = 30.0) -> Dict[str, Any]:
        if not self._is_windows_host:
            result = {"ok": False, "reason": "not_windows", "stdout": "", "stderr": "Host is not Windows"}
            self._record_windows_op({"type": "win_cmd", "command": command, **result})
            return result
        result = self._run_subprocess(["cmd", "/c", command], timeout_sec=timeout_sec)
        self._record_windows_op({"type": "win_cmd", "command": command, **result})
        return result

    def _win_beep(self, freq: int = 880, duration_ms: int = 120) -> Dict[str, Any]:
        freq = max(37, min(32767, int(freq)))
        duration_ms = max(10, int(duration_ms))
        if not self._is_windows_host:
            result = {"ok": False, "reason": "not_windows"}
            self._record_windows_op({"type": "win_beep", "freq": freq, "duration_ms": duration_ms, **result})
            return result
        script = f"[console]::Beep({freq}, {duration_ms}); Write-Output 'ok'"
        result = self._run_powershell(script, timeout_sec=max(2.0, duration_ms / 1000.0 + 1.0))
        self._record_windows_op({"type": "win_beep", "freq": freq, "duration_ms": duration_ms, **result})
        return result

    def _win_processes_json(self, rel_path: str, top_n: int = 20) -> Path:
        top_n = max(1, int(top_n))
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "processes": []}
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._record_windows_op({"type": "win_processes_json", "path": str(out), **payload})
            return out
        script = (
            f"Get-Process | Sort-Object -Property CPU -Descending | Select-Object -First {top_n} "
            "| Select-Object Name,Id,CPU,WS,PM,Path | ConvertTo-Json -Depth 3"
        )
        result = self._run_powershell(script, timeout_sec=20.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "top_n": top_n, "processes": []}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else []
                if isinstance(parsed, dict):
                    parsed = [parsed]
                payload["processes"] = parsed
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
                payload["raw"] = raw[:4000]
        else:
            payload["stderr"] = result.get("stderr", "")
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._record_windows_op({"type": "win_processes_json", "path": str(out), "top_n": top_n, "ok": payload.get("ok")})
        return out

    def _win_service_status(self, service_name: str) -> Dict[str, Any]:
        if not self._is_windows_host:
            result = {"ok": False, "reason": "not_windows", "service": service_name}
            self._record_windows_op({"type": "win_service_status", **result})
            return result
        safe = service_name.replace("'", "''")
        script = (
            f"$s = Get-Service -Name '{safe}' -ErrorAction Stop; "
            "$s | Select-Object Name,DisplayName,Status,StartType,ServiceType,CanPauseAndContinue,CanStop "
            "| ConvertTo-Json -Depth 3"
        )
        result = self._run_powershell(script, timeout_sec=15.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "service": service_name}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                payload["data"] = json.loads(raw) if raw else None
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
                payload["raw"] = raw[:2000]
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_service_status", "service": service_name, "ok": payload.get("ok")})
        return payload

    def _win_services_json(self, rel_path: str, top_n: int = 50) -> Path:
        top_n = max(1, int(top_n))
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "services": []}
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._record_windows_op({"type": "win_services_json", "path": str(out), **payload})
            return out
        script = (
            "Get-Service | Sort-Object -Property Status,DisplayName "
            f"| Select-Object -First {top_n} Name,DisplayName,Status,StartType | ConvertTo-Json -Depth 3"
        )
        result = self._run_powershell(script, timeout_sec=20.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "services": []}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else []
                if isinstance(parsed, dict):
                    parsed = [parsed]
                payload["services"] = parsed
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
                payload["raw"] = raw[:4000]
        else:
            payload["stderr"] = result.get("stderr", "")
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._record_windows_op({"type": "win_services_json", "path": str(out), "top_n": top_n, "ok": payload.get("ok")})
        return out

    def _win_registry_get(self, path: str, name: str, default: Any = None) -> Any:
        if not self._is_windows_host:
            self._record_windows_op({"type": "win_registry_get", "path": path, "name": name, "ok": False, "reason": "not_windows"})
            return default
        p = path.replace("'", "''")
        n = name.replace("'", "''")
        script = (
            f"try {{ $v = Get-ItemPropertyValue -Path '{p}' -Name '{n}' -ErrorAction Stop; "
            "if ($null -eq $v) { Write-Output '' } else { Write-Output $v } } "
            "catch { exit 2 }"
        )
        result = self._run_powershell(script, timeout_sec=10.0)
        ok = bool(result.get("ok"))
        self._record_windows_op({"type": "win_registry_get", "path": path, "name": name, "ok": ok})
        if not ok:
            return default
        text = (result.get("stdout") or "").rstrip("\r\n")
        return text if text != "" else default

    def _win_windows_list(self, top_n: int = 20) -> Dict[str, Any]:
        top_n = max(1, int(top_n))
        cache_key = f"win:windows_list:{top_n}"
        cached = self._cache_get(cache_key, 1000)
        if isinstance(cached, dict):
            self._record_windows_op({"type": "win_windows_list", "ok": cached.get("ok"), "top_n": top_n, "count": len(cached.get("windows", [])), "cached": True})
            return cached
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "top_n": top_n, "windows": []}
            self._record_windows_op({"type": "win_windows_list", **payload})
            return payload
        script = (
            "Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -and $_.MainWindowTitle.Trim().Length -gt 0 } "
            f"| Sort-Object -Property ProcessName,Id | Select-Object -First {top_n} "
            "Name,Id,MainWindowTitle,MainWindowHandle,Path | ConvertTo-Json -Depth 4"
        )
        result = self._run_powershell(script, timeout_sec=20.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "top_n": top_n, "windows": []}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else []
                if isinstance(parsed, dict):
                    parsed = [parsed]
                payload["windows"] = parsed
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
                payload["raw"] = raw[:4000]
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_windows_list", "ok": payload.get("ok"), "top_n": top_n, "count": len(payload.get("windows", []))})
        return self._cache_set(cache_key, payload, 1000)

    def _win_windows_json(self, rel_path: str, top_n: int = 20) -> Path:
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self._win_windows_list(top_n=top_n)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._record_windows_op({"type": "win_windows_json", "path": str(out), "top_n": top_n, "ok": payload.get("ok")})
        return out

    def _win_foreground_window(self) -> Dict[str, Any]:
        cache_key = "win:foreground"
        cached = self._cache_get(cache_key, 500)
        if isinstance(cached, dict):
            self._record_windows_op({"type": "win_foreground_window", "ok": cached.get("ok"), "title": str(cached.get("title", ""))[:120], "cached": True})
            return cached
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows"}
            self._record_windows_op({"type": "win_foreground_window", **payload})
            return payload
        script = r'''
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class NFWinFg {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowTextLength(IntPtr hWnd);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr hWnd, StringBuilder text, int count);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
}
"@ -ErrorAction SilentlyContinue | Out-Null
$h = [NFWinFg]::GetForegroundWindow()
if ($h -eq [IntPtr]::Zero) { [pscustomobject]@{ ok = $false; reason = 'no_foreground' } | ConvertTo-Json -Depth 4; exit 0 }
$len = [NFWinFg]::GetWindowTextLength($h)
$sb = New-Object System.Text.StringBuilder ([Math]::Max(4, $len + 2))
[void][NFWinFg]::GetWindowText($h, $sb, $sb.Capacity)
$csb = New-Object System.Text.StringBuilder 260
[void][NFWinFg]::GetClassName($h, $csb, $csb.Capacity)
[uint32]$procId = 0
[void][NFWinFg]::GetWindowThreadProcessId($h, [ref]$procId)
$pname = $null
try { $pname = (Get-Process -Id $procId -ErrorAction Stop).ProcessName } catch {}
[pscustomobject]@{
  ok = $true
  handle = [int64]$h
  title = $sb.ToString()
  class = $csb.ToString()
  pid = [int]$procId
  process = $pname
} | ConvertTo-Json -Depth 4
'''
        result = self._run_powershell(script, timeout_sec=10.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok"))}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload = parsed
                else:
                    payload["data"] = parsed
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
                payload["raw"] = raw[:2000]
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_foreground_window", "ok": payload.get("ok"), "title": str(payload.get("title", ""))[:120]})
        return self._cache_set(cache_key, payload, 500)

    def _win_mouse_pos(self) -> Dict[str, Any]:
        cache_key = "win:mouse_pos"
        cached = self._cache_get(cache_key, 250)
        if isinstance(cached, dict):
            self._record_windows_op({"type": "win_mouse_pos", "ok": cached.get("ok"), "x": cached.get("x"), "y": cached.get("y"), "cached": True})
            return cached
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "x": None, "y": None}
            self._record_windows_op({"type": "win_mouse_pos", **payload})
            return payload
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$p = [System.Windows.Forms.Cursor]::Position; "
            "[pscustomobject]@{ ok = $true; x = [int]$p.X; y = [int]$p.Y } | ConvertTo-Json -Compress"
        )
        result = self._run_powershell(script, timeout_sec=8.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "x": None, "y": None}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_mouse_pos", "ok": payload.get("ok"), "x": payload.get("x"), "y": payload.get("y")})
        return self._cache_set(cache_key, payload, 250)

    def _win_screen_size(self) -> Dict[str, Any]:
        cache_key = "win:screen_size"
        cached = self._cache_get(cache_key, 5000)
        if isinstance(cached, dict):
            self._record_windows_op({"type": "win_screen_size", "ok": cached.get("ok"), "width": cached.get("width"), "height": cached.get("height"), "cached": True})
            return cached
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "width": 0, "height": 0}
            self._record_windows_op({"type": "win_screen_size", **payload})
            return payload
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "[pscustomobject]@{ ok = $true; width = [int]$b.Width; height = [int]$b.Height; x = [int]$b.X; y = [int]$b.Y } | ConvertTo-Json -Compress"
        )
        result = self._run_powershell(script, timeout_sec=8.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "width": 0, "height": 0}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_screen_size", "ok": payload.get("ok"), "width": payload.get("width"), "height": payload.get("height")})
        return self._cache_set(cache_key, payload, 5000)

    def _win_activate_window(self, target: Any, timeout_sec: float = 5.0) -> Dict[str, Any]:
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "target": target}
            self._record_windows_op({"type": "win_activate_window", **payload})
            return payload
        script: str
        if isinstance(target, (int, float)):
            script = (
                "$ws = New-Object -ComObject WScript.Shell; "
                f"$ok = $ws.AppActivate({int(target)}); "
                "[pscustomobject]@{ ok = [bool]$ok; target = '" + str(int(target)).replace("'", "''") + "' } | ConvertTo-Json -Compress"
            )
        else:
            t = str(target).replace("'", "''")
            script = (
                "$ws = New-Object -ComObject WScript.Shell; "
                f"$ok = $ws.AppActivate('{t}'); "
                f"[pscustomobject]@{{ ok = [bool]$ok; target = '{t}' }} | ConvertTo-Json -Compress"
            )
        result = self._run_powershell(script, timeout_sec=max(1.0, float(timeout_sec)))
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "target": target}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_activate_window", "ok": payload.get("ok"), "target": str(target)[:120]})
        return payload

    def _win_sendkeys_escape(self, text: str) -> str:
        out: List[str] = []
        for ch in str(text):
            if ch == "{":
                out.append("{{}")
            elif ch == "}":
                out.append("{}}")
            elif ch in {"+", "^", "%", "~", "(", ")"}:
                out.append("{" + ch + "}")
            elif ch == "\n":
                out.append("~")
            elif ch == "\r":
                continue
            else:
                out.append(ch)
        return "".join(out)

    def _win_key_send(self, keys: str) -> Dict[str, Any]:
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "keys": keys}
            self._record_windows_op({"type": "win_key_send", **payload})
            return payload
        k = str(keys).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            f"[System.Windows.Forms.SendKeys]::SendWait('{k}'); "
            f"[pscustomobject]@{{ ok = $true; keys = '{k}' }} | ConvertTo-Json -Compress"
        )
        result = self._run_powershell(script, timeout_sec=8.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "keys": keys}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_key_send", "ok": payload.get("ok"), "keys": str(keys)[:80]})
        return payload

    def _win_type_text(self, text: str, interval_ms: int = 0) -> Dict[str, Any]:
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "chars": len(str(text))}
            self._record_windows_op({"type": "win_type_text", **payload})
            return payload
        chars = str(text)
        delay = max(0, int(interval_ms))
        escaped = self._win_sendkeys_escape(chars).replace("'", "''")
        if delay <= 0:
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                f"[System.Windows.Forms.SendKeys]::SendWait('{escaped}'); "
                f"[pscustomobject]@{{ ok = $true; chars = {len(chars)}; interval_ms = 0 }} | ConvertTo-Json -Compress"
            )
        else:
            ps_parts = []
            for ch in chars:
                part = self._win_sendkeys_escape(ch).replace("'", "''")
                ps_parts.append("'" + part + "'")
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$parts = @("
                + ",".join(ps_parts)
                + "); "
                f"$d = {delay}; "
                "foreach($p in $parts){ [System.Windows.Forms.SendKeys]::SendWait($p); if($d -gt 0){ Start-Sleep -Milliseconds $d } }; "
                f"[pscustomobject]@{{ ok = $true; chars = {len(chars)}; interval_ms = {delay} }} | ConvertTo-Json -Compress"
            )
        result = self._run_powershell(script, timeout_sec=max(8.0, 2.0 + (len(chars) * max(0, delay)) / 1000.0))
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "chars": len(chars), "interval_ms": delay}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_type_text", "ok": payload.get("ok"), "chars": len(chars), "interval_ms": delay})
        return payload

    def _win_mouse_move(self, x: int, y: int, relative: bool = False) -> Dict[str, Any]:
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "x": int(x), "y": int(y), "relative": bool(relative)}
            self._record_windows_op({"type": "win_mouse_move", **payload})
            return payload
        xv = int(x)
        yv = int(y)
        if relative:
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
                "$p = [System.Windows.Forms.Cursor]::Position; "
                f"$x = [int]$p.X + ({xv}); $y = [int]$p.Y + ({yv}); "
                "[System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point($x, $y); "
                "$q = [System.Windows.Forms.Cursor]::Position; "
                "[pscustomobject]@{ ok = $true; x = [int]$q.X; y = [int]$q.Y; relative = $true } | ConvertTo-Json -Compress"
            )
        else:
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
                f"[System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point({xv}, {yv}); "
                "$q = [System.Windows.Forms.Cursor]::Position; "
                "[pscustomobject]@{ ok = $true; x = [int]$q.X; y = [int]$q.Y; relative = $false } | ConvertTo-Json -Compress"
            )
        result = self._run_powershell(script, timeout_sec=8.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "x": xv, "y": yv, "relative": bool(relative)}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_mouse_move", "ok": payload.get("ok"), "x": payload.get("x"), "y": payload.get("y"), "relative": bool(relative)})
        return payload

    def _win_mouse_click(self, button: str = "left", clicks: int = 1, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        btn = str(button or "left").lower()
        clicks_n = max(1, int(clicks))
        if btn not in {"left", "right", "middle"}:
            btn = "left"
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "button": btn, "clicks": clicks_n, "x": x, "y": y}
            self._record_windows_op({"type": "win_mouse_click", **payload})
            return payload
        x_part = f"$x = {int(x)}; $y = {int(y)}; [NFMouseApi]::SetCursorPos($x, $y) | Out-Null; " if x is not None and y is not None else ""
        flag_map = {
            "left": (0x0002, 0x0004),
            "right": (0x0008, 0x0010),
            "middle": (0x0020, 0x0040),
        }
        down_flag, up_flag = flag_map[btn]
        script = (
            r'''
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class NFMouseApi {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
}
"@ -ErrorAction SilentlyContinue | Out-Null
'''
            + x_part
            + f"$btnDown = [uint32]{down_flag}; $btnUp = [uint32]{up_flag}; "
            + f"for($i=0; $i -lt {clicks_n}; $i++) {{ [NFMouseApi]::mouse_event($btnDown,0,0,0,[UIntPtr]::Zero); Start-Sleep -Milliseconds 18; [NFMouseApi]::mouse_event($btnUp,0,0,0,[UIntPtr]::Zero); if($i -lt {clicks_n - 1}){{ Start-Sleep -Milliseconds 50 }} }}; "
            + "Add-Type -AssemblyName System.Windows.Forms; $p = [System.Windows.Forms.Cursor]::Position; "
            + f"[pscustomobject]@{{ ok = $true; button = '{btn}'; clicks = {clicks_n}; x = [int]$p.X; y = [int]$p.Y }} | ConvertTo-Json -Compress"
        )
        result = self._run_powershell(script, timeout_sec=max(5.0, 1.0 + clicks_n * 0.15))
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "button": btn, "clicks": clicks_n, "x": x, "y": y}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_mouse_click", "ok": payload.get("ok"), "button": btn, "clicks": clicks_n})
        return payload

    def _win_mouse_scroll(self, delta: int = 120) -> Dict[str, Any]:
        delta_n = int(delta)
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "delta": delta_n}
            self._record_windows_op({"type": "win_mouse_scroll", **payload})
            return payload
        script = (
            r'''
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class NFMouseWheelApi {
  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
}
"@ -ErrorAction SilentlyContinue | Out-Null
'''
            + f"[NFMouseWheelApi]::mouse_event([uint32]0x0800,0,0,[uint32]{delta_n & 0xFFFFFFFF},[UIntPtr]::Zero); "
            + f"[pscustomobject]@{{ ok = $true; delta = {delta_n} }} | ConvertTo-Json -Compress"
        )
        result = self._run_powershell(script, timeout_sec=5.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "delta": delta_n}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_windows_op({"type": "win_mouse_scroll", "ok": payload.get("ok"), "delta": delta_n})
        return payload

    def _win_input_sequence(self, actions: Any, dry_run: bool = False) -> Dict[str, Any]:
        if not isinstance(actions, list):
            raise RuntimeErrorNF("win_input_sequence(actions[, dry_run]) expects actions to be a list")
        results: List[Dict[str, Any]] = []
        ok = True
        for idx, item in enumerate(actions):
            if not isinstance(item, dict):
                rec = {"ok": False, "index": idx, "reason": "invalid_action", "action": item}
                results.append(rec)
                ok = False
                continue
            action = str(item.get("action", item.get("type", ""))).lower().strip()
            if not action:
                rec = {"ok": False, "index": idx, "reason": "missing_action"}
                results.append(rec)
                ok = False
                continue
            if dry_run:
                rec = {"ok": True, "index": idx, "action": action, "dry_run": True, "spec": copy.deepcopy(item)}
                results.append(rec)
                if action == "sleep":
                    continue
                continue
            try:
                if action == "move":
                    rec = self._win_mouse_move(_as_int(item.get("x", 0)), _as_int(item.get("y", 0)), bool(item.get("relative", False)))
                elif action == "click":
                    rec = self._win_mouse_click(
                        str(item.get("button", "left")),
                        _as_int(item.get("clicks", 1)),
                        _as_int(item["x"]) if "x" in item and item.get("x") is not None else None,
                        _as_int(item["y"]) if "y" in item and item.get("y") is not None else None,
                    )
                elif action == "scroll":
                    rec = self._win_mouse_scroll(_as_int(item.get("delta", 120)))
                elif action in {"keys", "key_send", "chord"}:
                    rec = self._win_key_send(str(item.get("keys", item.get("value", ""))))
                elif action in {"type", "text"}:
                    rec = self._win_type_text(str(item.get("text", item.get("value", ""))), _as_int(item.get("interval_ms", 0)))
                elif action == "activate":
                    rec = self._win_activate_window(item.get("target", item.get("value", "")))
                elif action == "sleep":
                    ms = max(0, _as_int(item.get("ms", item.get("milliseconds", 0))))
                    time.sleep(ms / 1000.0)
                    rec = {"ok": True, "sleep_ms": ms}
                else:
                    rec = {"ok": False, "reason": "unknown_action", "action": action}
            except Exception as exc:  # noqa: BLE001
                rec = {"ok": False, "reason": "exception", "error": str(exc), "action": action}
            rec = {"index": idx, "action": action, **rec}
            results.append(rec)
            if not bool(rec.get("ok")):
                ok = False
        payload = {"ok": ok, "dry_run": bool(dry_run), "count": len(actions), "results": results}
        self._record_windows_op({"type": "win_input_sequence", "ok": ok, "dry_run": bool(dry_run), "count": len(actions)})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "win_input_sequence", "ok": ok, "count": len(actions), "dry_run": bool(dry_run)})
        return payload

    def _vui_supported(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "voice_ui": True,
            "tts": bool(self._is_windows_host),
            "speech_recognition": False,
            "simulated_input": True,
            "platform": self._host_info().get("platform"),
        }

    def _vui_list_voices(self, refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            cached = copy.deepcopy(self.vui_state.get("voices_cache"))
        if cached and not refresh:
            return cached
        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "voices": []}
            self._record_vui_op({"type": "vui_list_voices", **payload})
            with self._lock:
                self.vui_state["voices_cache"] = copy.deepcopy(payload)
            return payload
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$v = $s.GetInstalledVoices() | ForEach-Object { "
            "  $vi = $_.VoiceInfo; "
            "  $culture = $null; if($vi.Culture){ $culture = $vi.Culture.Name }; "
            "  [pscustomobject]@{ "
            "    Name = $vi.Name; "
            "    Culture = $culture; "
            "    Gender = [string]$vi.Gender; "
            "    Age = [string]$vi.Age; "
            "    Description = $vi.Description "
            "  } "
            "}; "
            "$v | ConvertTo-Json -Depth 4"
        )
        result = self._run_powershell(script, timeout_sec=12.0)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "voices": []}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else []
                if isinstance(parsed, dict):
                    parsed = [parsed]
                payload["voices"] = parsed
            except Exception as exc:
                payload["ok"] = False
                payload["parse_error"] = str(exc)
                payload["raw"] = raw[:4000]
        else:
            payload["stderr"] = result.get("stderr", "")
        self._record_vui_op({"type": "vui_list_voices", "ok": payload.get("ok"), "count": len(payload.get("voices", []))})
        with self._lock:
            self.vui_state["voices_cache"] = copy.deepcopy(payload)
        return payload

    def _vui_profile_set(self, name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(cfg, dict):
            raise RuntimeErrorNF("vui_profile(name, config) expects config to be a dict")
        profile = {
            "name": str(name),
            "voice": None if cfg.get("voice") is None else str(cfg.get("voice")),
            "rate": int(cfg.get("rate", 0)) if cfg.get("rate") is not None else 0,
            "volume": int(cfg.get("volume", 100)) if cfg.get("volume") is not None else 100,
            "style": None if cfg.get("style") is None else str(cfg.get("style")),
        }
        profile["rate"] = max(-10, min(10, int(profile["rate"])))
        profile["volume"] = max(0, min(100, int(profile["volume"])))
        with self._lock:
            self.vui_state.setdefault("profiles", {})[str(name)] = copy.deepcopy(profile)
            tick_now = self.tick_count
        self._record_vui_op({"type": "vui_profile_set", "ok": True, "profile": str(name)})
        self._append_event({"tick": tick_now, "event": "vui_profile_set", "profile": str(name)})
        return profile

    def _vui_profile_get(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            p = copy.deepcopy((self.vui_state.get("profiles") or {}).get(str(name)))
        return p if isinstance(p, dict) else None

    def _vui_intent(self, text: str, intents: Any) -> Dict[str, Any]:
        phrase = str(text or "")
        if not isinstance(intents, dict):
            raise RuntimeErrorNF("vui_intent(text, intents_map) expects a dict of intents")
        lower = phrase.lower()
        best_name: Optional[str] = None
        best_score = -1.0
        best_hits: List[str] = []
        for intent_name, spec in intents.items():
            score = 0.0
            hits: List[str] = []
            patterns: List[Any]
            weight = 1.0
            if isinstance(spec, dict):
                patterns = []
                if isinstance(spec.get("patterns"), list):
                    patterns.extend(spec.get("patterns") or [])
                if isinstance(spec.get("keywords"), list):
                    patterns.extend(spec.get("keywords") or [])
                if isinstance(spec.get("phrases"), list):
                    patterns.extend(spec.get("phrases") or [])
                try:
                    weight = float(spec.get("weight", 1.0))
                except Exception:
                    weight = 1.0
            elif isinstance(spec, list):
                patterns = list(spec)
            else:
                patterns = [spec]
            for pat in patterns:
                if pat is None:
                    continue
                if isinstance(pat, str):
                    ptxt = pat.strip()
                    if not ptxt:
                        continue
                    # Heuristic: regex if it contains common regex metacharacters
                    is_regex = bool(re.search(r"[\\^$.|?*+()[\]]", ptxt))
                    matched = False
                    if is_regex:
                        try:
                            if re.search(ptxt, phrase, re.IGNORECASE):
                                matched = True
                        except Exception:
                            matched = ptxt.lower() in lower
                    else:
                        matched = ptxt.lower() in lower
                    if matched:
                        hits.append(ptxt)
                        score += (2.0 if is_regex else 1.0)
            score *= weight
            if score > best_score:
                best_name = str(intent_name)
                best_score = score
                best_hits = hits
        return {
            "text": phrase,
            "intent": best_name if best_score > 0 else None,
            "score": round(best_score, 3) if best_score >= 0 else 0.0,
            "matched": best_hits,
        }

    def _vui_extract(self, text: str, slots: Any) -> Dict[str, Any]:
        if not isinstance(slots, dict):
            raise RuntimeErrorNF("vui_extract(text, slots_map) expects a dict of regex patterns")
        phrase = str(text or "")
        out: Dict[str, Any] = {}
        for key, pattern in slots.items():
            if pattern is None:
                out[str(key)] = None
                continue
            pat = str(pattern)
            try:
                m = re.search(pat, phrase, re.IGNORECASE)
            except Exception:
                m = None
            if not m:
                out[str(key)] = None
                continue
            if m.groupdict():
                out[str(key)] = m.groupdict()
            elif m.groups():
                out[str(key)] = list(m.groups())
            else:
                out[str(key)] = m.group(0)
        return out

    def _vui_log(self, role: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = {"type": "vui_log", "ok": True, "role": str(role), "transcript": str(text), "meta": copy.deepcopy(meta) if isinstance(meta, dict) else None}
        self._record_vui_op(rec)
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "vui_log", "role": str(role), "chars": len(str(text))})
        return rec

    def _vui_say(self, text: str, profile_or_cfg: Any = None, cfg: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
        phrase = str(text)
        config: Dict[str, Any] = {}
        profile_name: Optional[str] = None
        if isinstance(profile_or_cfg, str):
            profile_name = str(profile_or_cfg)
            prof = self._vui_profile_get(profile_name)
            if isinstance(prof, dict):
                config.update(prof)
        elif isinstance(profile_or_cfg, dict):
            config.update(copy.deepcopy(profile_or_cfg))
        if isinstance(cfg, dict):
            config.update(copy.deepcopy(cfg))
        if "dry_run" in config:
            dry_run = bool(config.get("dry_run"))
        voice = None if config.get("voice") in {None, ""} else str(config.get("voice"))
        rate = max(-10, min(10, int(config.get("rate", 0))))
        volume = max(0, min(100, int(config.get("volume", 100))))

        if dry_run:
            payload = {"ok": True, "dry_run": True, "text": phrase, "chars": len(phrase), "voice": voice, "rate": rate, "volume": volume, "profile": profile_name}
            self._record_vui_op({"type": "vui_say", **payload})
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "vui_say", "dry_run": True, "chars": len(phrase)})
            return payload

        if not self._is_windows_host:
            payload = {"ok": False, "reason": "not_windows", "text": phrase, "chars": len(phrase), "profile": profile_name}
            self._record_vui_op({"type": "vui_say", **payload})
            return payload

        ps_text = phrase.replace("'", "''")
        ps_voice = "" if voice is None else voice.replace("'", "''")
        select_voice = f"try {{ $s.SelectVoice('{ps_voice}') }} catch {{ }}; " if ps_voice else ""
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {rate}; $s.Volume = {volume}; "
            + select_voice
            + f"$s.Speak('{ps_text}'); "
            + f"[pscustomobject]@{{ ok = $true; chars = {len(phrase)}; rate = {rate}; volume = {volume}; voice = '{ps_voice}' }} | ConvertTo-Json -Compress"
        )
        timeout = max(10.0, 3.0 + len(phrase) / 12.0)
        result = self._run_powershell(script, timeout_sec=timeout)
        payload: Dict[str, Any] = {"ok": bool(result.get("ok")), "text": phrase, "chars": len(phrase), "voice": voice, "rate": rate, "volume": volume, "profile": profile_name}
        if result.get("ok"):
            raw = (result.get("stdout") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except Exception:
                pass
        else:
            payload["stderr"] = result.get("stderr", "")
            payload["reason"] = result.get("reason") if result.get("reason") else None
        self._record_vui_op({"type": "vui_say", "ok": payload.get("ok"), "chars": len(phrase), "voice": voice, "profile": profile_name})
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "vui_say", "ok": bool(payload.get("ok")), "chars": len(phrase)})
        return payload

    def _vui_voices_json(self, rel_path: str, refresh: bool = False) -> Path:
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self._vui_list_voices(refresh=refresh)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "vui_voices_json", "path": str(out), "ok": bool(payload.get("ok"))})
        return out

    def _vui_export_json(self, rel_path: str, limit: Optional[int] = None) -> Path:
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            ops = copy.deepcopy(self.vui_state.get("ops", []))
            profiles = copy.deepcopy(self.vui_state.get("profiles", {}))
            transcripts = copy.deepcopy(self.vui_state.get("transcripts", []))
            voices_cache = copy.deepcopy(self.vui_state.get("voices_cache"))
        if limit is not None:
            n = max(0, int(limit))
            ops = ops[-n:]
            transcripts = transcripts[-n:]
        payload = {"ops": ops, "profiles": profiles, "transcripts": transcripts, "voices_cache": voices_cache}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "vui_export_json", "path": str(out)})
        return out

    def _export_events_jsonl(self, rel_path: str) -> Path:
        out = self._resolve_runtime_path(rel_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            events = copy.deepcopy(self.events)
        with out.open("w", encoding="utf-8") as fh:
            for evt in events:
                fh.write(json.dumps(evt, ensure_ascii=False) + "\n")
        return out

    def _vec3(self, value: Any, default: Optional[List[float]] = None) -> List[float]:
        if default is None:
            default = [0.0, 0.0, 0.0]
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return [float(value[0]), float(value[1]), float(value[2])]
            except Exception:
                return list(default)
        return list(default)

    def _normalize_transform(self, value: Any) -> Dict[str, Any]:
        transform: Dict[str, Any] = {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
        if not isinstance(value, dict):
            return transform
        if "position" in value:
            transform["position"] = self._vec3(value.get("position"), [0.0, 0.0, 0.0])
        if "rotation" in value:
            transform["rotation"] = self._vec3(value.get("rotation"), [0.0, 0.0, 0.0])
        if "scale" in value:
            transform["scale"] = self._vec3(value.get("scale"), [1.0, 1.0, 1.0])
        if "name" in value:
            transform["name"] = str(value["name"])
        return transform

    def _extract_bounds_from_vertices(self, vertices: List[List[float]]) -> Optional[Dict[str, Any]]:
        if not vertices:
            return None
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        return {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
            "size": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
            "center": [(max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0, (max(zs) + min(zs)) / 2.0],
        }

    def _load_obj_asset(self, alias: str, rel_path: str) -> Dict[str, Any]:
        src = self._resolve_runtime_path(rel_path)
        text = src.read_text(encoding="utf-8", errors="replace")
        vertices: List[List[float]] = []
        normals = 0
        uvs = 0
        faces = 0
        tris = 0
        objects: List[str] = []
        groups: List[str] = []
        materials: List[str] = []
        mtllibs: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            head = parts[0]
            if head == "v" and len(parts) >= 4:
                try:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except Exception:
                    pass
            elif head == "vn":
                normals += 1
            elif head == "vt":
                uvs += 1
            elif head == "f" and len(parts) >= 4:
                faces += 1
                tris += max(1, len(parts) - 3)
            elif head == "o" and len(parts) >= 2:
                objects.append(" ".join(parts[1:]))
            elif head == "g" and len(parts) >= 2:
                groups.append(" ".join(parts[1:]))
            elif head == "usemtl" and len(parts) >= 2:
                materials.append(" ".join(parts[1:]))
            elif head == "mtllib" and len(parts) >= 2:
                mtllibs.append(" ".join(parts[1:]))
        payload = {
            "name": alias,
            "kind": "obj",
            "path": str(src),
            "bytes": src.stat().st_size if src.exists() else 0,
            "stats": {
                "vertices": len(vertices),
                "normals": normals,
                "uvs": uvs,
                "faces": faces,
                "triangles_estimate": tris,
                "objects": sorted(set(objects)),
                "groups": sorted(set(groups)),
                "materials": sorted(set(materials)),
                "mtllibs": sorted(set(mtllibs)),
            },
            "bounds": self._extract_bounds_from_vertices(vertices),
            "loaded_at": self._now_iso(),
        }
        with self._lock:
            self.assets3d[alias] = payload
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "load_3d", "asset": alias, "kind": "obj"})
        return payload

    def _gltf_payload_from_json(self, alias: str, src: Path, data: Dict[str, Any], kind: str) -> Dict[str, Any]:
        meshes = data.get("meshes") if isinstance(data.get("meshes"), list) else []
        nodes = data.get("nodes") if isinstance(data.get("nodes"), list) else []
        scenes = data.get("scenes") if isinstance(data.get("scenes"), list) else []
        materials = data.get("materials") if isinstance(data.get("materials"), list) else []
        animations = data.get("animations") if isinstance(data.get("animations"), list) else []
        accessors = data.get("accessors") if isinstance(data.get("accessors"), list) else []
        images = data.get("images") if isinstance(data.get("images"), list) else []
        buffers = data.get("buffers") if isinstance(data.get("buffers"), list) else []
        buffer_views = data.get("bufferViews") if isinstance(data.get("bufferViews"), list) else []
        primitives = sum(len(m.get("primitives") or []) for m in meshes if isinstance(m, dict))
        payload = {
            "name": alias,
            "kind": kind,
            "path": str(src),
            "bytes": src.stat().st_size if src.exists() else 0,
            "asset": data.get("asset", {}),
            "stats": {
                "scenes": len(scenes),
                "nodes": len(nodes),
                "meshes": len(meshes),
                "primitives": primitives,
                "materials": len(materials),
                "animations": len(animations),
                "accessors": len(accessors),
                "buffers": len(buffers),
                "bufferViews": len(buffer_views),
                "images": len(images),
            },
            "default_scene": data.get("scene"),
            "extensions_used": list(data.get("extensionsUsed") or []),
            "extensions_required": list(data.get("extensionsRequired") or []),
            "loaded_at": self._now_iso(),
        }
        return payload

    def _load_gltf_asset(self, alias: str, rel_path: str) -> Dict[str, Any]:
        src = self._resolve_runtime_path(rel_path)
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeErrorNF(f"gltf file must contain an object: {src}")
        payload = self._gltf_payload_from_json(alias, src, data, "gltf")
        with self._lock:
            self.assets3d[alias] = payload
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "load_3d", "asset": alias, "kind": "gltf"})
        return payload

    def _load_glb_asset(self, alias: str, rel_path: str) -> Dict[str, Any]:
        src = self._resolve_runtime_path(rel_path)
        raw = src.read_bytes()
        if len(raw) < 12:
            raise RuntimeErrorNF(f"Invalid GLB (too small): {src}")
        magic, version, total_len = struct.unpack_from("<4sII", raw, 0)
        if magic != b"glTF":
            raise RuntimeErrorNF(f"Invalid GLB magic for {src}")
        chunks: List[Dict[str, Any]] = []
        json_obj: Optional[Dict[str, Any]] = None
        offset = 12
        while offset + 8 <= len(raw):
            chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
            offset += 8
            chunk_data = raw[offset : offset + chunk_len]
            offset += chunk_len
            ctype = chunk_type.to_bytes(4, byteorder="little", signed=False).decode("ascii", errors="replace")
            chunk_meta: Dict[str, Any] = {"type": ctype, "length": chunk_len}
            if ctype == "JSON":
                try:
                    json_obj = json.loads(chunk_data.decode("utf-8"))
                    chunk_meta["json"] = True
                except Exception as exc:
                    chunk_meta["json_error"] = str(exc)
            chunks.append(chunk_meta)
        base = self._gltf_payload_from_json(alias, src, json_obj or {"asset": {}}, "glb")
        base["glb"] = {
            "version": version,
            "declared_length": total_len,
            "actual_length": len(raw),
            "chunks": chunks,
        }
        with self._lock:
            self.assets3d[alias] = base
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "load_3d", "asset": alias, "kind": "glb"})
        return base

    def _load_3d_asset(self, alias: str, rel_path: str, kind: Optional[str] = None) -> Dict[str, Any]:
        chosen = (kind or "").strip().lower()
        if not chosen:
            chosen = Path(rel_path).suffix.lower().lstrip(".")
        if chosen == "obj":
            return self._load_obj_asset(alias, rel_path)
        if chosen == "gltf":
            return self._load_gltf_asset(alias, rel_path)
        if chosen == "glb":
            return self._load_glb_asset(alias, rel_path)
        raise RuntimeErrorNF(f"Unsupported 3D asset type: {chosen} (expected obj/gltf/glb)")

    def _new_scene3d(self, scene_name: str, template: str = "stage") -> Dict[str, Any]:
        scene = {
            "name": scene_name,
            "template": template,
            "background": "#0b1020",
            "nodes": [],
            "lights": [],
            "camera": {
                "position": [4.0, 3.0, 8.0],
                "target": [0.0, 0.0, 0.0],
                "fov": 50.0,
            },
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.scenes3d[scene_name] = scene
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "scene_new", "scene": scene_name, "template": template})
        return scene

    def _require_scene3d(self, scene_name: str) -> Dict[str, Any]:
        scene = self.scenes3d.get(scene_name)
        if scene is None:
            raise RuntimeErrorNF(f"Unknown scene3d: {scene_name}")
        return scene

    def _require_asset3d(self, asset_name: str) -> Dict[str, Any]:
        asset = self.assets3d.get(asset_name)
        if asset is None:
            raise RuntimeErrorNF(f"Unknown 3D asset: {asset_name}")
        return asset

    def _scene3d_add_node(
        self,
        scene_name: str,
        asset_name: str,
        node_name: Optional[str] = None,
        transform: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        asset = self._require_asset3d(asset_name)
        scene = self._require_scene3d(scene_name)
        tf = self._normalize_transform(transform or {})
        node = {
            "name": node_name or f"{asset_name}_{len(scene['nodes'])}",
            "asset": asset_name,
            "kind": "mesh",
            "transform": tf,
            "asset_kind": asset.get("kind"),
            "asset_stats": copy.deepcopy(asset.get("stats", {})),
            "bounds": copy.deepcopy(asset.get("bounds")),
        }
        with self._lock:
            scene["nodes"].append(node)
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "scene_add", "scene": scene_name, "asset": asset_name})
        return node

    def _scene3d_add_light(self, scene_name: str, kind: str = "directional", intensity: float = 1.0, color: str = "#ffffff") -> Dict[str, Any]:
        scene = self._require_scene3d(scene_name)
        light = {
            "kind": str(kind),
            "intensity": float(intensity),
            "color": str(color),
        }
        with self._lock:
            scene["lights"].append(light)
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "scene_light", "scene": scene_name, "kind": str(kind)})
        return light

    def _scene3d_set_camera(self, scene_name: str, position: Any, target: Any, fov: float = 50.0) -> Dict[str, Any]:
        scene = self._require_scene3d(scene_name)
        camera = {
            "position": self._vec3(position, [4.0, 3.0, 8.0]),
            "target": self._vec3(target, [0.0, 0.0, 0.0]),
            "fov": float(fov),
        }
        with self._lock:
            scene["camera"] = camera
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "scene_camera", "scene": scene_name})
        return camera

    def _export_scene_json(self, scene_name: str, rel_path: str) -> Path:
        scene = self._require_scene3d(scene_name)
        out = self._resolve_output_path(rel_path)
        payload = {
            "project": self.project.name,
            "scene": copy.deepcopy(scene),
            "assets": {n["asset"]: copy.deepcopy(self.assets3d.get(n["asset"], {})) for n in scene.get("nodes", []) if isinstance(n, dict)},
            "tick": self.tick_count,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "export_scene_json", "scene": scene_name, "path": str(out)})
        return out

    def _build_scene_html(self, scene_name: str) -> str:
        scene = self._require_scene3d(scene_name)
        scene_json = json.dumps(copy.deepcopy(scene), indent=2)
        assets = {n["asset"]: self.assets3d.get(n["asset"], {}) for n in scene.get("nodes", []) if isinstance(n, dict)}
        assets_json = json.dumps(assets, indent=2)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self.project.name} - Scene {scene_name}</title>
  <style>
    :root {{
      --bg:#070b14; --panel:#0f172a; --line:#213049; --text:#e2e8f0; --muted:#94a3b8; --accent:#22d3ee;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Segoe UI,system-ui,sans-serif; background:radial-gradient(circle at 10% 10%, rgba(34,211,238,.15), transparent 45%), linear-gradient(180deg,#020617,var(--bg)); color:var(--text); }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:20px; }}
    .hero {{ border:1px solid var(--line); border-radius:14px; padding:14px; background:rgba(15,23,42,.75); }}
    .grid {{ display:grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, .8fr); gap:14px; margin-top:14px; }}
    .card {{ border:1px solid var(--line); border-radius:14px; padding:12px; background:rgba(15,23,42,.8); }}
    h1,h2 {{ margin:0 0 10px; }}
    h1 {{ font-size:22px; }}
    h2 {{ font-size:14px; }}
    .muted {{ color:var(--muted); }}
    canvas {{ width:100%; height:360px; display:block; background:linear-gradient(180deg,#050a14,#0b1220); border:1px solid rgba(255,255,255,.06); border-radius:10px; }}
    .kv {{ display:grid; grid-template-columns:1fr auto; gap:8px; font-size:13px; }}
    .kv div {{ padding:4px 0; border-bottom:1px dashed rgba(148,163,184,.12); }}
    pre {{ margin:0; font-size:12px; line-height:1.35; background:rgba(2,6,23,.55); border:1px solid rgba(255,255,255,.06); border-radius:10px; padding:10px; overflow:auto; max-height:360px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{self.project.name} / scene3d "{scene_name}"</h1>
      <div class="muted">Native NexusFlow scene export (dependency-free preview; metadata-based visualizer)</div>
    </section>
    <section class="grid">
      <div class="card">
        <h2>Viewport Preview</h2>
        <canvas id="cv" width="640" height="360"></canvas>
      </div>
      <div class="card">
        <h2>Scene Summary</h2>
        <div class="kv" id="summary"></div>
      </div>
      <div class="card">
        <h2>Scene JSON</h2>
        <pre id="scene_json"></pre>
      </div>
      <div class="card">
        <h2>Assets JSON</h2>
        <pre id="assets_json"></pre>
      </div>
    </section>
  </div>
  <script>
    const scene = {scene_json};
    const assets = {assets_json};
    const summary = document.getElementById('summary');
    const scenePairs = [
      ['template', scene.template || 'stage'],
      ['nodes', (scene.nodes || []).length],
      ['lights', (scene.lights || []).length],
      ['camera.fov', scene.camera?.fov ?? 50],
      ['assets', Object.keys(assets).length],
    ];
    for (const [k,v] of scenePairs) {{
      const a = document.createElement('div'); a.className='muted'; a.textContent = k;
      const b = document.createElement('div'); b.textContent = String(v);
      summary.append(a,b);
    }}
    document.getElementById('scene_json').textContent = JSON.stringify(scene, null, 2);
    document.getElementById('assets_json').textContent = JSON.stringify(assets, null, 2);
    const cv = document.getElementById('cv');
    const ctx = cv.getContext('2d');
    let t = 0;
    function draw() {{
      const w = cv.width, h = cv.height;
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle = '#081021';
      ctx.fillRect(0,0,w,h);
      ctx.strokeStyle = 'rgba(34,211,238,0.2)';
      for (let i=0;i<8;i++) {{
        const y = (i+1)*h/9;
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
      }}
      const nodes = Array.isArray(scene.nodes) ? scene.nodes : [];
      nodes.forEach((node, idx) => {{
        const pos = node?.transform?.position || [0,0,0];
        const x = Number(pos[0] || 0), y = Number(pos[1] || 0), z = Number(pos[2] || 0);
        const orbit = t * 0.0007 + idx * 1.2;
        const px = w/2 + (x*35 + Math.cos(orbit)*(18 + idx*3));
        const py = h/2 - (y*35 + Math.sin(orbit)*12) - z*8;
        const size = Math.max(4, 10 - z * 0.6);
        ctx.fillStyle = ['#22d3ee','#f59e0b','#86efac','#fb7185'][idx % 4];
        ctx.beginPath(); ctx.arc(px, py, size, 0, Math.PI*2); ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.22)';
        ctx.stroke();
        ctx.fillStyle = 'rgba(226,232,240,0.9)';
        ctx.font = '12px Segoe UI, sans-serif';
        ctx.fillText(String(node.name || node.asset || ('node'+idx)), px + size + 4, py + 4);
      }});
      t += 16;
      requestAnimationFrame(draw);
    }}
    draw();
  </script>
</body>
</html>
"""

    def _export_scene_html(self, scene_name: str, rel_path: str) -> Path:
        out = self._resolve_output_path(rel_path)
        out.write_text(self._build_scene_html(scene_name), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "export_scene_html", "scene": scene_name, "path": str(out)})
        return out

    def _web_slugify(self, text: Any) -> str:
        s = str(text).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        s = re.sub(r"-{2,}", "-", s).strip("-")
        return s or "tool"

    def _nexus_ide_tool_body(self) -> str:
        return """
<style>
.ide-wrap{display:grid;grid-template-columns:260px 1fr 340px;gap:12px;margin-top:12px}.ide-col{display:grid;gap:12px;align-content:start}
.ide-editor textarea{min-height:470px;resize:vertical}.ide-files{max-height:220px;overflow:auto}.file-item{display:flex;justify-content:space-between;gap:8px;padding:8px;border:1px solid rgba(255,255,255,.08);border-radius:8px;margin:6px 0;background:rgba(2,6,23,.32);cursor:pointer}.file-item.active{border-color:rgba(34,211,238,.35);background:rgba(34,211,238,.08)}
.mini{font-size:12px}.split{display:grid;grid-template-columns:1fr 1fr;gap:8px}.kbd{border:1px solid rgba(255,255,255,.12);border-bottom-width:2px;border-radius:6px;padding:1px 6px;font-size:11px;color:#cbd5e1}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}.tabs button{width:auto}.tabs button.active{border-color:rgba(34,211,238,.35)}.panel{display:none}.panel.active{display:block}.chip{display:inline-block;margin:2px 4px 0 0;padding:2px 7px;border-radius:999px;border:1px solid rgba(255,255,255,.1);font-size:11px}
@media (max-width:1080px){.ide-wrap{grid-template-columns:1fr}.ide-editor textarea{min-height:320px}}
</style>
<div class='ide-wrap'>
  <div class='ide-col'>
    <div class='card'>
      <h2>NexusFlow IDE</h2>
      <div class='muted mini'>File Explorer • snippets • local workspace</div>
      <div class='toolbar'>
        <button id='nf_new_file'>New File</button>
        <button id='nf_save_ws'>Save WS</button>
        <button id='nf_load_ws'>Load WS</button>
      </div>
      <div class='ide-files' id='nf_files'></div>
      <div class='split'>
        <button id='nf_download'>Download</button>
        <label class='mini' style='display:grid;place-items:center;border:1px dashed rgba(255,255,255,.14);border-radius:10px;padding:8px;cursor:pointer'>Import<input id='nf_import' type='file' style='display:none'></label>
      </div>
    </div>
    <div class='card'>
      <h2>Snippets</h2>
      <select id='nf_snippets' class='mono'>
        <option value='project'>Project Skeleton</option>
        <option value='agent'>Agent Block</option>
        <option value='pipeline'>Pipeline Block</option>
        <option value='fusion'>Fusion + Protein Pack</option>
        <option value='ide'>IDE Generator Steps</option>
      </select>
      <div class='toolbar'><button id='nf_insert'>Insert Snippet</button><button id='nf_format'>Format (Lite)</button></div>
      <div class='muted mini'>Shortcuts: <span class='kbd'>Ctrl+S</span> save workspace <span class='kbd'>Ctrl+Enter</span> run preview</div>
    </div>
  </div>
  <div class='ide-col ide-editor'>
    <div class='card'>
      <h2 id='nf_editor_title'>Editor</h2>
      <div class='toolbar'>
        <button id='nf_lint'>Lint</button>
        <button id='nf_ast'>AST Sketch</button>
        <button id='nf_run'>Run Pipeline</button>
        <button id='nf_clear_console'>Clear Console</button>
      </div>
      <textarea id='nf_editor' class='mono' spellcheck='false'></textarea>
    </div>
    <div class='card'>
      <h2>Console</h2>
      <div id='nf_console_meta' class='muted mini'>idle</div>
      <pre id='nf_console'></pre>
    </div>
  </div>
  <div class='ide-col'>
    <div class='card'>
      <h2>Inspector</h2>
      <div class='tabs'>
        <button data-tab='diag' class='nf_tab active'>Diagnostics</button>
        <button data-tab='trace' class='nf_tab'>Trace</button>
        <button data-tab='snap' class='nf_tab'>Snapshot</button>
      </div>
      <div id='nf_panel_diag' class='panel active'><pre id='nf_diag'></pre></div>
      <div id='nf_panel_trace' class='panel'><pre id='nf_trace'></pre></div>
      <div id='nf_panel_snap' class='panel'><pre id='nf_snap'></pre></div>
    </div>
    <div class='card'>
      <h2>Actions</h2>
      <div class='toolbar'>
        <button id='nf_gen_idle'>Open IDLE Tab</button>
        <button id='nf_theme'>Accent Shift</button>
      </div>
      <div id='nf_stats' class='muted mini'></div>
      <div id='nf_chips'></div>
    </div>
  </div>
</div>
<script>(function(){
const $=id=>document.getElementById(id);
const KEY='nexusflow.ide.workspace.v1';
const starter=`project "Studio" {
  config seed = 7;
  state food = 10;
  metric food_left = food;
  agent grazer count 2 {
    field energy = 1;
    on tick {
      if food > 0 { food -= 1; energy += 1; }
    }
  }
  pipeline preview {
    step simulate(2);
    step export_json("out/snap.json");
    step summary();
  }
}`;
let accentShift=false;
let ws={files:[{name:'main.nxf',code:starter},{name:'notes.txt',code:'# NexusFlow IDE\\n- edit\\n- lint\\n- run'}],active:0,history:[]};
const snippets={
  project:`project "MyProject" {\\n  pipeline preview {\\n    step summary();\\n  }\\n}\\n`,
  agent:`agent worker count 3 {\\n  field energy = 1;\\n  on tick {\\n    energy += 1;\\n  }\\n}\\n`,
  pipeline:`pipeline tools {\\n  step nexus_ide("out/nf_ide.html");\\n  step nexus_idle("out/nf_idle.html");\\n  step summary();\\n}\\n`,
  fusion:`step fusion_control_sim("ctrl_tok", 120, {"target_q":0.04});\\nstep fusion_sweep("sweep_tok", {"heating_mw":[145,165], "zones":[3,4]}, 50, {"engine":"multizone"});\\nstep protein_fold_sim_3d("pep", "MKWVTFISLLFL", 90);\\n`,
  ide:`step nexus_ide("out/nexus_ide.html");\\nstep nexus_idle("out/nexus_idle.html");\\nstep nexus_dev_suite("out/dev_suite");\\n`
};
function activeFile(){return ws.files[ws.active]||null;}
function saveEditorIntoActive(){const f=activeFile(); if(f) f.code=$('nf_editor').value;}
function syncEditor(){const f=activeFile(); $('nf_editor_title').textContent='Editor • '+(f?f.name:'(none)'); $('nf_editor').value=f?f.code:'';}
function pushHistory(type,payload){ws.history.push({t:new Date().toISOString(),type,payload}); if(ws.history.length>80) ws.history.shift();}
function setConsole(msg,meta){$('nf_console').textContent=typeof msg==='string'?msg:JSON.stringify(msg,null,2); $('nf_console_meta').textContent=meta||'ok';}
function renderFiles(){const box=$('nf_files'); box.innerHTML=''; ws.files.forEach((f,i)=>{const row=document.createElement('div'); row.className='file-item'+(i===ws.active?' active':''); row.innerHTML=`<span class='mono mini'>${f.name}</span><span class='muted mini'>${(f.code||'').length}c</span>`; row.onclick=()=>{saveEditorIntoActive(); ws.active=i; syncEditor(); renderFiles(); refreshStats();}; box.appendChild(row);});}
function saveWorkspace(){saveEditorIntoActive(); localStorage.setItem(KEY, JSON.stringify(ws)); pushHistory('save_ws',{files:ws.files.length}); setConsole({ok:true,action:'save_workspace',files:ws.files.length}, 'workspace saved'); renderInspector(); refreshStats();}
function loadWorkspace(){try{const raw=localStorage.getItem(KEY); if(!raw) throw new Error('no saved workspace'); const d=JSON.parse(raw); if(!Array.isArray(d.files)) throw new Error('invalid workspace'); ws={files:d.files,active:Number(d.active)||0,history:Array.isArray(d.history)?d.history:[]}; if(ws.active>=ws.files.length) ws.active=0; syncEditor(); renderFiles(); renderInspector(); refreshStats(); setConsole({ok:true,action:'load_workspace',files:ws.files.length}, 'workspace loaded');}catch(e){setConsole({ok:false,error:e.message},'load failed');}}
function liteFormat(code){return String(code||'').replace(/\\r\\n/g,'\\n').split('\\n').map(s=>s.replace(/\\s+$/,'')).join('\\n').replace(/\\n{3,}/g,'\\n\\n');}
function lint(code){const lines=String(code||'').split(/\\r?\\n/); let depth=0; const errors=[], warnings=[]; let steps=0; lines.forEach((line,idx)=>{const t=line.trim(); if(!t) return; for(const ch of t){if(ch==='{') depth++; if(ch==='}') depth--; if(depth<0){errors.push({line:idx+1,msg:'unmatched }'}); depth=0;}} if(/^step\\s+/.test(t)) steps++; if((/^config\\s+|^state\\s+|^metric\\s+/.test(t)) && !t.endsWith(';')) warnings.push({line:idx+1,msg:'statement usually ends with ;'}); if(/^project\\s+/.test(t) && !t.includes('{')) errors.push({line:idx+1,msg:'project line missing {'}); if(t.includes('TODO')) warnings.push({line:idx+1,msg:'TODO marker'});}); if(depth!==0) errors.push({line:lines.length,msg:'brace depth not zero: '+depth}); if(!/\\bproject\\b/.test(code)) warnings.push({line:1,msg:'missing project block'}); if(!/\\bpipeline\\b/.test(code)) warnings.push({line:1,msg:'missing pipeline block'}); return {ok:errors.length===0, errors, warnings, stats:{lines:lines.length,steps}};}
function astSketch(code){const lines=String(code||'').split(/\\r?\\n/); const project=(lines.find(l=>/^\\s*project\\s+/.test(l))||'').trim(); const pipelines=[]; let current=null; lines.forEach((l,idx)=>{const t=l.trim(); const pm=t.match(/^pipeline\\s+([A-Za-z_][A-Za-z0-9_]*)/); if(pm){current={name:pm[1],line:idx+1,steps:[]}; pipelines.push(current);} const sm=t.match(/^step\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(/); if(sm && current){current.steps.push({name:sm[1],line:idx+1});}}); return {project, pipelines, file:(activeFile()||{}).name||null};}
function runPreview(code){const l=lint(code); const ast=astSketch(code); const steps=(ast.pipelines||[]).flatMap(p=>p.steps.map(s=>({pipeline:p.name,step:s.name,line:s.line}))); const trace=steps.map((s,i)=>({tick:i+1,event:'step:'+s.step,pipeline:s.pipeline,line:s.line})); const snapshot={project:ast.project||'unknown',metrics:{line_count:l.stats.lines,step_count:l.stats.steps,error_count:l.errors.length,warning_count:l.warnings.length,simulated_tick:trace.length},time:new Date().toISOString(),files:ws.files.map(f=>({name:f.name,bytes:(f.code||'').length}))}; pushHistory('run_preview',{steps:steps.length}); return {ok:l.ok, lint:l, ast, trace, snapshot};}
function renderInspector(result){if(!result){$('nf_diag').textContent='No diagnostics yet'; $('nf_trace').textContent='No trace yet'; $('nf_snap').textContent='No snapshot yet'; return;} $('nf_diag').textContent=JSON.stringify(result.lint||{},null,2); $('nf_trace').textContent=JSON.stringify(result.trace||result.ast||{},null,2); $('nf_snap').textContent=JSON.stringify(result.snapshot||{},null,2);}
function refreshStats(){const totalChars=ws.files.reduce((n,f)=>n+String(f.code||'').length,0); $('nf_stats').textContent=`files=${ws.files.length} | chars=${totalChars} | history=${ws.history.length}`; $('nf_chips').innerHTML=ws.files.map(f=>`<span class='chip mono'>${f.name}</span>`).join('');}
function selectTab(name){document.querySelectorAll('.nf_tab').forEach(b=>b.classList.toggle('active', b.dataset.tab===name)); document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active')); const panel=document.getElementById('nf_panel_'+name); if(panel) panel.classList.add('active');}
function insertAtCursor(text){const ta=$('nf_editor'); const start=ta.selectionStart||0,end=ta.selectionEnd||0; ta.value=ta.value.slice(0,start)+text+ta.value.slice(end); ta.selectionStart=ta.selectionEnd=start+text.length; ta.focus();}
$('nf_new_file').onclick=()=>{saveEditorIntoActive(); ws.files.push({name:'untitled'+(ws.files.length+1)+'.nxf',code:''}); ws.active=ws.files.length-1; syncEditor(); renderFiles(); refreshStats();};
$('nf_save_ws').onclick=saveWorkspace;
$('nf_load_ws').onclick=loadWorkspace;
$('nf_download').onclick=()=>{saveEditorIntoActive(); const f=activeFile(); if(!f) return; const blob=new Blob([f.code||''],{type:'text/plain;charset=utf-8'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=f.name||'main.nxf'; a.click();};
$('nf_import').onchange=(e)=>{const file=(e.target.files||[])[0]; if(!file) return; const r=new FileReader(); r.onload=()=>{saveEditorIntoActive(); ws.files.push({name:file.name,code:String(r.result||'')}); ws.active=ws.files.length-1; syncEditor(); renderFiles(); refreshStats(); setConsole({ok:true,imported:file.name},'imported');}; r.readAsText(file);};
$('nf_insert').onclick=()=>insertAtCursor(snippets[$('nf_snippets').value]||'');
$('nf_format').onclick=()=>{ $('nf_editor').value=liteFormat($('nf_editor').value); saveEditorIntoActive(); renderFiles(); setConsole({ok:true,action:'format_lite'}, 'formatted'); };
$('nf_lint').onclick=()=>{saveEditorIntoActive(); const r={lint:lint($('nf_editor').value)}; renderInspector(r); selectTab('diag'); setConsole(r.lint, r.lint.ok?'lint ok':'lint issues'); pushHistory('lint',r.lint.stats); refreshStats();};
$('nf_ast').onclick=()=>{saveEditorIntoActive(); const ast=astSketch($('nf_editor').value); $('nf_trace').textContent=JSON.stringify(ast,null,2); selectTab('trace'); setConsole(ast,'ast sketch'); pushHistory('ast',{pipelines:(ast.pipelines||[]).length});};
$('nf_run').onclick=()=>{saveEditorIntoActive(); const result=runPreview($('nf_editor').value); renderInspector(result); selectTab('trace'); setConsole({ok:result.ok,metrics:result.snapshot.metrics,steps:result.trace.length}, result.ok?'run preview ok':'run preview with lint errors'); refreshStats();};
$('nf_clear_console').onclick=()=>{ $('nf_console').textContent=''; $('nf_console_meta').textContent='cleared'; };
$('nf_gen_idle').onclick=()=>{ window.open(location.href.replace(/[^/]+$/, 'nexus_idle.html'), '_blank'); };
$('nf_theme').onclick=()=>{ accentShift=!accentShift; document.documentElement.style.setProperty('--accent', accentShift ? '#f59e0b' : '#22d3ee'); };
document.querySelectorAll('.nf_tab').forEach(b=>b.onclick=()=>selectTab(b.dataset.tab));
$('nf_editor').addEventListener('input', ()=>{ saveEditorIntoActive(); renderFiles(); refreshStats(); });
window.addEventListener('keydown', (e)=>{ if((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='s'){e.preventDefault(); saveWorkspace();} if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault(); $('nf_run').click();}});
syncEditor(); renderFiles(); refreshStats(); renderInspector(); try{ if(localStorage.getItem(KEY)) loadWorkspace(); }catch(e){} setConsole({ready:true,tool:'NexusFlow IDE'}, 'ready');
})();</script>
"""

    def _nexus_idle_tool_body(self) -> str:
        return """
<style>
.idle-grid{display:grid;grid-template-columns:1.1fr .9fr;gap:12px;margin-top:12px}.repl-log{min-height:300px;max-height:300px}.scratch textarea{min-height:300px}
.cmd{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center}.cmd input{margin:0}.cmd button{width:auto}.hist{max-height:150px;overflow:auto}.tiny{font-size:12px}
@media (max-width:960px){.idle-grid{grid-template-columns:1fr}}
</style>
<div class='idle-grid'>
  <div class='card scratch'>
    <h2>NexusFlow IDLE</h2>
    <div class='muted tiny'>Interactive scratchpad + REPL for quick NexusFlow experiments</div>
    <div class='toolbar'>
      <button id='ni_example'>Load Example</button>
      <button id='ni_lint'>Lint Scratch</button>
      <button id='ni_run'>Run Scratch</button>
      <button id='ni_clear'>Clear Output</button>
    </div>
    <textarea id='ni_scratch' class='mono' spellcheck='false'></textarea>
  </div>
  <div class='card'>
    <h2>REPL</h2>
    <div class='cmd'><input id='ni_cmd' class='mono' placeholder='help, state, tick 5, emit alert, set food=3'><button id='ni_eval'>Eval</button></div>
    <div class='muted tiny'>Commands: <code>help</code>, <code>state</code>, <code>tick N</code>, <code>emit LABEL</code>, <code>set key=value</code>, <code>lint</code>, <code>run</code>, <code>clear</code>, <code>example</code></div>
    <pre id='ni_log' class='repl-log'></pre>
    <h2>State</h2>
    <pre id='ni_state'></pre>
    <h2>History</h2>
    <div id='ni_hist' class='hist muted tiny'></div>
  </div>
</div>
<script>(function(){
const $=id=>document.getElementById(id);
const K='nexusflow.idle.v1';
const example=`project "Scratch" {
  state food = 4;
  metric food_left = food;
  pipeline preview {
    step simulate(1);
    step summary();
  }
}`;
let runtime={tick:0,vars:{food:0},events:[],history:[]};
function save(){try{localStorage.setItem(K, JSON.stringify({scratch:$('ni_scratch').value,runtime}));}catch(e){}}
function load(){try{const raw=localStorage.getItem(K); if(!raw) return; const d=JSON.parse(raw); if(typeof d.scratch==='string') $('ni_scratch').value=d.scratch; if(d.runtime&&typeof d.runtime==='object') runtime=d.runtime;}catch(e){}}
function push(line){const out=$('ni_log'); out.textContent += (out.textContent?'\\n':'')+'['+new Date().toISOString()+'] '+line; out.scrollTop=out.scrollHeight;}
function render(){ $('ni_state').textContent=JSON.stringify(runtime,null,2); $('ni_hist').innerHTML=(runtime.history||[]).slice(-20).reverse().map(h=>`<div><code>${h.cmd}</code> <span class='tiny'>→ ${h.ok?'ok':'err'}</span></div>`).join('') || '<div>no history</div>'; save(); }
function lintScratch(){ const code=$('ni_scratch').value||''; const lines=code.split(/\\r?\\n/); let depth=0; const errs=[], warns=[]; let steps=0; lines.forEach((line,idx)=>{ const t=line.trim(); for(const ch of t){ if(ch==='{') depth++; if(ch==='}') depth--; } if(/^step\\s+/.test(t)) steps++; if(/^project\\s+/.test(t)&&!t.includes('{')) errs.push('line '+(idx+1)+': project missing {'); if(/^state\\s+/.test(t)&&t&&!t.endsWith(';')) warns.push('line '+(idx+1)+': state usually ends with ;'); }); if(depth!==0) errs.push('brace depth not zero: '+depth); if(!/\\bpipeline\\b/.test(code)) warns.push('no pipeline block'); return {ok:errs.length===0,errors:errs,warnings:warns,steps,lines:lines.length}; }
function runScratch(){ const l=lintScratch(); const code=$('ni_scratch').value||''; const stepNames=(code.match(/^\\s*step\\s+([A-Za-z_][A-Za-z0-9_]*)/gm)||[]).map(x=>x.replace(/^\\s*step\\s+/,'').replace(/\\(.*/,'')); const trace=stepNames.map((s,i)=>({tick:runtime.tick+i+1,event:'step:'+s})); runtime.tick += stepNames.length; runtime.events=(runtime.events||[]).concat(trace).slice(-100); return {ok:l.ok,lint:l,trace,summary:{ticks:runtime.tick,steps:stepNames.length,events:runtime.events.length}}; }
function record(cmd,ok,result){ runtime.history=(runtime.history||[]); runtime.history.push({ts:new Date().toISOString(),cmd,ok,result}); runtime.history=runtime.history.slice(-100); render(); }
function evalCmd(raw){ const cmd=String(raw||'').trim(); if(!cmd) return; try{ if(cmd==='help'){ const msg={commands:['help','state','tick N','emit LABEL','set k=v','lint','run','clear','example']}; push(JSON.stringify(msg)); record(cmd,true,msg); return; } if(cmd==='clear'){ $('ni_log').textContent=''; record(cmd,true,{cleared:true}); return; } if(cmd==='state'){ push(JSON.stringify(runtime)); record(cmd,true,{state:true}); return; } if(cmd==='example'){ $('ni_scratch').value=example; push('loaded example scratch program'); record(cmd,true,{example:true}); render(); return; } if(cmd==='lint'){ const r=lintScratch(); push(JSON.stringify(r)); record(cmd,r.ok,r); return; } if(cmd==='run'){ const r=runScratch(); push(JSON.stringify(r)); record(cmd,r.ok,r); return; } const tickm=cmd.match(/^tick\\s+(\\d+)$/i); if(tickm){ runtime.tick += Number(tickm[1]); push('tick -> '+runtime.tick); record(cmd,true,{tick:runtime.tick}); return; } const emitm=cmd.match(/^emit\\s+(.+)$/i); if(emitm){ const evt={tick:runtime.tick,event:emitm[1].trim()}; runtime.events=(runtime.events||[]).concat([evt]).slice(-100); push(JSON.stringify(evt)); record(cmd,true,evt); return; } const setm=cmd.match(/^set\\s+([A-Za-z_][A-Za-z0-9_]*)=(.+)$/i); if(setm){ const key=setm[1], rawv=setm[2].trim(); let v=rawv; if(/^[+-]?\\d+(\\.\\d+)?$/.test(rawv)) v=Number(rawv); runtime.vars=runtime.vars||{}; runtime.vars[key]=v; push('set '+key+'='+JSON.stringify(v)); record(cmd,true,{key,value:v}); return; } throw new Error('Unknown command. Try help'); } catch(e){ push('error: '+e.message); record(cmd,false,{error:e.message}); } }
$('ni_example').onclick=()=>{ $('ni_scratch').value=example; render(); };
$('ni_lint').onclick=()=>evalCmd('lint');
$('ni_run').onclick=()=>evalCmd('run');
$('ni_clear').onclick=()=>{ $('ni_log').textContent=''; };
$('ni_eval').onclick=()=>{ evalCmd($('ni_cmd').value); $('ni_cmd').value=''; $('ni_cmd').focus(); };
$('ni_cmd').addEventListener('keydown', e=>{ if(e.key==='Enter'){ e.preventDefault(); $('ni_eval').click(); }});
$('ni_scratch').addEventListener('input', render);
load(); if(!$('ni_scratch').value) $('ni_scratch').value=example; render(); push('NexusFlow IDLE ready • type help');
})();</script>
"""

    def _web_tool_template_html(self, kind: str, title: str) -> str:
        k = (kind or "json_lab").lower().strip()
        if k == "interaction_lab":
            body = """
<div class='row'><div class='card'><h2>Interaction Surface</h2><div id='zone' tabindex='0' class='zone'>Move / click / type / drag here</div><div class='toolbar'><button id='clear'>Clear</button><button id='dump'>Dump JSON</button></div></div><div class='card'><h2>Events</h2><div id='counts' class='muted'></div><pre id='out'></pre></div></div>
<script>(function(){const z=document.getElementById('zone'),o=document.getElementById('out'),c=document.getElementById('counts'),log=[],m={};function push(t,e){const item={t:new Date().toISOString(),evt:t,x:Math.round(e.offsetX||0),y:Math.round(e.offsetY||0),k:e.key||null,b:e.button??null};log.push(item);if(log.length>200)log.shift();m[t]=(m[t]||0)+1;c.textContent=Object.entries(m).map(([a,b])=>a+':'+b).join(' | ');o.textContent=JSON.stringify(log,null,2);}['pointermove','pointerdown','pointerup','click','wheel','keydown','keyup','dragover','drop'].forEach(t=>z.addEventListener(t,e=>{if(t==='dragover'||t==='drop')e.preventDefault();push(t,e);}));z.focus();document.getElementById('clear').onclick=()=>{log.length=0;for(const k of Object.keys(m))delete m[k];o.textContent='';c.textContent='cleared';};document.getElementById('dump').onclick=()=>{const b=new Blob([JSON.stringify(log,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='interaction-log.json';a.click();};})();</script>
"""
        elif k == "regex_lab":
            body = """
<div class='row'><div class='card'><h2>Regex</h2><input id='pat' value='(nexus)(flow)' class='mono'><input id='flags' value='gi' class='mono'><textarea id='txt' rows='12' class='mono'>NexusFlow builds tools.\nNEXUSFLOW builds simulations.</textarea><div class='toolbar'><button id='run'>Run</button></div></div><div class='card'><h2>Result</h2><div id='meta' class='muted'></div><pre id='out'></pre></div></div>
<script>(function(){const $=id=>document.getElementById(id);function run(){try{const r=new RegExp($('pat').value,$('flags').value);const t=$('txt').value;const ms=Array.from(t.matchAll(r)).map(m=>({match:m[0],index:m.index,groups:Array.from(m).slice(1)}));$('meta').textContent='matches='+ms.length;$('out').textContent=JSON.stringify(ms,null,2);}catch(e){$('meta').textContent='error: '+e.message;$('out').textContent='';}}$('run').onclick=run;run();})();</script>
"""
        elif k == "prompt_studio":
            body = """
<div class='row'><div class='card'><h2>Compose Prompt</h2><textarea id='sys' rows='5'>You are a pragmatic coding assistant.</textarea><textarea id='ctx' rows='5'>Project context...</textarea><textarea id='usr' rows='5'>User task...</textarea><div class='toolbar'><button id='build'>Build</button><button id='copy'>Copy</button></div></div><div class='card'><h2>Output</h2><div id='meta' class='muted'></div><pre id='out'></pre></div></div>
<script>(function(){const $=id=>document.getElementById(id);function build(){const p='[SYSTEM]\\n'+$('sys').value.trim()+'\\n\\n[CONTEXT]\\n'+$('ctx').value.trim()+'\\n\\n[USER]\\n'+$('usr').value.trim();$('out').textContent=p;$('meta').textContent='chars='+p.length+' | est_tokens≈'+Math.ceil(p.length/4);}$('build').onclick=build;$('copy').onclick=()=>navigator.clipboard&&navigator.clipboard.writeText($('out').textContent);build();})();</script>
"""
        elif k == "api_builder":
            body = """
<div class='row'><div class='card'><h2>API Builder</h2><select id='method'><option>GET</option><option>POST</option><option>PUT</option></select><input id='url' class='mono' value='https://api.example.com/items'><textarea id='hdrs' rows='5' class='mono'>{\"Content-Type\":\"application/json\"}</textarea><textarea id='body' rows='8' class='mono'>{\"name\":\"example\"}</textarea><div class='toolbar'><button id='build'>Generate</button></div></div><div class='card'><h2>Fetch Snippet</h2><pre id='out'></pre><h2>curl</h2><pre id='curl'></pre></div></div>
<script>(function(){const $=id=>document.getElementById(id);function build(){let h={};try{h=JSON.parse($('hdrs').value||'{}');}catch{} const m=$('method').value,u=$('url').value.trim(),b=$('body').value;const cfg={method:m,headers:h};if(!['GET','DELETE'].includes(m))cfg.body=b;$('out').textContent='fetch('+JSON.stringify(u)+', '+JSON.stringify(cfg,null,2)+')';$('curl').textContent='curl -X '+m+' '+Object.entries(h).map(([k,v])=>'-H '+JSON.stringify(k+': '+v)).join(' ')+' '+JSON.stringify(u)+(!['GET','DELETE'].includes(m)?' --data '+JSON.stringify(b):'');}$('build').onclick=build;build();})();</script>
"""
        elif k == "mock_server_lab":
            body = """
<div class='row'><div class='card'><h2>Mock API Lab</h2><textarea id='routes' rows='14' class='mono'>{\n  \"/api/ping\": {\"status\": 200, \"body\": {\"pong\": true}},\n  \"/api/items\": {\"status\": 200, \"body\": [{\"id\":1,\"name\":\"alpha\"},{\"id\":2,\"name\":\"beta\"}]}\n}</textarea><input id='path' class='mono' value='/api/ping'><select id='method'><option>GET</option><option>POST</option></select><textarea id='req' rows='5' class='mono'>{\"demo\":1}</textarea><div class='toolbar'><button id='serve'>Install fetch mock</button><button id='test'>Test Route</button></div></div><div class='card'><h2>Output</h2><div id='meta' class='muted'></div><pre id='out'></pre></div></div>
<script>(function(){const $=id=>document.getElementById(id);let installed=false;function parseRoutes(){return JSON.parse($('routes').value||'{}');}function install(){const real=window.fetch.bind(window);window.fetch=async (input,init={})=>{const url=typeof input==='string'?input:input.url;const u=new URL(url, location.href);const routes=parseRoutes();const r=routes[u.pathname];if(r){const status=Number(r.status||200);const body=r.body===undefined?null:r.body;return new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json','X-Mock':'NexusFlow'}});}return real(input,init);};installed=true;$('meta').textContent='fetch mock installed';}async function test(){try{if(!installed) install();const p=$('path').value.trim();const m=$('method').value;const init={method:m};if(m!=='GET') init.body=$('req').value;const resp=await fetch(p, init);const text=await resp.text();$('meta').textContent='status='+resp.status+' mock='+resp.headers.get('X-Mock');$('out').textContent=text;}catch(e){$('meta').textContent='error: '+e.message;}}$('serve').onclick=install;$('test').onclick=test;})();</script>
"""
        elif k == "websocket_lab":
            body = """
<div class='row'><div class='card'><h2>WebSocket Lab</h2><input id='url' class='mono' value='wss://echo.websocket.events'><input id='msg' class='mono' value='hello from nexusflow'><div class='toolbar'><button id='connect'>Connect</button><button id='send'>Send</button><button id='close'>Close</button></div></div><div class='card'><h2>Session Log</h2><div id='meta' class='muted'>disconnected</div><pre id='out'></pre></div></div>
<script>(function(){const $=id=>document.getElementById(id);let ws=null;const log=[];function push(x){log.push('['+new Date().toISOString()+'] '+x);$('out').textContent=log.join('\\n');$('out').scrollTop=$('out').scrollHeight;}$('connect').onclick=()=>{try{if(ws&&ws.readyState<2)return;ws=new WebSocket($('url').value.trim());$('meta').textContent='connecting';ws.onopen=()=>{$('meta').textContent='open';push('open')};ws.onmessage=e=>push('message '+String(e.data));ws.onerror=()=>push('error');ws.onclose=e=>{$('meta').textContent='closed';push('close code='+e.code)};}catch(e){push('connect error '+e.message)}};$('send').onclick=()=>{if(!ws||ws.readyState!==1){push('not connected');return;}ws.send($('msg').value);push('send '+$('msg').value);};$('close').onclick=()=>{if(ws)ws.close();};})();</script>
"""
        elif k == "nexus_ide":
            body = self._nexus_ide_tool_body()
        elif k == "nexus_idle":
            body = self._nexus_idle_tool_body()
        else:
            body = """
<div class='row'><div class='card'><h2>JSON Lab</h2><textarea id='txt' rows='14' class='mono'>{\"project\":\"NexusFlow\",\"features\":[\"web tools\",\"fusion\",\"protein\"]}</textarea><input id='path' class='mono' placeholder='path e.g. features.0'><div class='toolbar'><button id='run'>Parse</button></div></div><div class='card'><h2>Result</h2><div id='meta' class='muted'></div><pre id='out'></pre><h2>Path</h2><pre id='pathout'></pre></div></div>
<script>(function(){const $=id=>document.getElementById(id);function get(v,p){if(!p)return v;let c=v;for(const part of String(p).split('.')){if(part==='')continue;if(Array.isArray(c)){const i=Number(part);if(!Number.isInteger(i)||i<0||i>=c.length)return undefined;c=c[i];continue;}if(c&&typeof c==='object'){if(!(part in c))return undefined;c=c[part];continue;}return undefined;}return c;}function run(){try{const v=JSON.parse($('txt').value);$('meta').textContent='ok';$('out').textContent=JSON.stringify(v,null,2);$('pathout').textContent=JSON.stringify(get(v,$('path').value.trim()),null,2);}catch(e){$('meta').textContent='error: '+e.message;$('out').textContent='';$('pathout').textContent='';}}$('run').onclick=run;run();})();</script>
"""
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{py_html.escape(title)}</title>
<style>
:root{{--bg:#07111f;--panel:#101b31;--line:#23324d;--text:#e2e8f0;--muted:#9fb1c9;--accent:#22d3ee;}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Segoe UI,system-ui,sans-serif;color:var(--text);background:radial-gradient(900px 320px at 0% 0%,rgba(34,211,238,.11),transparent 60%),linear-gradient(180deg,#030712,var(--bg));}}
.wrap{{max-width:1100px;margin:0 auto;padding:18px}} .hero{{border:1px solid var(--line);background:rgba(16,27,49,.76);border-radius:14px;padding:14px}} .hero h1{{margin:0;font-size:22px}}
.muted{{color:var(--muted)}} .row{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}} .card{{border:1px solid var(--line);border-radius:12px;background:rgba(16,27,49,.75);padding:12px}}
.card h2{{margin:0 0 8px;font-size:14px}} textarea,input,select,button{{width:100%;margin:6px 0;padding:9px;border-radius:10px;border:1px solid rgba(255,255,255,.12);background:rgba(2,6,23,.6);color:var(--text);font:inherit}}
button{{cursor:pointer;background:linear-gradient(90deg,rgba(34,211,238,.14),rgba(134,239,172,.12))}} pre{{margin:6px 0 0;background:rgba(2,6,23,.55);border:1px solid rgba(255,255,255,.06);border-radius:10px;padding:10px;min-height:70px;overflow:auto;font-size:12px}}
.mono{{font-family:Consolas,'Cascadia Code',monospace}} .toolbar{{display:flex;gap:8px;flex-wrap:wrap}} .toolbar>*{{width:auto}} .zone{{height:260px;border-radius:10px;border:1px dashed rgba(255,255,255,.18);display:grid;place-items:center;background:rgba(34,211,238,.03)}} @media (max-width:900px){{.row{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap"><section class="hero"><h1>{py_html.escape(title)}</h1><div class="muted">NexusFlow generated web tool • kind: <code>{py_html.escape(k)}</code></div></section>{body}</div></body></html>"""

    def _record_web_tool(self, kind: str, out_path: Path, title: str, suite: Optional[str] = None) -> None:
        meta = {"kind": kind, "path": str(out_path), "title": title, "suite": suite, "bytes": out_path.stat().st_size if out_path.exists() else 0}
        with self._lock:
            self.web_tools[str(out_path)] = meta
            tick_now = self.tick_count
        evt = {"tick": tick_now, "event": "web_tool_generate", "kind": kind, "path": str(out_path)}
        if suite:
            evt["suite"] = suite
        self._append_event(evt)

    def _web_tool_generate(self, kind: str, rel_path: str, title: Optional[str] = None) -> Path:
        out = self._resolve_output_path(rel_path)
        final_title = title or f"NexusFlow {str(kind).replace('_', ' ').title()}"
        out.write_text(self._web_tool_template_html(str(kind), final_title), encoding="utf-8")
        self._record_web_tool(str(kind), out, final_title)
        return out

    def _build_web_tool_suite_index(self, preset: str, items: List[Dict[str, Any]]) -> str:
        links = "".join(
            f"<a class='tile' href='{py_html.escape(i['file'])}'><div class='k'>{py_html.escape(i['kind'])}</div><div>{py_html.escape(i['title'])}</div></a>"
            for i in items
        )
        return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>NexusFlow Web Tool Suite</title>
<style>:root{{--bg:#07111f;--card:#101b31;--line:#23324d;--text:#e2e8f0;--muted:#9fb1c9;--accent:#22d3ee}}*{{box-sizing:border-box}}body{{margin:0;font-family:Segoe UI,system-ui,sans-serif;color:var(--text);background:radial-gradient(900px 320px at 0% 0%,rgba(34,211,238,.11),transparent 60%),linear-gradient(180deg,#030712,var(--bg))}}.wrap{{max-width:1000px;margin:0 auto;padding:20px}}.hero{{border:1px solid var(--line);background:rgba(16,27,49,.76);border-radius:14px;padding:14px}}.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px}}.tile{{display:block;color:inherit;text-decoration:none;border:1px solid var(--line);background:rgba(16,27,49,.75);border-radius:12px;padding:12px}}.tile:hover{{border-color:rgba(34,211,238,.35)}}.k{{color:var(--accent);font-size:12px;text-transform:uppercase}}</style>
</head><body><div class='wrap'><section class='hero'><h1 style='margin:0'>NexusFlow {py_html.escape(preset.title())} Web Tools</h1><div class='muted'>Generated series of web tools and interaction utilities.</div></section><section class='grid'>{links}</section></div></body></html>"""

    def _web_tool_suite(self, rel_dir: str, preset: str = "lab") -> Dict[str, Any]:
        out_dir = self._resolve_output_path(rel_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        preset_key = str(preset or "lab").lower()
        kinds_by_preset = {
            "lab": ["json_lab", "regex_lab", "prompt_studio", "interaction_lab", "api_builder"],
            "dev": ["json_lab", "regex_lab", "api_builder", "interaction_lab"],
            "ai": ["prompt_studio", "json_lab", "interaction_lab", "api_builder"],
            "science": ["json_lab", "regex_lab", "interaction_lab", "prompt_studio"],
            "live_api": ["api_builder", "mock_server_lab", "websocket_lab", "interaction_lab", "json_lab"],
            "nexus_dev": ["nexus_ide", "nexus_idle", "json_lab", "prompt_studio", "interaction_lab"],
            "ide": ["nexus_ide", "nexus_idle", "json_lab"],
        }
        kinds = kinds_by_preset.get(preset_key, kinds_by_preset["lab"])
        suite = f"{preset_key}_suite"
        items: List[Dict[str, Any]] = []
        for i, kind in enumerate(kinds, start=1):
            fname = f"{i:02d}_{self._web_slugify(kind)}.html"
            title = f"NexusFlow {kind.replace('_', ' ').title()}"
            path = out_dir / fname
            path.write_text(self._web_tool_template_html(kind, title), encoding="utf-8")
            self._record_web_tool(kind, path, title, suite=suite)
            items.append({"kind": kind, "file": fname, "title": title})
        index_path = out_dir / "index.html"
        index_path.write_text(self._build_web_tool_suite_index(preset_key, items), encoding="utf-8")
        self._record_web_tool("suite_index", index_path, f"NexusFlow {preset_key.title()} Web Tools", suite=suite)
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "web_tool_suite", "preset": preset_key, "count": len(items)})
        return {"preset": preset_key, "dir": str(out_dir), "count": len(items), "items": items}

    def _http_get_text(
        self,
        url: str,
        timeout_sec: float = 15.0,
        *,
        headers: Optional[Dict[str, Any]] = None,
        auth_preset: Optional[str] = None,
    ) -> str:
        result = self._http_request("GET", url, timeout_sec=timeout_sec, headers=headers, auth_preset=auth_preset)
        if not result.get("ok"):
            raise RuntimeErrorNF(f"http_get_text failed: {result.get('status')} {result.get('error', '')}".strip())
        return (result.get("body_bytes") or b"").decode("utf-8", errors="replace")

    def _http_get_json(
        self,
        url: str,
        timeout_sec: float = 15.0,
        *,
        headers: Optional[Dict[str, Any]] = None,
        auth_preset: Optional[str] = None,
    ) -> Any:
        return json.loads(self._http_get_text(url, timeout_sec, headers=headers, auth_preset=auth_preset))

    def _http_post_json(
        self,
        url: str,
        payload: Any,
        timeout_sec: float = 15.0,
        headers: Optional[Dict[str, Any]] = None,
        auth_preset: Optional[str] = None,
    ) -> Any:
        hdrs = {"Content-Type": "application/json"}
        if isinstance(headers, dict):
            hdrs.update({str(k): str(v) for k, v in headers.items()})
        result = self._http_request(
            "POST",
            url,
            headers=hdrs,
            auth_preset=auth_preset,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout_sec=timeout_sec,
        )
        if not result.get("ok"):
            raise RuntimeErrorNF(f"http_post_json failed: {result.get('status')} {result.get('error', '')}".strip())
        raw = result.get("body_bytes") or b""
        return json.loads(raw.decode("utf-8", errors="replace")) if raw else None

    def _http_download(
        self,
        url: str,
        rel_path: str,
        timeout_sec: float = 30.0,
        *,
        headers: Optional[Dict[str, Any]] = None,
        auth_preset: Optional[str] = None,
    ) -> Path:
        result = self._http_request("GET", url, timeout_sec=timeout_sec, headers=headers, auth_preset=auth_preset)
        if not result.get("ok"):
            raise RuntimeErrorNF(f"http_download failed: {result.get('status')} {result.get('error', '')}".strip())
        out = self._resolve_output_path(rel_path)
        body = result.get("body_bytes") or b""
        out.write_bytes(body)
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "http_download", "url": url, "path": str(out), "bytes": len(body)})
        return out

    def _http_fetch_json_to_file(
        self,
        url: str,
        rel_path: str,
        timeout_sec: float = 15.0,
        *,
        headers: Optional[Dict[str, Any]] = None,
        auth_preset: Optional[str] = None,
    ) -> Path:
        data = self._http_get_json(url, timeout_sec, headers=headers, auth_preset=auth_preset)
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "http_fetch_json", "url": url, "path": str(out)})
        return out

    def _export_http_history_json(self, rel_path: str, limit: Optional[int] = None) -> Path:
        out = self._resolve_output_path(rel_path)
        with self._lock:
            ops = copy.deepcopy(self.http_ops)
            presets = copy.deepcopy(self.http_auth_presets)
            tick_now = self.tick_count
        if limit is not None:
            ops = ops[-max(0, int(limit)) :]
        out.write_text(json.dumps({"project": self.project.name, "tick": tick_now, "ops": ops, "auth_presets": presets}, indent=2), encoding="utf-8")
        self._append_event({"tick": tick_now, "event": "export_http_history_json", "path": str(out), "count": len(ops)})
        return out

    def _fusion_sim(self, run_name: str, steps: int = 120, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(cfg, dict):
            mode = str(cfg.get("mode", "")).lower()
            if mode in {"multizone", "advanced", "tokamak_multizone"}:
                return self._fusion_sim_multizone(run_name, steps=steps, cfg=cfg)
        rng = self._get_rng()
        c: Dict[str, Any] = {
            "dt": 0.02,
            "heating_mw": 125.0,
            "magnetic_field_t": 5.5,
            "density_20m3": 1.1,
            "confinement_s": 0.9,
            "impurity_fraction": 0.025,
            "fueling_rate": 0.010,
            "control_gain": 0.11,
            "wall_load_limit": 2.4,
        }
        if isinstance(cfg, dict):
            c.update(cfg)
        steps = max(1, int(steps))
        dt = max(0.001, float(c.get("dt", 0.02)))
        T = max(0.5, float(c.get("temp_keV", 8.5)))
        n = max(0.1, float(c.get("density_20m3", 1.1)))
        tau = max(0.05, float(c.get("confinement_s", 0.9)))
        Z = min(0.3, max(0.0, float(c.get("impurity_fraction", 0.025))))
        B = max(1.0, float(c.get("magnetic_field_t", 5.5)))
        Paux = max(1.0, float(c.get("heating_mw", 125.0)))
        fuel = float(c.get("fueling_rate", 0.01))
        gain = float(c.get("control_gain", 0.11))
        wall_limit = max(0.1, float(c.get("wall_load_limit", 2.4)))
        disruptions = 0
        ignition_steps = 0
        trace: List[Dict[str, Any]] = []

        for i in range(steps):
            P = Paux * (1.0 + 0.03 * rng.gauss(0, 1))
            n += dt * (fuel - 0.011 * n - 0.08 * Z) + rng.gauss(0, 0.002)
            n = max(0.15, min(4.0, n))
            beta = (n * T) / max(1.0, (B * B) * 1.9)
            tau += dt * (gain * (1.0 - 0.95 * beta) + 0.01 * (B - 5.0) - 0.3 * Z) + rng.gauss(0, 0.002)
            tau = max(0.04, min(4.0, tau))
            alpha = 0.20 * n * n * max(T - 4.0, 0.0) * (1 - 1.3 * Z)
            rad = 0.11 * n * n * math.sqrt(max(T, 0.1)) * (1 + 7 * Z)
            trans = (T / max(tau, 0.05)) * (0.55 + 0.05 * max(beta - 0.5, 0))
            T += dt * (0.023 * P + 0.06 * alpha - rad - trans) + rng.gauss(0, 0.04)
            T = max(0.5, min(80.0, T))
            Z += dt * (0.0015 + 0.0008 * max(P - 140.0, 0.0) / 100.0 - 0.003 * n) + rng.gauss(0, 0.00035)
            Z = min(0.3, max(0.0, Z))
            triple = T * n * tau
            Pf = 0.18 * n * n * (max(T - 2.0, 0.0) ** 2) / (T + 8.0)
            Pin = P + 6.0
            q = Pf / max(0.001, Pin)
            wall = Pf / max(0.8, 4.0 + B * 0.3)
            risk = max(0.0, min(1.0, (beta - 0.85) * 1.2 + max(0.0, Z - 0.08) * 4.0 + (1.8 - tau) * 0.08))
            if q >= 1.0 and T > 8.0:
                ignition_steps += 1
            disrupted = False
            if risk > 0.72 and rng.random() < min(0.6, risk * 0.5):
                disruptions += 1
                disrupted = True
                T *= 0.55
                tau *= 0.62
                n *= 0.88
                with self._lock:
                    tick_now = self.tick_count
                self._append_event({"tick": tick_now, "event": "fusion_disruption", "run": run_name, "step": i, "risk": round(risk, 4)})
            trace.append({
                "step": i,
                "temp_keV": round(T, 4),
                "density_20m3": round(n, 4),
                "confinement_s": round(tau, 4),
                "beta": round(beta, 4),
                "q_estimate": round(q, 5),
                "fusion_power_mw": round(Pf, 4),
                "net_input_mw": round(Pin, 4),
                "wall_load_mw_m2": round(wall, 4),
                "impurity_fraction": round(Z, 5),
                "triple_product": round(triple, 5),
                "disruption_risk": round(risk, 4),
                "disrupted": disrupted,
            })
        final = trace[-1]
        run = {
            "name": run_name,
            "kind": "fusion_reactor",
            "steps": steps,
            "config": c,
            "trace": trace[-600:],
            "final": final,
            "metrics": {
                "q_estimate": final["q_estimate"],
                "triple_product": final["triple_product"],
                "fusion_power_mw": final["fusion_power_mw"],
                "wall_load_mw_m2": final["wall_load_mw_m2"],
                "lawson_margin": round(final["triple_product"] / 12.0, 5),
                "wall_load_ok": bool(final["wall_load_mw_m2"] <= wall_limit),
                "disruptions": disruptions,
                "sustained_ignition_steps": ignition_steps,
                "stability_score": round(max(0.0, 1.0 - final["disruption_risk"] - disruptions * 0.08), 5),
            },
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.fusion_runs[run_name] = run
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "fusion_sim", "run": run_name, "steps": steps, "q": final["q_estimate"]})
        return run

    def _export_fusion_json(self, run_name: str, rel_path: str) -> Path:
        run = self.fusion_runs.get(run_name)
        if run is None:
            raise RuntimeErrorNF(f"Unknown fusion run: {run_name}")
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps({"project": self.project.name, "fusion": run}, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "export_fusion_json", "run": run_name, "path": str(out)})
        return out

    def _fusion_sim_multizone(self, run_name: str, steps: int = 120, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rng = self._get_rng()
        c: Dict[str, Any] = {
            "mode": "multizone",
            "dt": 0.02,
            "zones": 3,
            "heating_mw": 145.0,
            "magnetic_field_t": 5.8,
            "fueling_rate": 0.011,
            "core_heating_fraction": 0.62,
            "transport_coupling": 0.16,
            "edge_radiation_boost": 1.6,
            "impurity_fraction": 0.022,
            "wall_load_limit": 2.5,
            "control_gain": 0.12,
        }
        if isinstance(cfg, dict):
            c.update(cfg)
        steps = max(1, int(steps))
        dt = max(0.001, float(c.get("dt", 0.02)))
        zones_n = max(2, min(5, int(c.get("zones", 3))))
        B = max(1.0, float(c.get("magnetic_field_t", 5.8)))
        Paux = max(1.0, float(c.get("heating_mw", 145.0)))
        fuel = float(c.get("fueling_rate", 0.011))
        core_frac = min(0.9, max(0.3, float(c.get("core_heating_fraction", 0.62))))
        coupling = max(0.01, float(c.get("transport_coupling", 0.16)))
        edge_rad_boost = max(1.0, float(c.get("edge_radiation_boost", 1.6)))
        impurity = min(0.3, max(0.0, float(c.get("impurity_fraction", 0.022))))
        control_gain = float(c.get("control_gain", 0.12))
        wall_limit = max(0.1, float(c.get("wall_load_limit", 2.5)))

        temps = [max(0.5, 12.0 - i * 3.0 + rng.random()) for i in range(zones_n)]
        dens = [max(0.2, 1.2 - i * 0.12 + 0.05 * rng.random()) for i in range(zones_n)]
        taus = [max(0.05, 1.0 - i * 0.12 + 0.05 * rng.random()) for i in range(zones_n)]
        zone_weights = [max(0.1, (zones_n - i) / sum(range(1, zones_n + 1))) for i in range(zones_n)]
        sw = sum(zone_weights)
        zone_weights = [w / sw for w in zone_weights]
        disruptions = 0
        ignition_steps = 0
        trace: List[Dict[str, Any]] = []

        for step in range(steps):
            heating_split = []
            remaining = 1.0 - core_frac
            for i in range(zones_n):
                if i == 0:
                    heating_split.append(core_frac)
                else:
                    heating_split.append(remaining / (zones_n - 1))
            fusion_powers: List[float] = []
            qs: List[float] = []
            betas: List[float] = []
            walls: List[float] = []
            zones_payload: List[Dict[str, Any]] = []

            mean_temp = sum(temps) / zones_n
            mean_den = sum(dens) / zones_n
            mean_tau = sum(taus) / zones_n
            radial_stress = 0.0

            next_temps = list(temps)
            next_dens = list(dens)
            next_taus = list(taus)

            for i in range(zones_n):
                T = temps[i]
                n = dens[i]
                tau = taus[i]
                zone_heat = Paux * heating_split[i] * (1.0 + 0.03 * rng.gauss(0, 1))
                coupling_term_T = coupling * ((temps[i - 1] if i > 0 else mean_temp) + (temps[i + 1] if i < zones_n - 1 else mean_temp) - 2 * T)
                coupling_term_n = 0.7 * coupling * ((dens[i - 1] if i > 0 else mean_den) + (dens[i + 1] if i < zones_n - 1 else mean_den) - 2 * n)
                edge_factor = 1.0 + (edge_rad_boost - 1.0) * (i / max(1, zones_n - 1))
                beta = (n * T) / max(1.0, (B * B) * (1.7 + 0.1 * i))
                alpha = 0.17 * n * n * max(T - 2.0, 0.0) * (1.0 - impurity * (1.1 + 0.25 * i))
                rad = 0.10 * edge_factor * n * n * math.sqrt(max(T, 0.1)) * (1.0 + impurity * 6.5)
                trans = (T / max(tau, 0.04)) * (0.50 + 0.07 * max(beta - 0.55, 0.0))
                next_temps[i] = max(0.5, min(80.0, T + dt * (0.020 * zone_heat + 0.065 * alpha - rad - trans + coupling_term_T) + rng.gauss(0, 0.03)))
                next_dens[i] = max(0.12, min(4.5, n + dt * (fuel * (1.15 if i == 0 else 0.85) - 0.010 * n + coupling_term_n - 0.03 * impurity) + rng.gauss(0, 0.0015)))
                target_tau = max(0.08, 1.15 - 0.20 * i - 0.55 * max(beta - 0.75, 0.0))
                next_taus[i] = max(0.04, min(4.0, tau + dt * (control_gain * (target_tau - tau) + 0.01 * (B - 5.0)) + rng.gauss(0, 0.0015)))

                pf = 0.16 * n * n * (max(T - 2.0, 0.0) ** 2) / (T + 8.0)
                pin = zone_heat + 2.0
                qz = pf / max(0.001, pin)
                wallz = pf / max(0.8, 5.2 + B * 0.2 + (zones_n - i) * 0.1)
                fusion_powers.append(pf)
                qs.append(qz)
                betas.append(beta)
                walls.append(wallz)
                radial_stress += abs(coupling_term_T)
                zones_payload.append({
                    "zone": i,
                    "temp_keV": round(T, 4),
                    "density_20m3": round(n, 4),
                    "confinement_s": round(tau, 4),
                    "beta": round(beta, 4),
                    "fusion_power_mw": round(pf, 4),
                    "q_zone": round(qz, 5),
                    "wall_load_mw_m2": round(wallz, 4),
                })

            temps, dens, taus = next_temps, next_dens, next_taus
            impurity = min(0.3, max(0.0, impurity + dt * (0.001 + 0.0007 * max(Paux - 150.0, 0.0) / 100.0 - 0.0018 * mean_den) + rng.gauss(0, 0.0002)))

            temp_w = sum(t * w for t, w in zip(temps, zone_weights))
            den_w = sum(n * w for n, w in zip(dens, zone_weights))
            tau_w = sum(tau * w for tau, w in zip(taus, zone_weights))
            triple = temp_w * den_w * tau_w
            pf_total = sum(fusion_powers)
            pin_total = Paux + 8.0
            q = pf_total / max(0.001, pin_total)
            wall = max(walls) if walls else 0.0
            beta_avg = sum(betas) / len(betas)
            risk = max(0.0, min(1.0, (beta_avg - 0.88) * 1.3 + max(0.0, impurity - 0.07) * 3.5 + radial_stress * 0.08))
            if q >= 1.0 and temp_w > 8.0:
                ignition_steps += 1
            disrupted = False
            if risk > 0.72 and rng.random() < min(0.7, risk * 0.55):
                disruptions += 1
                disrupted = True
                temps = [max(0.5, t * (0.5 if i == 0 else 0.7)) for i, t in enumerate(temps)]
                taus = [max(0.04, t * (0.58 if i == 0 else 0.72)) for i, t in enumerate(taus)]
                dens = [max(0.12, d * 0.9) for d in dens]
                with self._lock:
                    tick_now = self.tick_count
                self._append_event({"tick": tick_now, "event": "fusion_disruption", "run": run_name, "step": step, "risk": round(risk, 4), "mode": "multizone"})

            trace.append({
                "step": step,
                "mode": "multizone",
                "temp_keV": round(temp_w, 4),
                "density_20m3": round(den_w, 4),
                "confinement_s": round(tau_w, 4),
                "beta": round(beta_avg, 4),
                "q_estimate": round(q, 5),
                "fusion_power_mw": round(pf_total, 4),
                "net_input_mw": round(pin_total, 4),
                "wall_load_mw_m2": round(wall, 4),
                "impurity_fraction": round(impurity, 5),
                "triple_product": round(triple, 5),
                "disruption_risk": round(risk, 4),
                "radial_stress": round(radial_stress, 5),
                "disrupted": disrupted,
                "zones": zones_payload,
            })

        final = trace[-1]
        run = {
            "name": run_name,
            "kind": "fusion_reactor_multizone",
            "steps": steps,
            "config": c,
            "trace": trace[-600:],
            "final": final,
            "metrics": {
                "q_estimate": final["q_estimate"],
                "triple_product": final["triple_product"],
                "fusion_power_mw": final["fusion_power_mw"],
                "wall_load_mw_m2": final["wall_load_mw_m2"],
                "lawson_margin": round(final["triple_product"] / 12.0, 5),
                "wall_load_ok": bool(final["wall_load_mw_m2"] <= wall_limit),
                "disruptions": disruptions,
                "sustained_ignition_steps": ignition_steps,
                "stability_score": round(max(0.0, 1.0 - final["disruption_risk"] - disruptions * 0.08), 5),
                "radial_stress": final["radial_stress"],
                "zones": zones_n,
            },
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.fusion_runs[run_name] = run
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "fusion_sim_multizone", "run": run_name, "steps": steps, "q": final["q_estimate"]})
        return run

    def _fusion_control_sim(self, run_name: str, steps: int = 180, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "controller": "pid_mpc_lite",
            "mode": "multizone",
            "cycles": 6,
            "horizon_steps": 30,
            "target_q": 0.08,
            "wall_load_limit": 2.5,
            "heating_mw": 155.0,
            "fueling_rate": 0.011,
            "magnetic_field_t": 5.9,
            "kp": 95.0,
            "ki": 2.5,
            "kd": 22.0,
            "fuel_gain": 0.012,
            "b_gain": 0.09,
            "wall_penalty_gain": 55.0,
            "risk_penalty_gain": 35.0,
            "heating_min_mw": 60.0,
            "heating_max_mw": 260.0,
            "fueling_min": 0.004,
            "fueling_max": 0.03,
            "b_min_t": 3.5,
            "b_max_t": 8.0,
        }
        if isinstance(cfg, dict):
            params.update(cfg)
        total_steps = max(1, int(steps))
        cycles = max(1, int(params.get("cycles", 6)))
        horizon_steps = max(4, int(params.get("horizon_steps", max(6, total_steps // max(1, cycles)))))
        target_q = float(params.get("target_q", 0.08))
        wall_limit = max(0.1, float(params.get("wall_load_limit", params.get("wall_limit", 2.5))))
        heating = float(params.get("heating_mw", 155.0))
        fuel = float(params.get("fueling_rate", 0.011))
        bfield = float(params.get("magnetic_field_t", 5.9))
        kp = float(params.get("kp", 95.0))
        ki = float(params.get("ki", 2.5))
        kd = float(params.get("kd", 22.0))
        fuel_gain = float(params.get("fuel_gain", 0.012))
        b_gain = float(params.get("b_gain", 0.09))
        wall_penalty_gain = float(params.get("wall_penalty_gain", 55.0))
        risk_penalty_gain = float(params.get("risk_penalty_gain", 35.0))
        heat_min = float(params.get("heating_min_mw", 60.0))
        heat_max = float(params.get("heating_max_mw", 260.0))
        fuel_min = float(params.get("fueling_min", 0.004))
        fuel_max = float(params.get("fueling_max", 0.03))
        b_min = float(params.get("b_min_t", 3.5))
        b_max = float(params.get("b_max_t", 8.0))

        control_trace: List[Dict[str, Any]] = []
        cycle_results: List[Dict[str, Any]] = []
        simulated_steps = 0
        integral_err = 0.0
        prev_err = 0.0
        best_score = -1e18
        best_cycle: Optional[Dict[str, Any]] = None
        last_subrun: Optional[Dict[str, Any]] = None
        last_final: Optional[Dict[str, Any]] = None
        last_metrics: Optional[Dict[str, Any]] = None
        last_q = 0.0

        base_sim_cfg = {
            k: copy.deepcopy(v)
            for k, v in params.items()
            if k
            not in {
                "controller",
                "cycles",
                "horizon_steps",
                "target_q",
                "kp",
                "ki",
                "kd",
                "fuel_gain",
                "b_gain",
                "wall_penalty_gain",
                "risk_penalty_gain",
                "heating_min_mw",
                "heating_max_mw",
                "fueling_min",
                "fueling_max",
                "b_min_t",
                "b_max_t",
            }
        }
        base_sim_cfg["mode"] = "multizone"

        for cycle_idx in range(cycles):
            remaining = total_steps - simulated_steps
            if remaining <= 0:
                break
            horizon = min(horizon_steps, remaining)
            sim_cfg = {
                **copy.deepcopy(base_sim_cfg),
                "mode": "multizone",
                "heating_mw": heating,
                "fueling_rate": fuel,
                "magnetic_field_t": bfield,
                "wall_load_limit": wall_limit,
            }
            tmp_name = f"__nf_ctrl_{run_name}_{cycle_idx}"
            subrun = self._fusion_sim_multizone(tmp_name, steps=horizon, cfg=sim_cfg)
            with self._lock:
                self.fusion_runs.pop(tmp_name, None)
            last_subrun = subrun
            final = subrun.get("final", {}) if isinstance(subrun, dict) else {}
            metrics = subrun.get("metrics", {}) if isinstance(subrun, dict) else {}
            last_final = final if isinstance(final, dict) else {}
            last_metrics = metrics if isinstance(metrics, dict) else {}
            q_est = float(self._deep_get(metrics, "q_estimate", self._deep_get(final, "q_estimate", 0.0)) or 0.0)
            wall = float(self._deep_get(metrics, "wall_load_mw_m2", self._deep_get(final, "wall_load_mw_m2", 0.0)) or 0.0)
            risk = float(self._deep_get(final, "disruption_risk", 0.0) or 0.0)
            stable = float(self._deep_get(metrics, "stability_score", 0.0) or 0.0)
            err = target_q - q_est
            integral_err = max(-5.0, min(5.0, integral_err + err * max(1, horizon)))
            derr = err - prev_err
            prev_err = err
            heat_delta = (kp * err) + (ki * integral_err) + (kd * derr)
            heat_delta -= wall_penalty_gain * max(0.0, wall - wall_limit)
            heat_delta -= risk_penalty_gain * max(0.0, risk - 0.45)
            fuel_delta = fuel_gain * err - 0.006 * max(0.0, risk - 0.55)
            b_delta = (b_gain * err) - (0.035 * max(0.0, wall - wall_limit)) + (0.02 * max(0.0, 0.5 - risk))
            next_heating = max(heat_min, min(heat_max, heating + heat_delta))
            next_fuel = max(fuel_min, min(fuel_max, fuel + fuel_delta))
            next_b = max(b_min, min(b_max, bfield + b_delta))
            score = q_est - 0.14 * max(0.0, wall - wall_limit) - 0.18 * risk + 0.08 * stable

            cycle_payload = {
                "cycle": cycle_idx,
                "start_step": simulated_steps,
                "steps": horizon,
                "target_q": round(target_q, 5),
                "inputs": {
                    "heating_mw": round(heating, 4),
                    "fueling_rate": round(fuel, 6),
                    "magnetic_field_t": round(bfield, 4),
                },
                "outputs": {
                    "q_estimate": round(q_est, 5),
                    "wall_load_mw_m2": round(wall, 5),
                    "disruption_risk": round(risk, 5),
                    "stability_score": round(stable, 5),
                },
                "control": {
                    "error_q": round(err, 5),
                    "integral_q": round(integral_err, 5),
                    "derivative_q": round(derr, 5),
                    "delta_heating_mw": round(heat_delta, 5),
                    "delta_fueling_rate": round(fuel_delta, 6),
                    "delta_b_t": round(b_delta, 5),
                },
                "next_inputs": {
                    "heating_mw": round(next_heating, 4),
                    "fueling_rate": round(next_fuel, 6),
                    "magnetic_field_t": round(next_b, 4),
                },
                "score": round(score, 6),
            }
            cycle_results.append(cycle_payload)
            if score > best_score:
                best_score = score
                best_cycle = copy.deepcopy(cycle_payload)
            if isinstance(subrun.get("trace"), list):
                for item in subrun["trace"][-min(120, len(subrun["trace"])) :]:
                    if isinstance(item, dict):
                        control_trace.append({"cycle": cycle_idx, **copy.deepcopy(item)})
            simulated_steps += horizon
            heating, fuel, bfield = next_heating, next_fuel, next_b
            last_q = q_est

        if last_subrun is None or not isinstance(last_final, dict) or not isinstance(last_metrics, dict):
            raise RuntimeErrorNF("fusion_control_sim failed to produce control iterations")
        final = {**copy.deepcopy(last_final), "controller_cycle": max(0, len(cycle_results) - 1)}
        metrics = {
            "q_estimate": round(last_q, 5),
            "wall_load_mw_m2": float(self._deep_get(last_metrics, "wall_load_mw_m2", final.get("wall_load_mw_m2", 0.0)) or 0.0),
            "triple_product": float(self._deep_get(last_metrics, "triple_product", final.get("triple_product", 0.0)) or 0.0),
            "stability_score": float(self._deep_get(last_metrics, "stability_score", 0.0) or 0.0),
            "disruptions": int(self._deep_get(last_metrics, "disruptions", 0) or 0),
            "sustained_ignition_steps": int(self._deep_get(last_metrics, "sustained_ignition_steps", 0) or 0),
            "scenario_steps_total": simulated_steps,
            "controller": {
                "kind": "pid_mpc_lite",
                "cycles": len(cycle_results),
                "horizon_steps": horizon_steps,
                "target_q": target_q,
                "best_cycle": best_cycle,
            },
            "best_q_estimate": round(
                max([float((c.get("outputs") or {}).get("q_estimate", 0.0)) for c in cycle_results] or [0.0]),
                5,
            ),
            "best_score": round(best_score if cycle_results else 0.0, 6),
            "final_inputs": {
                "heating_mw": round(heating, 4),
                "fueling_rate": round(fuel, 6),
                "magnetic_field_t": round(bfield, 4),
            },
        }
        run = {
            "name": run_name,
            "kind": "fusion_reactor_control",
            "steps": total_steps,
            "config": params,
            "controller": {
                "kind": "pid_mpc_lite",
                "target_q": target_q,
                "cycles": len(cycle_results),
                "horizon_steps": horizon_steps,
                "gains": {"kp": kp, "ki": ki, "kd": kd, "fuel_gain": fuel_gain, "b_gain": b_gain},
                "penalties": {"wall": wall_penalty_gain, "risk": risk_penalty_gain},
            },
            "cycles": cycle_results,
            "trace": control_trace[-600:],
            "final": final,
            "metrics": metrics,
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.fusion_runs[run_name] = run
            tick_now = self.tick_count
        self._append_event(
            {
                "tick": tick_now,
                "event": "fusion_control_sim",
                "run": run_name,
                "cycles": len(cycle_results),
                "q": round(last_q, 5),
            }
        )
        return run

    def _fusion_sweep(self, run_name: str, grid: Dict[str, Any], steps: int = 80, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not isinstance(grid, dict) or not grid:
            raise RuntimeErrorNF("fusion_sweep requires a non-empty grid dict")
        settings: Dict[str, Any] = {}
        if isinstance(cfg, dict):
            settings.update(cfg)
        engine = str(settings.pop("engine", settings.get("mode", "multizone"))).lower()
        objective = str(settings.pop("objective", "q_estimate"))
        minimize = bool(settings.pop("minimize", False))
        top_k = max(1, int(settings.pop("top_k", 5)))
        max_cases = max(1, int(settings.pop("max_cases", 256)))

        keys: List[str] = []
        values_lists: List[List[Any]] = []
        normalized_grid: Dict[str, List[Any]] = {}
        for key, val in grid.items():
            k = str(key)
            if isinstance(val, list):
                vals = list(val)
            else:
                vals = [val]
            if not vals:
                vals = [None]
            keys.append(k)
            values_lists.append(vals)
            normalized_grid[k] = copy.deepcopy(vals)

        scenarios: List[Dict[str, Any]] = []
        best_record: Optional[Dict[str, Any]] = None
        case_iter = itertools.product(*values_lists)
        for idx, combo in enumerate(case_iter):
            if idx >= max_cases:
                break
            params = {k: copy.deepcopy(v) for k, v in zip(keys, combo)}
            sim_cfg = {**copy.deepcopy(settings), **params}
            tmp_name = f"__nf_sweep_{run_name}_{idx}"
            if engine in {"control", "pid", "pid_mpc", "pid_mpc_lite"}:
                sim = self._fusion_control_sim(tmp_name, steps=steps, cfg=sim_cfg)
            elif engine in {"multizone", "advanced", "tokamak_multizone"}:
                sim_cfg = {**sim_cfg, "mode": "multizone"}
                sim = self._fusion_sim_multizone(tmp_name, steps=steps, cfg=sim_cfg)
            else:
                sim = self._fusion_sim(tmp_name, steps=steps, cfg=sim_cfg)
            with self._lock:
                self.fusion_runs.pop(tmp_name, None)

            sim_metrics = sim.get("metrics", {}) if isinstance(sim, dict) else {}
            sim_final = sim.get("final", {}) if isinstance(sim, dict) else {}
            score_val = self._deep_get(sim_metrics, objective, None)
            if score_val is None:
                score_val = self._deep_get(sim_final, objective, None)
            try:
                score_num = float(score_val) if score_val is not None else None
            except Exception:
                score_num = None
            q_est = self._deep_get(sim_metrics, "q_estimate", self._deep_get(sim_final, "q_estimate", None))
            wall = self._deep_get(sim_metrics, "wall_load_mw_m2", self._deep_get(sim_final, "wall_load_mw_m2", None))
            risk = self._deep_get(sim_final, "disruption_risk", None)
            record = {
                "case": idx,
                "engine": engine,
                "params": params,
                "score": score_num,
                "objective": objective,
                "metrics": {
                    "q_estimate": q_est,
                    "wall_load_mw_m2": wall,
                    "disruption_risk": risk,
                    "stability_score": self._deep_get(sim_metrics, "stability_score", None),
                    "disruptions": self._deep_get(sim_metrics, "disruptions", None),
                    "triple_product": self._deep_get(sim_metrics, "triple_product", None),
                },
                "kind": sim.get("kind") if isinstance(sim, dict) else "fusion",
                "final": {
                    "q_estimate": self._deep_get(sim_final, "q_estimate", None),
                    "temp_keV": self._deep_get(sim_final, "temp_keV", None),
                    "confinement_s": self._deep_get(sim_final, "confinement_s", None),
                    "radial_stress": self._deep_get(sim_final, "radial_stress", None),
                },
            }
            scenarios.append(record)
            if score_num is not None:
                if best_record is None:
                    best_record = record
                else:
                    prev = best_record.get("score")
                    if isinstance(prev, (int, float)):
                        if (minimize and score_num < float(prev)) or ((not minimize) and score_num > float(prev)):
                            best_record = record

        def _sort_key(item: Dict[str, Any]) -> float:
            v = item.get("score")
            if isinstance(v, (int, float)):
                return float(v)
            return float("inf") if minimize else float("-inf")

        leaderboard = sorted(scenarios, key=_sort_key, reverse=not minimize)[:top_k]
        best_metrics = copy.deepcopy((best_record or {}).get("metrics", {}))
        run = {
            "name": run_name,
            "kind": "fusion_sweep",
            "steps_per_case": max(1, int(steps)),
            "grid": normalized_grid,
            "config": {**settings, "engine": engine, "objective": objective, "minimize": minimize},
            "scenarios": scenarios,
            "leaderboard": leaderboard,
            "best": copy.deepcopy(best_record),
            "metrics": {
                "scenario_count": len(scenarios),
                "objective": objective,
                "engine": engine,
                "best": best_metrics,
                "best_q_estimate": self._deep_get(best_metrics, "q_estimate", None),
                "best_wall_load_mw_m2": self._deep_get(best_metrics, "wall_load_mw_m2", None),
            },
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.fusion_runs[run_name] = run
            tick_now = self.tick_count
        self._append_event(
            {
                "tick": tick_now,
                "event": "fusion_sweep",
                "run": run_name,
                "cases": len(scenarios),
                "engine": engine,
                "objective": objective,
            }
        )
        return run

    def _export_fusion_html(self, run_name: str, rel_path: str) -> Path:
        run = self.fusion_runs.get(run_name)
        if run is None:
            raise RuntimeErrorNF(f"Unknown fusion run: {run_name}")
        trace = run.get("trace") if isinstance(run, dict) else []
        if not isinstance(trace, list):
            trace = []
        run_json = json.dumps(run, ensure_ascii=False)
        out = self._resolve_output_path(rel_path)
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{py_html.escape(self.project.name)} Fusion Viz</title>
<style>
:root{{--bg:#07111f;--card:#101b31;--line:#23324d;--text:#e2e8f0;--muted:#9fb1c9;--accent:#22d3ee;--accent2:#86efac;}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Segoe UI,system-ui,sans-serif;color:var(--text);background:radial-gradient(900px 320px at 0% 0%,rgba(34,211,238,.11),transparent 60%),linear-gradient(180deg,#030712,var(--bg))}}
.wrap{{max-width:1200px;margin:0 auto;padding:18px}}.hero,.card{{border:1px solid var(--line);background:rgba(16,27,49,.76);border-radius:14px;padding:12px}}.grid{{display:grid;grid-template-columns:1.2fr .8fr;gap:12px;margin-top:12px}}canvas{{width:100%;height:360px;border-radius:10px;border:1px solid rgba(255,255,255,.08);background:rgba(2,6,23,.5)}}pre{{margin:0;background:rgba(2,6,23,.55);border:1px solid rgba(255,255,255,.06);border-radius:10px;padding:10px;max-height:360px;overflow:auto;font-size:12px}}.muted{{color:var(--muted)}}.pill{{display:inline-block;border:1px solid rgba(255,255,255,.12);border-radius:999px;padding:3px 8px;font-size:12px;margin-right:6px}}
@media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<section class="hero"><h1 style="margin:0;font-size:22px">Fusion Visualization: {py_html.escape(run_name)}</h1>
<div class="muted"><span class="pill">{py_html.escape(str(run.get("kind","fusion")))}</span>timeseries dashboard export</div></section>
<section class="grid"><div class="card"><canvas id="cv" width="760" height="360"></canvas></div><div class="card"><pre id="meta"></pre></div></section>
<section class="card" style="margin-top:12px"><pre id="json"></pre></section>
</div>
<script>
const run = {run_json};
const trace = Array.isArray(run.trace) ? run.trace : [];
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
document.getElementById('meta').textContent = JSON.stringify({{metrics: run.metrics, final: run.final, config: run.config}}, null, 2);
document.getElementById('json').textContent = JSON.stringify(run, null, 2);
function draw() {{
  const w=cv.width,h=cv.height; ctx.clearRect(0,0,w,h); ctx.fillStyle='#081021'; ctx.fillRect(0,0,w,h);
  const left=46,right=12,top=14,bottom=28, pw=w-left-right, ph=h-top-bottom;
  ctx.strokeStyle='rgba(255,255,255,.08)'; for(let i=0;i<=5;i++){{ const y=top+i*ph/5; ctx.beginPath(); ctx.moveTo(left,y); ctx.lineTo(left+pw,y); ctx.stroke();}}
  if (!trace.length) return;
  const series = [
    {{key:'q_estimate', color:'#22d3ee'}},
    {{key:'temp_keV', color:'#f59e0b'}},
    {{key:'confinement_s', color:'#86efac'}},
    {{key:'disruption_risk', color:'#fb7185'}},
  ];
  const maxX = Math.max(1, trace.length-1);
  const vals = series.flatMap(s => trace.map(t => Number(t[s.key]||0)));
  const minY = Math.min(...vals);
  const maxY = Math.max(...vals, minY + 1e-6);
  const yPad = (maxY-minY)*0.08 || 0.1;
  const lo=minY-yPad, hi=maxY+yPad;
  const sx = (i)=> left + (i/maxX)*pw;
  const sy = (v)=> top + (1-((v-lo)/(hi-lo)))*ph;
  ctx.strokeStyle='rgba(148,163,184,.35)'; ctx.beginPath(); ctx.moveTo(left,top); ctx.lineTo(left,top+ph); ctx.lineTo(left+pw,top+ph); ctx.stroke();
  ctx.fillStyle='rgba(148,163,184,.8)'; ctx.font='12px Segoe UI, sans-serif'; ctx.fillText(hi.toFixed(2), 4, top+10); ctx.fillText(lo.toFixed(2), 4, top+ph);
  for (const s of series) {{
    ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.beginPath();
    trace.forEach((t,i)=>{{ const x=sx(i), y=sy(Number(t[s.key]||0)); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }});
    ctx.stroke();
  }}
  series.forEach((s,idx)=>{{ ctx.fillStyle=s.color; ctx.fillRect(left+idx*160, h-20, 12, 3); ctx.fillStyle='rgba(226,232,240,.9)'; ctx.fillText(s.key, left+idx*160+18, h-14); }});
}}
draw();
</script></body></html>"""
        out.write_text(html, encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "export_fusion_html", "run": run_name, "path": str(out)})
        return out

    def _protein_energy(self, seq: str, coords: List[tuple[int, int]]) -> Dict[str, Any]:
        hydro = set("AILMFWVYC")
        pos = set("KRH")
        neg = set("DE")
        idx_by_pos = {coords[i]: i for i in range(len(coords))}
        energy = 0.0
        h_contacts = 0
        salt = 0
        for i, (x, y) in enumerate(coords):
            ai = seq[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = idx_by_pos.get((x + dx, y + dy))
                if j is None or j <= i or abs(i - j) == 1:
                    continue
                aj = seq[j]
                if ai in hydro and aj in hydro:
                    energy -= 1.0
                    h_contacts += 1
                if (ai in pos and aj in neg) or (ai in neg and aj in pos):
                    energy -= 0.6
                    salt += 1
                if (ai in pos and aj in pos) or (ai in neg and aj in neg):
                    energy += 0.25
            if ai == "P" and 0 < i < len(coords) - 1:
                a0, a1, a2 = coords[i - 1], coords[i], coords[i + 1]
                if (a1[0] - a0[0], a1[1] - a0[1]) != (a2[0] - a1[0], a2[1] - a1[1]):
                    energy += 0.15
        cx = sum(x for x, _ in coords) / len(coords)
        cy = sum(y for _, y in coords) / len(coords)
        rg = math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in coords) / len(coords))
        return {
            "energy": round(energy, 5),
            "hydrophobic_contacts": h_contacts,
            "salt_bridges": salt,
            "radius_gyration": round(rg, 5),
            "compactness": round((h_contacts + salt) / max(1, len(seq)), 5),
        }

    def _protein_pivot(self, coords: List[tuple[int, int]], pivot: int, rot: int) -> Optional[List[tuple[int, int]]]:
        if pivot <= 0 or pivot >= len(coords) - 1:
            return None
        px, py = coords[pivot]
        out = coords[: pivot + 1]
        for x, y in coords[pivot + 1 :]:
            dx, dy = x - px, y - py
            if rot == 1:
                ndx, ndy = -dy, dx
            elif rot == 2:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = dy, -dx
            out.append((px + ndx, py + ndy))
        if len(set(out)) != len(out):
            return None
        for i in range(1, len(out)):
            ax, ay = out[i - 1]
            bx, by = out[i]
            if abs(ax - bx) + abs(ay - by) != 1:
                return None
        return out

    def _protein_fold_sim(self, run_name: str, sequence: str, steps: int = 200, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(cfg, dict):
            mode = str(cfg.get("mode", "")).lower()
            if mode in {"3d", "lattice3d", "advanced"}:
                return self._protein_fold_sim_3d(run_name, sequence, steps=steps, cfg=cfg)
        seq = re.sub(r"[^A-Za-z]", "", sequence or "").upper()
        if len(seq) < 3:
            raise RuntimeErrorNF("protein_fold_sim requires sequence length >= 3")
        rng = self._get_rng()
        steps = max(1, int(steps))
        params: Dict[str, Any] = {"temperature": 1.0, "sample_every": max(1, steps // 20)}
        if isinstance(cfg, dict):
            params.update(cfg)
        temp = max(0.01, float(params.get("temperature", 1.0)))
        sample_every = max(1, int(params.get("sample_every", max(1, steps // 20))))
        coords: List[tuple[int, int]] = [(i, 0) for i in range(len(seq))]
        cur_stats = self._protein_energy(seq, coords)
        cur_e = float(cur_stats["energy"])
        best_coords = list(coords)
        best_stats = dict(cur_stats)
        best_e = cur_e
        accepted = 0
        traj: List[Dict[str, Any]] = []
        frames: List[Dict[str, Any]] = []
        for step in range(steps):
            cand = self._protein_pivot(coords, rng.randint(1, len(seq) - 2), rng.choice([1, 2, 3]))
            if cand is not None:
                cand_stats = self._protein_energy(seq, cand)
                cand_e = float(cand_stats["energy"])
                dE = cand_e - cur_e
                if dE <= 0 or rng.random() < math.exp(-dE / max(temp, 1e-9)):
                    coords = cand
                    cur_stats = cand_stats
                    cur_e = cand_e
                    accepted += 1
                    if cand_e < best_e:
                        best_e = cand_e
                        best_coords = list(cand)
                        best_stats = dict(cand_stats)
            if step % sample_every == 0 or step == steps - 1:
                traj.append({
                    "step": step,
                    "energy": round(cur_e, 5),
                    "best_energy": round(best_e, 5),
                    "radius_gyration": cur_stats["radius_gyration"],
                    "hydrophobic_contacts": cur_stats["hydrophobic_contacts"],
                })
                frames.append({"step": step, "coords2d": [[x, y] for x, y in coords]})
        final_stats = self._protein_energy(seq, coords)
        best_stats = self._protein_energy(seq, best_coords)
        run = {
            "name": run_name,
            "kind": "protein_folding",
            "sequence": seq,
            "length": len(seq),
            "steps": steps,
            "config": params,
            "acceptance_ratio": round(accepted / max(1, steps), 5),
            "final": {**final_stats, "coords2d": [[x, y] for x, y in coords]},
            "best": {**best_stats, "coords2d": [[x, y] for x, y in best_coords]},
            "trajectory": traj[-500:],
            "frames": frames[-200:],
            "residue_types": {
                "hydrophobic": sum(1 for aa in seq if aa in set("AILMFWVYC")),
                "polar": sum(1 for aa in seq if aa in set("STNQG")),
                "positive": sum(1 for aa in seq if aa in set("KRH")),
                "negative": sum(1 for aa in seq if aa in set("DE")),
            },
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.protein_runs[run_name] = run
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "protein_fold_sim", "run": run_name, "steps": steps, "best_energy": run["best"]["energy"]})
        return run

    def _export_protein_json(self, run_name: str, rel_path: str) -> Path:
        run = self.protein_runs.get(run_name)
        if run is None:
            raise RuntimeErrorNF(f"Unknown protein fold run: {run_name}")
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps({"project": self.project.name, "protein": run}, indent=2), encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "export_protein_json", "run": run_name, "path": str(out)})
        return out

    def _protein_energy_3d(self, seq: str, coords: List[tuple[int, int, int]]) -> Dict[str, Any]:
        hydro = set("AILMFWVYC")
        pos = set("KRH")
        neg = set("DE")
        idx_by_pos = {coords[i]: i for i in range(len(coords))}
        energy = 0.0
        h_contacts = 0
        salt = 0
        for i, (x, y, z) in enumerate(coords):
            ai = seq[i]
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                j = idx_by_pos.get((x + dx, y + dy, z + dz))
                if j is None or j <= i or abs(i - j) == 1:
                    continue
                aj = seq[j]
                if ai in hydro and aj in hydro:
                    energy -= 1.15
                    h_contacts += 1
                if (ai in pos and aj in neg) or (ai in neg and aj in pos):
                    energy -= 0.65
                    salt += 1
                if (ai in pos and aj in pos) or (ai in neg and aj in neg):
                    energy += 0.25
        cx = sum(x for x, _, _ in coords) / len(coords)
        cy = sum(y for _, y, _ in coords) / len(coords)
        cz = sum(z for _, _, z in coords) / len(coords)
        rg = math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 for x, y, z in coords) / len(coords))
        return {
            "energy": round(energy, 5),
            "hydrophobic_contacts": h_contacts,
            "salt_bridges": salt,
            "radius_gyration": round(rg, 5),
            "compactness": round((h_contacts + salt) / max(1, len(seq)), 5),
        }

    def _protein_rotate_3d(self, dx: int, dy: int, dz: int, axis: str, turns: int) -> tuple[int, int, int]:
        turns = turns % 4
        x, y, z = dx, dy, dz
        for _ in range(turns):
            if axis == "x":
                y, z = -z, y
            elif axis == "y":
                x, z = z, -x
            else:
                x, y = -y, x
        return x, y, z

    def _protein_pivot_3d(self, coords: List[tuple[int, int, int]], pivot: int, axis: str, turns: int) -> Optional[List[tuple[int, int, int]]]:
        if pivot <= 0 or pivot >= len(coords) - 1:
            return None
        px, py, pz = coords[pivot]
        out = coords[: pivot + 1]
        for x, y, z in coords[pivot + 1 :]:
            dx, dy, dz = x - px, y - py, z - pz
            ndx, ndy, ndz = self._protein_rotate_3d(dx, dy, dz, axis, turns)
            out.append((px + ndx, py + ndy, pz + ndz))
        if len(set(out)) != len(out):
            return None
        for i in range(1, len(out)):
            ax, ay, az = out[i - 1]
            bx, by, bz = out[i]
            if abs(ax - bx) + abs(ay - by) + abs(az - bz) != 1:
                return None
        return out

    def _protein_fold_sim_3d(self, run_name: str, sequence: str, steps: int = 250, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        seq = re.sub(r"[^A-Za-z]", "", sequence or "").upper()
        if len(seq) < 3:
            raise RuntimeErrorNF("protein_fold_sim_3d requires sequence length >= 3")
        rng = self._get_rng()
        steps = max(1, int(steps))
        params: Dict[str, Any] = {"mode": "lattice3d", "temperature": 1.0, "sample_every": max(1, steps // 20)}
        if isinstance(cfg, dict):
            params.update(cfg)
        temp = max(0.01, float(params.get("temperature", 1.0)))
        sample_every = max(1, int(params.get("sample_every", max(1, steps // 20))))
        coords: List[tuple[int, int, int]] = [(i, 0, 0) for i in range(len(seq))]
        cur_stats = self._protein_energy_3d(seq, coords)
        cur_e = float(cur_stats["energy"])
        best_coords = list(coords)
        best_e = cur_e
        accepted = 0
        traj: List[Dict[str, Any]] = []
        frames: List[Dict[str, Any]] = []
        axes = ["x", "y", "z"]
        for step in range(steps):
            cand = self._protein_pivot_3d(coords, rng.randint(1, len(seq) - 2), rng.choice(axes), rng.choice([1, 2, 3]))
            if cand is not None:
                cand_stats = self._protein_energy_3d(seq, cand)
                cand_e = float(cand_stats["energy"])
                dE = cand_e - cur_e
                if dE <= 0 or rng.random() < math.exp(-dE / max(temp, 1e-9)):
                    coords = cand
                    cur_stats = cand_stats
                    cur_e = cand_e
                    accepted += 1
                    if cand_e < best_e:
                        best_e = cand_e
                        best_coords = list(cand)
            if step % sample_every == 0 or step == steps - 1:
                traj.append({
                    "step": step,
                    "energy": round(cur_e, 5),
                    "best_energy": round(best_e, 5),
                    "radius_gyration": cur_stats["radius_gyration"],
                    "hydrophobic_contacts": cur_stats["hydrophobic_contacts"],
                })
                frames.append({"step": step, "coords3d": [[x, y, z] for x, y, z in coords]})
        final_stats = self._protein_energy_3d(seq, coords)
        best_stats = self._protein_energy_3d(seq, best_coords)
        run = {
            "name": run_name,
            "kind": "protein_folding_3d",
            "sequence": seq,
            "length": len(seq),
            "steps": steps,
            "config": params,
            "acceptance_ratio": round(accepted / max(1, steps), 5),
            "final": {**final_stats, "coords3d": [[x, y, z] for x, y, z in coords]},
            "best": {**best_stats, "coords3d": [[x, y, z] for x, y, z in best_coords]},
            "trajectory": traj[-500:],
            "frames": frames[-200:],
            "residue_types": {
                "hydrophobic": sum(1 for aa in seq if aa in set("AILMFWVYC")),
                "polar": sum(1 for aa in seq if aa in set("STNQG")),
                "positive": sum(1 for aa in seq if aa in set("KRH")),
                "negative": sum(1 for aa in seq if aa in set("DE")),
            },
            "created_at": self._now_iso(),
        }
        with self._lock:
            self.protein_runs[run_name] = run
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "protein_fold_sim_3d", "run": run_name, "steps": steps, "best_energy": run["best"]["energy"]})
        return run

    def _export_protein_html(self, run_name: str, rel_path: str) -> Path:
        run = self.protein_runs.get(run_name)
        if run is None:
            raise RuntimeErrorNF(f"Unknown protein fold run: {run_name}")
        run_json = json.dumps(run, ensure_ascii=False)
        out = self._resolve_output_path(rel_path)
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{py_html.escape(self.project.name)} Protein Viz</title>
<style>
:root{{--bg:#07111f;--card:#101b31;--line:#23324d;--text:#e2e8f0;--muted:#9fb1c9;--accent:#22d3ee;--accent2:#86efac;}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Segoe UI,system-ui,sans-serif;color:var(--text);background:radial-gradient(900px 320px at 0% 0%,rgba(34,211,238,.11),transparent 60%),linear-gradient(180deg,#030712,var(--bg))}}
.wrap{{max-width:1300px;margin:0 auto;padding:18px}}.hero,.card{{border:1px solid var(--line);background:rgba(16,27,49,.76);border-radius:14px;padding:12px}}
.grid{{display:grid;grid-template-columns:1.25fr .75fr;gap:12px;margin-top:12px}}.subgrid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}
.toolbar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}}button{{background:#0f1b31;color:var(--text);border:1px solid rgba(255,255,255,.12);border-radius:10px;padding:6px 10px;cursor:pointer}}button:hover{{border-color:rgba(34,211,238,.35)}}
label{{font-size:12px;color:var(--muted)}}input[type=range]{{width:100%}}
canvas{{width:100%;height:420px;border-radius:10px;border:1px solid rgba(255,255,255,.08);background:rgba(2,6,23,.5)}}.small canvas{{height:260px}}
pre{{margin:0;background:rgba(2,6,23,.55);border:1px solid rgba(255,255,255,.06);border-radius:10px;padding:10px;max-height:420px;overflow:auto;font-size:12px}}.muted{{color:var(--muted)}}.pill{{display:inline-block;border:1px solid rgba(255,255,255,.12);border-radius:999px;padding:3px 8px;font-size:12px;margin-right:6px}}
@media (max-width:980px){{.grid{{grid-template-columns:1fr}}.subgrid{{grid-template-columns:1fr}}canvas{{height:340px}}.small canvas{{height:240px}}}}
</style></head><body><div class="wrap">
<section class="hero"><h1 style="margin:0;font-size:22px">Protein Folding Visualization: {py_html.escape(run_name)}</h1><div class="muted"><span class="pill">trajectory playback</span><span class="pill">Contact Map</span><span class="pill">Ramachandran-like</span></div></section>
<section class="grid"><div class="card"><canvas id="cv" width="760" height="420"></canvas><div class="toolbar"><button id="prev">Prev</button><button id="play">Play</button><button id="pause">Pause</button><button id="next">Next</button><label>Speed <input id="speed" type="range" min="1" max="20" value="6"></label></div><div style="margin-top:8px"><input id="slider" type="range" min="0" max="0" value="0"><div id="status" class="muted"></div></div></div><div class="card"><pre id="meta"></pre></div></section>
<section class="subgrid"><div class="card small"><h3 style="margin:0 0 8px 0;font-size:15px">Contact Map</h3><canvas id="contactMap" width="520" height="260"></canvas><div id="contactMeta" class="muted" style="margin-top:6px"></div></div><div class="card small"><h3 style="margin:0 0 8px 0;font-size:15px">Ramachandran-like Approximation</h3><canvas id="rama" width="520" height="260"></canvas><div id="ramaMeta" class="muted" style="margin-top:6px"></div></div></section>
<section class="card" style="margin-top:12px"><pre id="json"></pre></section>
</div>
<script>
const run = {run_json};
const frames = Array.isArray(run.frames) ? run.frames : [];
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const slider = document.getElementById('slider');
const speed = document.getElementById('speed');
const contactCv = document.getElementById('contactMap'), contactCtx = contactCv.getContext('2d');
const ramaCv = document.getElementById('rama'), ramaCtx = ramaCv.getContext('2d');
let playTimer = null;
let playing = false;
slider.max = String(Math.max(0, frames.length - 1));
document.getElementById('meta').textContent = JSON.stringify({{kind: run.kind, length: run.length, metrics_best: run.best, metrics_final: run.final, config: run.config, frames: frames.length, trajectory: Array.isArray(run.trajectory)?run.trajectory.length:0}}, null, 2);
document.getElementById('json').textContent = JSON.stringify(run, null, 2);
function project(p) {{
  if (!Array.isArray(p)) return [0, 0];
  if (p.length === 2) return [Number(p[0]||0), Number(p[1]||0)];
  const x=Number(p[0]||0), y=Number(p[1]||0), z=Number(p[2]||0);
  return [x + z * 0.55, y - z * 0.35];
}}
function currentFrameIndex() {{
  return Math.max(0, Math.min(frames.length - 1, Number(slider.value)||0));
}}
function getFrame() {{
  return frames.length ? (frames[currentFrameIndex()] || null) : null;
}}
function getCoords() {{
  const f = getFrame();
  if (f) return f.coords3d || f.coords2d || [];
  return (run.final && (run.final.coords3d || run.final.coords2d)) || [];
}}
function latticeDistance(a, b) {{
  let s = 0;
  const n = Math.max(Array.isArray(a)?a.length:0, Array.isArray(b)?b.length:0);
  for (let i=0;i<n;i++) s += Math.abs(Number((a||[])[i]||0) - Number((b||[])[i]||0));
  return s;
}}
function bondPairs(coords) {{
  const pts = coords.map(p => [Number(p[0]||0), Number(p[1]||0), Number(p[2]||0)]);
  const out = [];
  for (let i=1;i<pts.length-1;i++) {{
    const p0=pts[i-1], p1=pts[i], p2=pts[i+1];
    const v1=[p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
    const v2=[p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];
    const phi = Math.atan2(v1[1], v1[0]) * 180 / Math.PI;
    const psi = Math.atan2(v2[2], Math.hypot(v2[0], v2[1]) || 1) * 180 / Math.PI * 2;
    const dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    const den = (Math.hypot(v1[0], v1[1], v1[2]) || 1) * (Math.hypot(v2[0], v2[1], v2[2]) || 1);
    const bend = Math.acos(Math.max(-1, Math.min(1, dot / den))) * 180 / Math.PI;
    out.push([phi, psi, bend]);
  }}
  return out;
}}
function drawContactMap(coords) {{
  const w=contactCv.width,h=contactCv.height;
  contactCtx.clearRect(0,0,w,h); contactCtx.fillStyle='#081021'; contactCtx.fillRect(0,0,w,h);
  const pad = 20; const n = Math.max(1, coords.length); const sz = Math.min(w - pad*2, h - pad*2); const cell = Math.max(2, Math.floor(sz / Math.max(8, n)));
  contactCtx.strokeStyle='rgba(148,163,184,.25)'; contactCtx.strokeRect(pad,pad,sz,sz);
  let contacts = 0;
  for (let i=0;i<coords.length;i++) {{
    for (let j=0;j<coords.length;j++) {{
      const x = pad + (i / n) * sz;
      const y = pad + (j / n) * sz;
      if (i === j) {{ contactCtx.fillStyle='rgba(148,163,184,.35)'; contactCtx.fillRect(x,y,cell,cell); continue; }}
      const backbone = Math.abs(i-j) <= 1;
      const dist = latticeDistance(coords[i], coords[j]);
      if (!backbone && dist === 1) {{ if (i < j) contacts++; contactCtx.fillStyle='rgba(134,239,172,.82)'; contactCtx.fillRect(x,y,cell,cell); }}
      else if (!backbone && dist === 2) {{ contactCtx.fillStyle='rgba(34,211,238,.18)'; contactCtx.fillRect(x,y,cell,cell); }}
    }}
  }}
  document.getElementById('contactMeta').textContent = 'contacts=' + contacts + ' | lattice adjacency';
}}
function drawRama(coords) {{
  const w=ramaCv.width,h=ramaCv.height, pad=28, pw=w-pad*2, ph=h-pad*2;
  ramaCtx.clearRect(0,0,w,h); ramaCtx.fillStyle='#081021'; ramaCtx.fillRect(0,0,w,h);
  ramaCtx.strokeStyle='rgba(148,163,184,.2)';
  for (let i=0;i<=4;i++) {{ const x=pad+i*pw/4, y=pad+i*ph/4; ramaCtx.beginPath(); ramaCtx.moveTo(x,pad); ramaCtx.lineTo(x,pad+ph); ramaCtx.stroke(); ramaCtx.beginPath(); ramaCtx.moveTo(pad,y); ramaCtx.lineTo(pad+pw,y); ramaCtx.stroke(); }}
  ramaCtx.strokeStyle='rgba(148,163,184,.42)'; ramaCtx.strokeRect(pad,pad,pw,ph);
  const pairs = bondPairs(coords);
  const sx = (v)=> pad + ((v + 180) / 360) * pw;
  const sy = (v)=> pad + (1 - ((v + 180) / 360)) * ph;
  pairs.forEach((p, idx) => {{
    const alpha = Math.min(1, 0.3 + 0.6 * ((Number(p[2]||0)) / 180));
    ramaCtx.fillStyle = 'rgba(245,158,11,' + alpha + ')';
    ramaCtx.beginPath(); ramaCtx.arc(sx(Number(p[0]||0)), sy(Number(p[1]||0)), 3.2, 0, Math.PI*2); ramaCtx.fill();
    if (idx === 0 || idx === pairs.length-1) {{ ramaCtx.strokeStyle='white'; ramaCtx.stroke(); }}
  }});
  ramaCtx.fillStyle='rgba(148,163,184,.75)'; ramaCtx.font='12px Segoe UI, sans-serif'; ramaCtx.fillText('phi approx', pad, 14); ramaCtx.fillText('psi approx', w-62, h-8);
  const avgBend = pairs.length ? (pairs.reduce((a,b)=>a+Number(b[2]||0),0)/pairs.length) : 0;
  document.getElementById('ramaMeta').textContent = 'points=' + pairs.length + ' | avg bend=' + avgBend.toFixed(1) + '° (Ramachandran-like lattice proxy)';
}}
function draw() {{
  const coords = getCoords();
  const w=cv.width,h=cv.height; ctx.clearRect(0,0,w,h); ctx.fillStyle='#081021'; ctx.fillRect(0,0,w,h);
  if (!coords.length) {{ drawContactMap([]); drawRama([]); return; }}
  const pts = coords.map(project);
  const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
  const left=40,right=20,top=20,bottom=20,pw=w-left-right,ph=h-top-bottom;
  const sx=(x)=> left + ((x-minX)/Math.max(1e-6,(maxX-minX)||1))*pw;
  const sy=(y)=> top + (1-((y-minY)/Math.max(1e-6,(maxY-minY)||1)))*ph;
  ctx.strokeStyle='rgba(148,163,184,.22)'; for(let i=0;i<6;i++){{let gy=top+i*ph/5;ctx.beginPath();ctx.moveTo(left,gy);ctx.lineTo(left+pw,gy);ctx.stroke();}}
  ctx.lineWidth=2;
  for (let i=1;i<pts.length;i++) {{
    const ax=pts[i-1][0], ay=pts[i-1][1], bx=pts[i][0], by=pts[i][1];
    ctx.strokeStyle='rgba(34,211,238,.65)'; ctx.beginPath(); ctx.moveTo(sx(ax), sy(ay)); ctx.lineTo(sx(bx), sy(by)); ctx.stroke();
  }}
  for (let i=0;i<pts.length;i++) {{
    const x=pts[i][0], y=pts[i][1];
    const aa = (run.sequence && run.sequence[i]) ? run.sequence[i] : '';
    const color = 'AILMFWVYC'.includes(aa) ? '#86efac' : ('KRH'.includes(aa) ? '#60a5fa' : ('DE'.includes(aa) ? '#fb7185' : '#f59e0b'));
    ctx.fillStyle=color; ctx.beginPath(); ctx.arc(sx(x), sy(y), 4, 0, Math.PI*2); ctx.fill();
    if (i===0 || i===pts.length-1) {{ ctx.strokeStyle='white'; ctx.stroke(); }}
  }}
  const idx = currentFrameIndex();
  const frame = frames[idx] || null;
  let trajMsg = '';
  if (Array.isArray(run.trajectory) && frame) {{
    const t = run.trajectory.find(it => it && it.step === frame.step) || null;
    if (t) trajMsg = ' | energy=' + t.energy + ' | rg=' + t.radius_gyration;
  }}
  document.getElementById('status').textContent = frame ? ('frame ' + (idx+1) + '/' + frames.length + ' step=' + frame.step + trajMsg) : 'final structure';
  drawContactMap(coords);
  drawRama(coords);
}}
function stopPlayback() {{
  playing = false;
  if (playTimer) {{ clearInterval(playTimer); playTimer = null; }}
}}
function startPlayback() {{
  stopPlayback();
  if (!frames.length) return;
  playing = true;
  const fps = Math.max(1, Number(speed.value)||6);
  playTimer = setInterval(() => {{
    const max = Math.max(0, frames.length - 1);
    const next = currentFrameIndex() >= max ? 0 : currentFrameIndex() + 1;
    slider.value = String(next);
    draw();
  }}, Math.round(1000 / fps));
}}
document.getElementById('prev').onclick = () => {{ stopPlayback(); slider.value = String(Math.max(0, currentFrameIndex() - 1)); draw(); }};
document.getElementById('next').onclick = () => {{ stopPlayback(); slider.value = String(Math.min(Math.max(0, frames.length - 1), currentFrameIndex() + 1)); draw(); }};
document.getElementById('play').onclick = startPlayback;
document.getElementById('pause').onclick = stopPlayback;
speed.oninput = () => {{ if (playing) startPlayback(); }};
slider.oninput = () => {{ if (playing) stopPlayback(); draw(); }};
draw();
</script></body></html>"""
        out.write_text(html, encoding="utf-8")
        with self._lock:
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": "export_protein_html", "run": run_name, "path": str(out)})
        return out

    def eval_expr(self, expr: Expr, local: Optional[Dict[str, Any]]) -> Any:
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, ListLiteral):
            return [self.eval_expr(i, local) for i in expr.items]
        if isinstance(expr, DictLiteral):
            return {k: self.eval_expr(v, local) for k, v in expr.items}
        if isinstance(expr, Var):
            if local is not None and expr.name in local:
                return local[expr.name]
            if expr.name in self.metric_exprs:
                return self._compute_metric(expr.name, local)
            with self._lock:
                if expr.name in self.state:
                    return self.state[expr.name]
                if expr.name in self.config:
                    return self.config[expr.name]
            if expr.name == "tick":
                with self._lock:
                    return self.tick_count
            raise RuntimeErrorNF(f"Unknown variable: {expr.name}")
        if isinstance(expr, Unary):
            v = self.eval_expr(expr.expr, local)
            if expr.op == "-":
                return -_num(v)
            if expr.op == "!":
                return not bool(v)
            raise RuntimeErrorNF(f"Unsupported unary operator {expr.op}")
        if isinstance(expr, Binary):
            a = self.eval_expr(expr.left, local)
            if expr.op == "&&":
                return bool(a) and bool(self.eval_expr(expr.right, local))
            if expr.op == "||":
                return bool(a) or bool(self.eval_expr(expr.right, local))
            b = self.eval_expr(expr.right, local)
            if expr.op == "+":
                return a + b
            if expr.op == "-":
                return a - b
            if expr.op == "*":
                return a * b
            if expr.op == "/":
                return a / b
            if expr.op == "%":
                return a % b
            if expr.op == "==":
                return a == b
            if expr.op == "!=":
                return a != b
            if expr.op == "<":
                return a < b
            if expr.op == ">":
                return a > b
            if expr.op == "<=":
                return a <= b
            if expr.op == ">=":
                return a >= b
            raise RuntimeErrorNF(f"Unsupported binary operator {expr.op}")
        if isinstance(expr, Call):
            return self.call_builtin(expr.name, [self.eval_expr(a, local) for a in expr.args], local)
        raise RuntimeErrorNF(f"Unsupported expression {expr!r}")

    def call_builtin(self, name: str, args: List[Any], local: Optional[Dict[str, Any]]) -> Any:
        if name == "now_iso":
            if args:
                raise RuntimeErrorNF("now_iso() expects 0 args")
            return self._now_iso()
        if name == "platform":
            if args:
                raise RuntimeErrorNF("platform() expects 0 args")
            return self._host_info()["platform"]
        if name == "hostname":
            if args:
                raise RuntimeErrorNF("hostname() expects 0 args")
            return socket.gethostname()
        if name == "cwd":
            if args:
                raise RuntimeErrorNF("cwd() expects 0 args")
            return str(self.base_dir)
        if name == "env":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("env(name[, default]) expects 1 or 2 args")
            key = str(args[0])
            default = args[1] if len(args) == 2 else None
            return os.environ.get(key, default)
        if name == "path_exists":
            if len(args) != 1:
                raise RuntimeErrorNF("path_exists(path) expects 1 arg")
            return self._resolve_runtime_path(str(args[0])).exists()
        if name == "read_text":
            if len(args) != 1:
                raise RuntimeErrorNF("read_text(path) expects 1 arg")
            p = self._resolve_runtime_path(str(args[0]))
            return p.read_text(encoding="utf-8") if p.exists() else None
        if name == "file_size":
            if len(args) != 1:
                raise RuntimeErrorNF("file_size(path) expects 1 arg")
            p = self._resolve_runtime_path(str(args[0]))
            return p.stat().st_size if p.exists() else 0
        if name == "slugify":
            if len(args) != 1:
                raise RuntimeErrorNF("slugify(text) expects 1 arg")
            return self._web_slugify(args[0])
        if name == "html_escape":
            if len(args) != 1:
                raise RuntimeErrorNF("html_escape(text) expects 1 arg")
            return py_html.escape(str(args[0]))
        if name == "url_encode":
            if len(args) != 1:
                raise RuntimeErrorNF("url_encode(text) expects 1 arg")
            return urllib.parse.quote(str(args[0]), safe="")
        if name == "url_decode":
            if len(args) != 1:
                raise RuntimeErrorNF("url_decode(text) expects 1 arg")
            return urllib.parse.unquote(str(args[0]))
        if name == "query_parse":
            if len(args) != 1:
                raise RuntimeErrorNF("query_parse(text) expects 1 arg")
            parsed = urllib.parse.parse_qs(str(args[0]), keep_blank_values=True)
            out: Dict[str, Any] = {}
            for k, v in parsed.items():
                out[k] = v[0] if len(v) == 1 else v
            return out
        if name == "query_stringify":
            if len(args) != 1 or not isinstance(args[0], dict):
                raise RuntimeErrorNF("query_stringify(map) expects 1 dict arg")
            pairs: List[tuple[str, str]] = []
            for k, v in args[0].items():
                if isinstance(v, list):
                    for item in v:
                        pairs.append((str(k), str(item)))
                elif v is None:
                    pairs.append((str(k), ""))
                else:
                    pairs.append((str(k), str(v)))
            return urllib.parse.urlencode(pairs)
        if name == "web_tool_count":
            if args:
                raise RuntimeErrorNF("web_tool_count() expects 0 args")
            with self._lock:
                return len(self.web_tools)
        if name == "http_history_count":
            if args:
                raise RuntimeErrorNF("http_history_count() expects 0 args")
            with self._lock:
                return len(self.http_ops)
        if name == "http_last":
            if args:
                raise RuntimeErrorNF("http_last() expects 0 args")
            with self._lock:
                return copy.deepcopy(self.http_ops[-1]) if self.http_ops else None
        if name == "http_history":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("http_history([limit]) expects 0 or 1 args")
            with self._lock:
                ops = copy.deepcopy(self.http_ops)
            if args:
                return ops[-max(0, _as_int(args[0])) :]
            return ops
        if name == "http_auth_preset_count":
            if args:
                raise RuntimeErrorNF("http_auth_preset_count() expects 0 args")
            with self._lock:
                return len(self.http_auth_presets)
        if name == "http_auth_preset_info":
            if len(args) != 1:
                raise RuntimeErrorNF("http_auth_preset_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy(self.http_auth_presets.get(str(args[0])))
        if name == "mock_http_server_count":
            if args:
                raise RuntimeErrorNF("mock_http_server_count() expects 0 args")
            with self._lock:
                return len(self.mock_http_servers)
        if name == "mock_http_server_info":
            if len(args) != 1:
                raise RuntimeErrorNF("mock_http_server_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy(self.mock_http_servers.get(str(args[0])))
        if name == "mock_http_server_url":
            if len(args) != 1:
                raise RuntimeErrorNF("mock_http_server_url(name) expects 1 arg")
            with self._lock:
                meta = copy.deepcopy(self.mock_http_servers.get(str(args[0])))
            if isinstance(meta, dict):
                return meta.get("base_url")
            return None
        if name == "lang_module_count":
            if args:
                raise RuntimeErrorNF("lang_module_count() expects 0 args")
            with self._lock:
                return len(self.lang_modules)
        if name == "lang_module_info":
            if len(args) != 1:
                raise RuntimeErrorNF("lang_module_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy(self.lang_modules.get(str(args[0])))
        if name == "lang_module_last":
            if len(args) != 1:
                raise RuntimeErrorNF("lang_module_last(name) expects 1 arg")
            with self._lock:
                meta = copy.deepcopy(self.lang_modules.get(str(args[0])))
            if isinstance(meta, dict):
                return meta.get("last_run")
            return None
        if name == "cpp_compiler_info":
            if args:
                raise RuntimeErrorNF("cpp_compiler_info() expects 0 args")
            return self._cpp_compiler_info()
        if name == "csharp_compiler_info":
            if args:
                raise RuntimeErrorNF("csharp_compiler_info() expects 0 args")
            return self._csharp_compiler_info()
        if name == "web_tool_info":
            if len(args) != 1:
                raise RuntimeErrorNF("web_tool_info(path) expects 1 arg")
            key = str(args[0])
            with self._lock:
                if key in self.web_tools:
                    return copy.deepcopy(self.web_tools[key])
                for path, meta in self.web_tools.items():
                    if Path(path).name == key:
                        return copy.deepcopy(meta)
            return None
        if name == "sqlite_db_count":
            if args:
                raise RuntimeErrorNF("sqlite_db_count() expects 0 args")
            with self._lock:
                return len((self.sqlite_state or {}).get("databases", {}))
        if name == "sqlite_history_count":
            if args:
                raise RuntimeErrorNF("sqlite_history_count() expects 0 args")
            with self._lock:
                return len((self.sqlite_state or {}).get("ops", []))
        if name == "sqlite_last":
            if args:
                raise RuntimeErrorNF("sqlite_last() expects 0 args")
            with self._lock:
                ops = (self.sqlite_state or {}).get("ops", [])
                return copy.deepcopy(ops[-1]) if ops else None
        if name == "sqlite_db_info":
            if len(args) != 1:
                raise RuntimeErrorNF("sqlite_db_info(path_or_name) expects 1 arg")
            key = str(args[0])
            with self._lock:
                dbs = copy.deepcopy((self.sqlite_state or {}).get("databases", {}))
            if key in dbs:
                return dbs[key]
            for path, meta in dbs.items():
                if Path(str(path)).name == key:
                    return meta
            return None
        if name == "sqlite_query":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("sqlite_query(db, sql[, params]) expects 2 or 3 args")
            params = args[2] if len(args) == 3 else None
            return self._sqlite_query(str(args[0]), str(args[1]), params=params)
        if name == "sqlite_scalar":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("sqlite_scalar(db, sql[, params]) expects 2 or 3 args")
            params = args[2] if len(args) == 3 else None
            return self._sqlite_scalar(str(args[0]), str(args[1]), params=params)
        if name == "archive_count":
            if args:
                raise RuntimeErrorNF("archive_count() expects 0 args")
            with self._lock:
                return len((self.archive_state or {}).get("archives", {}))
        if name == "archive_op_count":
            if args:
                raise RuntimeErrorNF("archive_op_count() expects 0 args")
            with self._lock:
                return len((self.archive_state or {}).get("ops", []))
        if name == "archive_last":
            if args:
                raise RuntimeErrorNF("archive_last() expects 0 args")
            with self._lock:
                ops = (self.archive_state or {}).get("ops", [])
                return copy.deepcopy(ops[-1]) if ops else None
        if name == "archive_info":
            if len(args) != 1:
                raise RuntimeErrorNF("archive_info(path_or_name) expects 1 arg")
            key = str(args[0])
            with self._lock:
                archives = copy.deepcopy((self.archive_state or {}).get("archives", {}))
            if key in archives:
                return archives[key]
            for path, meta in archives.items():
                if Path(str(path)).name == key:
                    return meta
            return None
        if name == "iso_count":
            if args:
                raise RuntimeErrorNF("iso_count() expects 0 args")
            with self._lock:
                return len((self.iso_state or {}).get("images", {}))
        if name == "iso_op_count":
            if args:
                raise RuntimeErrorNF("iso_op_count() expects 0 args")
            with self._lock:
                return len((self.iso_state or {}).get("ops", []))
        if name == "iso_last":
            if args:
                raise RuntimeErrorNF("iso_last() expects 0 args")
            with self._lock:
                last = (self.iso_state or {}).get("last")
                if last is not None:
                    return copy.deepcopy(last)
                ops = (self.iso_state or {}).get("ops", [])
                return copy.deepcopy(ops[-1]) if ops else None
        if name == "iso_info":
            if len(args) != 1:
                raise RuntimeErrorNF("iso_info(path_or_name) expects 1 arg")
            key = str(args[0])
            with self._lock:
                images = copy.deepcopy((self.iso_state or {}).get("images", {}))
            if key in images:
                return images[key]
            for path, meta in images.items():
                if Path(str(path)).name == key:
                    return meta
            return None
        if name == "iso_tool_info":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("iso_tool_info([refresh]) expects 0 or 1 args")
            refresh = bool(args[0]) if len(args) == 1 else False
            return self._iso_tool_info(refresh=refresh)
        if name == "file_hash":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("file_hash(path[, algo]) expects 1 or 2 args")
            algo = str(args[1]) if len(args) == 2 and args[1] is not None else None
            return self._file_hash(str(args[0]), algo=algo).get("digest")
        if name == "file_hash_info":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("file_hash_info(path[, algo]) expects 1 or 2 args")
            algo = str(args[1]) if len(args) == 2 and args[1] is not None else None
            return self._file_hash(str(args[0]), algo=algo)
        if name == "file_hash_verify":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("file_hash_verify(path, expected[, algo]) expects 2 or 3 args")
            algo = str(args[2]) if len(args) == 3 and args[2] is not None else None
            return bool(self._file_hash_verify(str(args[0]), str(args[1]), algo=algo).get("match"))
        if name == "dir_manifest":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("dir_manifest(path[, recursive_or_cfg[, cfg]]) expects 1-3 args")
            recursive = True
            cfg = None
            if len(args) >= 2:
                if isinstance(args[1], dict):
                    cfg = args[1]
                else:
                    recursive = bool(args[1])
            if len(args) == 3:
                if not isinstance(args[2], dict):
                    raise RuntimeErrorNF("dir_manifest(..., cfg) third arg must be dict")
                cfg = args[2]
            return self._dir_manifest_data(str(args[0]), recursive=recursive, cfg=cfg)
        if name == "dir_diff":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("dir_diff(left, right[, cfg]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            return self._dir_diff(str(args[0]), str(args[1]), cfg=cfg)
        if name == "resource_count":
            if args:
                raise RuntimeErrorNF("resource_count() expects 0 args")
            with self._lock:
                return len((self.resource_state or {}).get("ops", []))
        if name == "resource_last":
            if args:
                raise RuntimeErrorNF("resource_last() expects 0 args")
            with self._lock:
                ops = copy.deepcopy((self.resource_state or {}).get("ops", []))
            return ops[-1] if ops else None
        if name == "resource_limits":
            if args:
                raise RuntimeErrorNF("resource_limits() expects 0 args")
            with self._lock:
                return copy.deepcopy((self.resource_state or {}).get("limits", {}))
        if name == "resource_runtime_info":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("resource_runtime_info([label]) expects 0 or 1 args")
            label = str(args[0]) if args else None
            return self._resource_runtime_snapshot(label=label)
        if name == "resource_host_info":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("resource_host_info([label]) expects 0 or 1 args")
            label = str(args[0]) if args else None
            return self._resource_host_snapshot(label=label)
        if name == "resource_memory_bytes":
            if args:
                raise RuntimeErrorNF("resource_memory_bytes() expects 0 args")
            return self._process_memory_bytes()
        if name == "cpu_count":
            if args:
                raise RuntimeErrorNF("cpu_count() expects 0 args")
            return os.cpu_count() or 1
        if name == "thread_count":
            if args:
                raise RuntimeErrorNF("thread_count() expects 0 args")
            return threading.active_count()
        if name == "dir_stats":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("dir_stats(path[, recursive]) expects 1 or 2 args")
            recursive = bool(args[1]) if len(args) == 2 else True
            return self._dir_stats(str(args[0]), recursive=recursive)
        if name == "http_get_text":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("http_get_text(url[, timeout_or_preset_or_headers[, timeout]]) expects 1-3 args")
            timeout_sec = 15.0
            headers = None
            auth_preset = None
            if len(args) >= 2:
                if isinstance(args[1], (int, float)):
                    timeout_sec = float(args[1])
                elif isinstance(args[1], str):
                    auth_preset = str(args[1])
                elif isinstance(args[1], dict):
                    headers = args[1]
            if len(args) == 3:
                timeout_sec = float(args[2])
            return self._http_get_text(str(args[0]), timeout_sec=timeout_sec, headers=headers, auth_preset=auth_preset)
        if name == "http_get_json":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("http_get_json(url[, timeout_or_preset_or_headers[, timeout]]) expects 1-3 args")
            timeout_sec = 15.0
            headers = None
            auth_preset = None
            if len(args) >= 2:
                if isinstance(args[1], (int, float)):
                    timeout_sec = float(args[1])
                elif isinstance(args[1], str):
                    auth_preset = str(args[1])
                elif isinstance(args[1], dict):
                    headers = args[1]
            if len(args) == 3:
                timeout_sec = float(args[2])
            return self._http_get_json(str(args[0]), timeout_sec=timeout_sec, headers=headers, auth_preset=auth_preset)
        if name == "http_post_json":
            if len(args) not in {2, 3, 4, 5}:
                raise RuntimeErrorNF("http_post_json(url, payload[, timeout_or_preset_or_headers[, timeout_or_headers[, headers]]]) expects 2-5 args")
            timeout_sec = 15.0
            headers = None
            auth_preset = None
            tail = list(args[2:])
            for item in tail:
                if isinstance(item, (int, float)):
                    timeout_sec = float(item)
                elif isinstance(item, str):
                    auth_preset = str(item)
                elif isinstance(item, dict):
                    headers = item
            return self._http_post_json(str(args[0]), args[1], timeout_sec=timeout_sec, headers=headers, auth_preset=auth_preset)
        if name == "asset3d_count":
            if args:
                raise RuntimeErrorNF("asset3d_count() expects 0 args")
            with self._lock:
                return len(self.assets3d)
        if name == "scene3d_count":
            if args:
                raise RuntimeErrorNF("scene3d_count() expects 0 args")
            with self._lock:
                return len(self.scenes3d)
        if name == "asset3d_info":
            if len(args) != 1:
                raise RuntimeErrorNF("asset3d_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy(self.assets3d.get(str(args[0])))
        if name == "scene3d_info":
            if len(args) != 1:
                raise RuntimeErrorNF("scene3d_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy(self.scenes3d.get(str(args[0])))
        if name == "mesh_vertex_count":
            if len(args) != 1:
                raise RuntimeErrorNF("mesh_vertex_count(asset) expects 1 arg")
            with self._lock:
                asset = copy.deepcopy(self.assets3d.get(str(args[0]), {}))
            return _as_int(((asset.get("stats") or {}).get("vertices", 0))) if isinstance(asset, dict) else 0
        if name == "mesh_face_count":
            if len(args) != 1:
                raise RuntimeErrorNF("mesh_face_count(asset) expects 1 arg")
            with self._lock:
                asset = copy.deepcopy(self.assets3d.get(str(args[0]), {}))
            if not isinstance(asset, dict):
                return 0
            stats = asset.get("stats") or {}
            if isinstance(stats, dict):
                return _as_int(stats.get("faces", stats.get("primitives", 0)))
            return 0
        if name == "scene_node_count":
            if len(args) != 1:
                raise RuntimeErrorNF("scene_node_count(scene) expects 1 arg")
            scene_name = str(args[0])
            with self._lock:
                scene = copy.deepcopy(self.scenes3d.get(scene_name, {}))
            if isinstance(scene, dict):
                nodes = scene.get("nodes")
                if isinstance(nodes, list):
                    return len(nodes)
            return 0
        if name == "fusion_run_count":
            if args:
                raise RuntimeErrorNF("fusion_run_count() expects 0 args")
            with self._lock:
                return len(self.fusion_runs)
        if name == "fusion_info":
            if len(args) != 1:
                raise RuntimeErrorNF("fusion_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy(self.fusion_runs.get(str(args[0])))
        if name == "fusion_metric":
            if len(args) != 2:
                raise RuntimeErrorNF("fusion_metric(run, key) expects 2 args")
            with self._lock:
                run = copy.deepcopy(self.fusion_runs.get(str(args[0])))
            if not isinstance(run, dict):
                return None
            key = str(args[1])
            v = self._deep_get(run.get("metrics", {}), key, None)
            if v is None:
                v = self._deep_get(run, key, None)
            return v
        if name == "protein_run_count":
            if args:
                raise RuntimeErrorNF("protein_run_count() expects 0 args")
            with self._lock:
                return len(self.protein_runs)
        if name == "protein_info":
            if len(args) != 1:
                raise RuntimeErrorNF("protein_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy(self.protein_runs.get(str(args[0])))
        if name == "protein_metric":
            if len(args) != 2:
                raise RuntimeErrorNF("protein_metric(run, key) expects 2 args")
            with self._lock:
                run = copy.deepcopy(self.protein_runs.get(str(args[0])))
            if not isinstance(run, dict):
                return None
            key = str(args[1])
            for root in ("best", "final", ""):
                source = run if root == "" else run.get(root, {})
                v = self._deep_get(source, key, None) if isinstance(source, (dict, list)) else None
                if v is not None:
                    return v
            return None
        if name == "python_model_count":
            if args:
                raise RuntimeErrorNF("python_model_count() expects 0 args")
            with self._lock:
                return len(self._python_trained_models)
        if name == "python_model_info":
            if len(args) != 1:
                raise RuntimeErrorNF("python_model_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy(self._python_trained_models.get(str(args[0])))
        if name == "python_model_metric":
            if len(args) != 2:
                raise RuntimeErrorNF("python_model_metric(name, key) expects 2 args")
            with self._lock:
                rec = copy.deepcopy(self._python_trained_models.get(str(args[0])))
            if not isinstance(rec, dict):
                return None
            key = str(args[1])
            v = self._deep_get(rec.get("metrics", {}), key, None)
            if v is None:
                v = self._deep_get(rec, key, None)
            return v
        if name == "protein_length":
            if len(args) != 1:
                raise RuntimeErrorNF("protein_length(sequence) expects 1 arg")
            return len(re.sub(r"[^A-Za-z]", "", str(args[0])))
        if name == "protein_hydrophobicity":
            if len(args) != 1:
                raise RuntimeErrorNF("protein_hydrophobicity(sequence) expects 1 arg")
            seq = re.sub(r"[^A-Za-z]", "", str(args[0]).upper())
            hydro = set("AILMFWVYC")
            return (sum(1 for aa in seq if aa in hydro) / len(seq)) if seq else 0
        if name == "sha256":
            if len(args) != 1:
                raise RuntimeErrorNF("sha256(x) expects 1 arg")
            return hashlib.sha256(self._safe_text(args[0]).encode("utf-8")).hexdigest()
        if name == "uuid4":
            if args:
                raise RuntimeErrorNF("uuid4() expects 0 args")
            return str(uuid.uuid4())
        if name == "win_clipboard_get":
            if args:
                raise RuntimeErrorNF("win_clipboard_get() expects 0 args")
            return self._win_clipboard_get()
        if name == "win_service_status":
            if len(args) != 1:
                raise RuntimeErrorNF("win_service_status(name) expects 1 arg")
            return self._win_service_status(str(args[0]))
        if name == "win_registry_get":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("win_registry_get(path, name[, default]) expects 2 or 3 args")
            default = args[2] if len(args) == 3 else None
            return self._win_registry_get(str(args[0]), str(args[1]), default)
        if name == "win_op_count":
            if args:
                raise RuntimeErrorNF("win_op_count() expects 0 args")
            with self._lock:
                return len(self.windows_ops)
        if name == "win_last_op":
            if args:
                raise RuntimeErrorNF("win_last_op() expects 0 args")
            with self._lock:
                return copy.deepcopy(self.windows_ops[-1]) if self.windows_ops else None
        if name == "win_host_info":
            if args:
                raise RuntimeErrorNF("win_host_info() expects 0 args")
            return self._host_info()
        if name == "win_mouse_pos":
            if args:
                raise RuntimeErrorNF("win_mouse_pos() expects 0 args")
            return self._win_mouse_pos()
        if name == "win_screen_size":
            if args:
                raise RuntimeErrorNF("win_screen_size() expects 0 args")
            return self._win_screen_size()
        if name == "win_foreground_window":
            if args:
                raise RuntimeErrorNF("win_foreground_window() expects 0 args")
            return self._win_foreground_window()
        if name == "win_windows":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("win_windows([top_n]) expects 0 or 1 args")
            top_n = _as_int(args[0]) if args else 20
            payload = self._win_windows_list(top_n=top_n)
            return payload.get("windows", [])
        if name == "vui_supported":
            if args:
                raise RuntimeErrorNF("vui_supported() expects 0 args")
            return self._vui_supported()
        if name == "vui_op_count":
            if args:
                raise RuntimeErrorNF("vui_op_count() expects 0 args")
            with self._lock:
                return len(self.vui_state.get("ops", []))
        if name == "vui_last":
            if args:
                raise RuntimeErrorNF("vui_last() expects 0 args")
            with self._lock:
                ops = self.vui_state.get("ops", [])
                return copy.deepcopy(ops[-1]) if ops else None
        if name == "vui_profile_info":
            if len(args) != 1:
                raise RuntimeErrorNF("vui_profile_info(name) expects 1 arg")
            return self._vui_profile_get(str(args[0]))
        if name == "vui_voices":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("vui_voices([refresh]) expects 0 or 1 args")
            refresh = bool(args[0]) if args else False
            return self._vui_list_voices(refresh=refresh).get("voices", [])
        if name == "vui_voice_count":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("vui_voice_count([refresh]) expects 0 or 1 args")
            refresh = bool(args[0]) if args else False
            return len(self._vui_list_voices(refresh=refresh).get("voices", []))
        if name == "vui_intent":
            if len(args) != 2:
                raise RuntimeErrorNF("vui_intent(text, intents_map) expects 2 args")
            return self._vui_intent(str(args[0]), args[1])
        if name == "vui_extract":
            if len(args) != 2:
                raise RuntimeErrorNF("vui_extract(text, slots_map) expects 2 args")
            return self._vui_extract(str(args[0]), args[1])
        if name == "get":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("get(container, key[, default]) expects 2 or 3 args")
            container, key = args[0], args[1]
            default = args[2] if len(args) == 3 else None
            if isinstance(container, dict):
                return container.get(key, default)
            if isinstance(container, list):
                try:
                    idx = _as_int(key)
                    return container[idx]
                except Exception:
                    return default
            return default
        if name == "keys":
            if len(args) != 1:
                raise RuntimeErrorNF("keys(map) expects 1 arg")
            return list(args[0].keys()) if isinstance(args[0], dict) else []
        if name == "values":
            if len(args) != 1:
                raise RuntimeErrorNF("values(map) expects 1 arg")
            return list(args[0].values()) if isinstance(args[0], dict) else []
        if name == "merge":
            if not args:
                raise RuntimeErrorNF("merge(map1[, map2...]) expects at least one arg")
            out: Dict[Any, Any] = {}
            for item in args:
                if not isinstance(item, dict):
                    raise RuntimeErrorNF("merge() expects dict arguments")
                out.update(item)
            return out
        if name == "len":
            if len(args) != 1:
                raise RuntimeErrorNF("len(x) expects 1 arg")
            v = args[0]
            if v is None:
                return 0
            if isinstance(v, (str, list, dict, tuple, set)):
                return len(v)
            raise RuntimeErrorNF(f"len() unsupported for {type(v).__name__}")
        if name == "contains":
            if len(args) != 2:
                raise RuntimeErrorNF("contains(container, item) expects 2 args")
            container, item = args
            if container is None:
                return False
            if isinstance(container, (str, list, tuple, set)):
                return item in container
            if isinstance(container, dict):
                return item in container
            raise RuntimeErrorNF(f"contains() unsupported for {type(container).__name__}")
        if name == "lower":
            if len(args) != 1:
                raise RuntimeErrorNF("lower(text) expects 1 arg")
            return str(args[0]).lower()
        if name == "upper":
            if len(args) != 1:
                raise RuntimeErrorNF("upper(text) expects 1 arg")
            return str(args[0]).upper()
        if name == "strip":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("strip(text[, chars]) expects 1 or 2 args")
            s = str(args[0])
            return s.strip(None if len(args) == 1 or args[1] is None else str(args[1]))
        if name == "split":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("split(text[, sep]) expects 1 or 2 args")
            s = str(args[0])
            if len(args) == 1 or args[1] is None:
                return s.split()
            return s.split(str(args[1]))
        if name == "join":
            if len(args) != 2:
                raise RuntimeErrorNF("join(sep, items) expects 2 args")
            sep = str(args[0])
            items = args[1]
            if not isinstance(items, list):
                raise RuntimeErrorNF("join() second arg must be list")
            return sep.join(str(i) for i in items)
        if name == "replace":
            if len(args) != 3:
                raise RuntimeErrorNF("replace(text, old, new) expects 3 args")
            return str(args[0]).replace(str(args[1]), str(args[2]))
        if name == "to_number":
            if len(args) != 1:
                raise RuntimeErrorNF("to_number(x) expects 1 arg")
            v = args[0]
            if isinstance(v, (int, float)):
                return v
            s = str(v).strip()
            try:
                return int(s) if re.fullmatch(r"[+-]?\d+", s) else float(s)
            except Exception as exc:
                raise RuntimeErrorNF(f"to_number() failed for {v!r}: {exc}") from exc
        if name == "to_string":
            if len(args) != 1:
                raise RuntimeErrorNF("to_string(x) expects 1 arg")
            return self._safe_text(args[0])
        if name == "json_stringify":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("json_stringify(value[, pretty]) expects 1 or 2 args")
            pretty = bool(args[1]) if len(args) == 2 else False
            return json.dumps(args[0], ensure_ascii=False, indent=2 if pretty else None)
        if name == "json_parse":
            if len(args) != 1:
                raise RuntimeErrorNF("json_parse(text) expects 1 arg")
            return json.loads(str(args[0]))
        if name == "json_path_get":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("json_path_get(value, path[, default]) expects 2 or 3 args")
            default = args[2] if len(args) == 3 else None
            return self._deep_get(args[0], str(args[1]), default)
        if name == "template_fill":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("template_fill(template, data[, strict]) expects 2 or 3 args")
            strict = bool(args[2]) if len(args) == 3 else False
            return self._template_fill(str(args[0]), args[1], strict=strict)
        if name == "regex_test":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("regex_test(pattern, text[, flags]) expects 2 or 3 args")
            flags = self._regex_flags(args[2]) if len(args) == 3 else 0
            return re.search(str(args[0]), str(args[1]), flags) is not None
        if name == "regex_findall":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("regex_findall(pattern, text[, flags]) expects 2 or 3 args")
            flags = self._regex_flags(args[2]) if len(args) == 3 else 0
            return re.findall(str(args[0]), str(args[1]), flags)
        if name == "regex_replace":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("regex_replace(pattern, repl, text[, flags]) expects 3 or 4 args")
            flags = self._regex_flags(args[3]) if len(args) == 4 else 0
            return re.sub(str(args[0]), str(args[1]), str(args[2]), flags=flags)
        if name == "regex_match":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("regex_match(pattern, text[, flags]) expects 2 or 3 args")
            flags = self._regex_flags(args[2]) if len(args) == 3 else 0
            m = re.search(str(args[0]), str(args[1]), flags)
            if not m:
                return None
            return {
                "match": m.group(0),
                "groups": list(m.groups()),
                "named": m.groupdict(),
                "span": [int(m.start()), int(m.end())],
            }
        if name == "accelerator_info":
            if len(args) not in {0, 1, 2}:
                raise RuntimeErrorNF("accelerator_info([refresh[, deep]]) expects 0-2 args")
            refresh = bool(args[0]) if len(args) >= 1 else False
            deep = bool(args[1]) if len(args) == 2 else False
            return self._accelerator_probe(refresh=refresh, deep=deep)
        if name == "npu_info":
            if len(args) not in {0, 1, 2}:
                raise RuntimeErrorNF("npu_info([refresh[, deep]]) expects 0-2 args")
            refresh = bool(args[0]) if len(args) >= 1 else False
            deep = bool(args[1]) if len(args) == 2 else False
            return (self._accelerator_probe(refresh=refresh, deep=deep) or {}).get("npu", {})
        if name == "npu_available":
            if len(args) not in {0, 1, 2}:
                raise RuntimeErrorNF("npu_available([refresh[, deep]]) expects 0-2 args")
            refresh = bool(args[0]) if len(args) >= 1 else False
            deep = bool(args[1]) if len(args) == 2 else False
            info = self._accelerator_probe(refresh=refresh, deep=deep)
            return bool(((info.get("npu") or {}).get("available")))
        if name == "npu_probe_count":
            if args:
                raise RuntimeErrorNF("npu_probe_count() expects 0 args")
            with self._lock:
                return len(self.npu_state.get("probes", []))
        if name == "npu_profile_count":
            if args:
                raise RuntimeErrorNF("npu_profile_count() expects 0 args")
            with self._lock:
                return len((self.npu_state.get("profiles") or {}))
        if name == "npu_profile_info":
            if len(args) != 1:
                raise RuntimeErrorNF("npu_profile_info(name) expects 1 arg")
            return self._npu_profile_get(str(args[0]))
        if name == "npu_run_count":
            if args:
                raise RuntimeErrorNF("npu_run_count() expects 0 args")
            with self._lock:
                return len((self.npu_state.get("runs") or []))
        if name == "npu_last_run":
            if args:
                raise RuntimeErrorNF("npu_last_run() expects 0 args")
            with self._lock:
                return copy.deepcopy(self.npu_state.get("last_run"))
        if name == "npu_last_plan":
            if args:
                raise RuntimeErrorNF("npu_last_plan() expects 0 args")
            with self._lock:
                return copy.deepcopy(self.npu_state.get("last_plan"))
        if name == "npu_last_benchmark":
            if args:
                raise RuntimeErrorNF("npu_last_benchmark() expects 0 args")
            with self._lock:
                return copy.deepcopy(self.npu_state.get("last_benchmark"))
        if name == "npu_provider_count":
            if len(args) not in {0, 1, 2}:
                raise RuntimeErrorNF("npu_provider_count([refresh[, deep]]) expects 0-2 args")
            refresh = bool(args[0]) if len(args) >= 1 else False
            deep = bool(args[1]) if len(args) == 2 else False
            probe = self._accelerator_probe(refresh=refresh, deep=deep)
            return len(self._npu_detect_providers(probe))
        if name == "npu_plan":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("npu_plan(workload_or_model[, config]) expects 1 or 2 args")
            cfg = args[1] if len(args) == 2 and isinstance(args[1], dict) else None
            return self._npu_plan(args[0], cfg=cfg)
        if name == "npu_benchmark":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("npu_benchmark(workload_or_model[, config]) expects 1 or 2 args")
            cfg = args[1] if len(args) == 2 and isinstance(args[1], dict) else None
            return self._npu_benchmark(args[0], cfg=cfg)
        if name == "wifi_supported":
            if args:
                raise RuntimeErrorNF("wifi_supported() expects 0 args")
            return self._is_windows_host and shutil.which("netsh") is not None
        if name == "wifi_interfaces":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("wifi_interfaces([refresh]) expects 0 or 1 arg")
            refresh = bool(args[0]) if args else False
            if refresh:
                return self._wifi_interfaces().get("interfaces", [])
            with self._lock:
                return copy.deepcopy(self.wifi_state.get("interfaces", []))
        if name == "wifi_profiles":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("wifi_profiles([refresh]) expects 0 or 1 arg")
            refresh = bool(args[0]) if args else False
            if refresh:
                return self._wifi_profiles().get("profiles", [])
            with self._lock:
                return copy.deepcopy(self.wifi_state.get("profiles", []))
        if name == "wifi_last_scan":
            if args:
                raise RuntimeErrorNF("wifi_last_scan() expects 0 args")
            with self._lock:
                return copy.deepcopy(self.wifi_state.get("last_scan"))
        if name == "wifi_interface_count":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("wifi_interface_count([refresh]) expects 0 or 1 arg")
            if args and bool(args[0]):
                return _as_int(self._wifi_interfaces().get("count", 0))
            with self._lock:
                return len(self.wifi_state.get("interfaces", []))
        if name == "wifi_profile_count":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("wifi_profile_count([refresh]) expects 0 or 1 arg")
            if args and bool(args[0]):
                return _as_int(self._wifi_profiles().get("count", 0))
            with self._lock:
                return len(self.wifi_state.get("profiles", []))
        if name == "proc_op_count":
            if args:
                raise RuntimeErrorNF("proc_op_count() expects 0 args")
            with self._lock:
                return len(self.process_state.get("ops", []))
        if name == "proc_last":
            if args:
                raise RuntimeErrorNF("proc_last() expects 0 args")
            with self._lock:
                ops = copy.deepcopy(self.process_state.get("ops", []))
            return ops[-1] if ops else None
        if name == "proc_managed_count":
            if args:
                raise RuntimeErrorNF("proc_managed_count() expects 0 args")
            with self._lock:
                return len(self.process_state.get("managed", {}))
        if name == "proc_managed_info":
            if len(args) != 1:
                raise RuntimeErrorNF("proc_managed_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy((self.process_state.get("managed", {}) or {}).get(str(args[0])))
        if name == "proc_profile_info":
            if len(args) != 1:
                raise RuntimeErrorNF("proc_profile_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy((self.process_state.get("profiles", {}) or {}).get(str(args[0])))
        if name == "photo_count":
            if args:
                raise RuntimeErrorNF("photo_count() expects 0 args")
            with self._lock:
                return len(self.photo_state.get("images", {}))
        if name == "photo_info":
            if len(args) != 1:
                raise RuntimeErrorNF("photo_info(path_or_name) expects 1 arg")
            key = str(args[0])
            with self._lock:
                imgs = copy.deepcopy(self.photo_state.get("images", {}))
            if key in imgs:
                return imgs[key]
            for pth, meta in imgs.items():
                if Path(str(pth)).name == key:
                    return meta
            return None
        if name == "photo_list":
            if args:
                raise RuntimeErrorNF("photo_list() expects 0 args")
            with self._lock:
                imgs = copy.deepcopy(self.photo_state.get("images", {}))
            return [imgs[k] for k in sorted(imgs.keys(), key=lambda v: str(v).lower())]
        if name == "photo_op_count":
            if args:
                raise RuntimeErrorNF("photo_op_count() expects 0 args")
            with self._lock:
                return len(self.photo_state.get("ops", []))
        if name == "photo_last":
            if args:
                raise RuntimeErrorNF("photo_last() expects 0 args")
            with self._lock:
                ops = copy.deepcopy(self.photo_state.get("ops", []))
            return ops[-1] if ops else None
        if name == "photo_batch_count":
            if args:
                raise RuntimeErrorNF("photo_batch_count() expects 0 args")
            with self._lock:
                return len(self.photo_state.get("batches", {}))
        if name == "photo_batch_info":
            if len(args) != 1:
                raise RuntimeErrorNF("photo_batch_info(name_or_path) expects 1 arg")
            return self._photo_batch_find(args[0])
        if name == "photo_batch_list":
            if args:
                raise RuntimeErrorNF("photo_batch_list() expects 0 args")
            with self._lock:
                batches = copy.deepcopy(self.photo_state.get("batches", {}))
            return [batches[k] for k in sorted(batches.keys(), key=lambda v: str(v).lower())]
        if name == "graph_count":
            if args:
                raise RuntimeErrorNF("graph_count() expects 0 args")
            with self._lock:
                return len(self.graph_state.get("graphs", {}))
        if name == "graph_info":
            if len(args) != 1:
                raise RuntimeErrorNF("graph_info(name) expects 1 arg")
            with self._lock:
                return copy.deepcopy((self.graph_state.get("graphs", {}) or {}).get(str(args[0])))
        if name == "graph_stats":
            if len(args) != 1:
                raise RuntimeErrorNF("graph_stats(graph_or_name) expects 1 arg")
            return self._graph_stats(args[0])
        if name == "graph_degrees":
            if len(args) != 1:
                raise RuntimeErrorNF("graph_degrees(graph_or_name) expects 1 arg")
            return self._graph_degrees(args[0])
        if name == "graph_shortest_path":
            if len(args) != 3:
                raise RuntimeErrorNF("graph_shortest_path(graph_or_name, source, target) expects 3 args")
            return self._graph_shortest_path(args[0], args[1], args[2])
        if name == "graph_components":
            if len(args) != 1:
                raise RuntimeErrorNF("graph_components(graph_or_name) expects 1 arg")
            return self._graph_components(args[0])
        if name == "convert_modes":
            if args:
                raise RuntimeErrorNF("convert_modes() expects 0 args")
            return self._convert_modes()
        if name == "convert_last":
            if args:
                raise RuntimeErrorNF("convert_last() expects 0 args")
            with self._lock:
                ops = copy.deepcopy(self.convert_state.get("ops", []))
            return ops[-1] if ops else None
        if name == "convert_op_count":
            if args:
                raise RuntimeErrorNF("convert_op_count() expects 0 args")
            with self._lock:
                return len(self.convert_state.get("ops", []))
        if name == "exe_count":
            if args:
                raise RuntimeErrorNF("exe_count() expects 0 args")
            with self._lock:
                return len((self.exe_state or {}).get("artifacts", {}))
        if name == "exe_op_count":
            if args:
                raise RuntimeErrorNF("exe_op_count() expects 0 args")
            with self._lock:
                return len((self.exe_state or {}).get("ops", []))
        if name == "exe_last":
            if args:
                raise RuntimeErrorNF("exe_last() expects 0 args")
            with self._lock:
                last = (self.exe_state or {}).get("last")
                if last is not None:
                    return copy.deepcopy(last)
                ops = (self.exe_state or {}).get("ops", [])
                return copy.deepcopy(ops[-1]) if ops else None
        if name == "exe_info":
            if len(args) != 1:
                raise RuntimeErrorNF("exe_info(path_or_name) expects 1 arg")
            key = str(args[0])
            with self._lock:
                artifacts = copy.deepcopy((self.exe_state or {}).get("artifacts", {}))
            if key in artifacts:
                return artifacts[key]
            for path, meta in artifacts.items():
                if Path(str(path)).name == key:
                    return meta
            return None
        if name == "exe_tool_info":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("exe_tool_info([refresh]) expects 0 or 1 arg")
            refresh = bool(args[0]) if len(args) == 1 else False
            return self._exe_tool_info(refresh=refresh)
        if name == "iso_tool_info":
            if args:
                raise RuntimeErrorNF("iso_tool_info() expects 0 args")
            return self._iso_tool_info()
        if name == "iso_last":
            if args:
                raise RuntimeErrorNF("iso_last() expects 0 args")
            with self._lock:
                return copy.deepcopy(self.iso_state.get("last"))
        if name == "iso_build_count":
            if args:
                raise RuntimeErrorNF("iso_build_count() expects 0 args")
            with self._lock:
                return len(self.iso_state.get("ops", []))
        if name == "github_local_summary":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("github_local_summary([path]) expects 0 or 1 arg")
            return self._github_local_summary(str(args[0])) if args else self._github_local_summary()
        if name == "github_repo_find":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("github_repo_find(term[, path]) expects 1 or 2 args")
            path = str(args[1]) if len(args) == 2 else "repos_metadata.json"
            return self._github_repo_find(str(args[0]), path)
        if name == "rust_toolchain_info":
            if args:
                raise RuntimeErrorNF("rust_toolchain_info() expects 0 args")
            return self._rust_toolchain_info()
        if name == "stats_summary":
            if len(args) != 1:
                raise RuntimeErrorNF("stats_summary(values) expects 1 arg")
            return self._stats_summary(args[0])
        if name == "mean":
            if len(args) != 1:
                raise RuntimeErrorNF("mean(values) expects 1 arg")
            vals = self._numeric_series(args[0])
            return (sum(vals) / len(vals)) if vals else 0
        if name == "median":
            if len(args) != 1:
                raise RuntimeErrorNF("median(values) expects 1 arg")
            vals = sorted(self._numeric_series(args[0]))
            if not vals:
                return 0
            n = len(vals)
            mid = n // 2
            return vals[mid] if n % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2.0
        if name == "variance":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("variance(values[, sample]) expects 1 or 2 args")
            vals = self._numeric_series(args[0])
            if not vals:
                return 0
            sample = bool(args[1]) if len(args) == 2 else False
            mean_v = sum(vals) / len(vals)
            denom = len(vals) - 1 if sample and len(vals) > 1 else len(vals)
            return (sum((v - mean_v) ** 2 for v in vals) / max(1, denom))
        if name == "stddev":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("stddev(values[, sample]) expects 1 or 2 args")
            vals = self._numeric_series(args[0])
            if not vals:
                return 0
            sample = bool(args[1]) if len(args) == 2 else False
            mean_v = sum(vals) / len(vals)
            denom = len(vals) - 1 if sample and len(vals) > 1 else len(vals)
            var = sum((v - mean_v) ** 2 for v in vals) / max(1, denom)
            return math.sqrt(var)
        if name == "linspace":
            if len(args) != 3:
                raise RuntimeErrorNF("linspace(start, stop, count) expects 3 args")
            start = float(args[0])
            stop = float(args[1])
            count = max(0, _as_int(args[2]))
            if count <= 0:
                return []
            if count == 1:
                return [start]
            return [start + (stop - start) * (i / (count - 1)) for i in range(count)]
        if name == "dot":
            if len(args) != 2:
                raise RuntimeErrorNF("dot(a, b) expects 2 args")
            a = self._numeric_series(args[0])
            b = self._numeric_series(args[1])
            n = min(len(a), len(b))
            return sum(a[i] * b[i] for i in range(n))
        if name == "norm":
            if len(args) != 1:
                raise RuntimeErrorNF("norm(v) expects 1 arg")
            v = self._numeric_series(args[0])
            return math.sqrt(sum(x * x for x in v))
        if name == "distance":
            if len(args) != 2:
                raise RuntimeErrorNF("distance(a, b) expects 2 args")
            a = self._numeric_series(args[0])
            b = self._numeric_series(args[1])
            n = min(len(a), len(b))
            return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))
        if name == "differentiate":
            if len(args) != 2:
                raise RuntimeErrorNF("differentiate(xs, ys) expects 2 args")
            return self._differentiate(args[0], args[1])
        if name == "integrate_trapz":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("integrate_trapz(ys[, xs_or_dx]) expects 1 or 2 args")
            return self._integrate_trapz(args[0], args[1] if len(args) == 2 else None)
        if name == "poly_eval":
            if len(args) != 2:
                raise RuntimeErrorNF("poly_eval(coeffs, x) expects 2 args")
            return self._poly_eval(args[0], args[1])
        if name == "linear_fit":
            if len(args) != 2:
                raise RuntimeErrorNF("linear_fit(xs, ys) expects 2 args")
            return self._linear_fit(args[0], args[1])
        if name == "rand":
            rng = self._get_rng()
            if len(args) == 0:
                return rng.random()
            if len(args) == 2:
                return rng.uniform(_num(args[0]), _num(args[1]))
            raise RuntimeErrorNF("rand() expects 0 or 2 args")
        if name == "randint":
            if len(args) != 2:
                raise RuntimeErrorNF("randint(a,b) expects 2 args")
            return self._get_rng().randint(_as_int(args[0]), _as_int(args[1]))
        if name == "randn":
            if len(args) not in {0, 2}:
                raise RuntimeErrorNF("randn() or randn(mu,sigma)")
            rng = self._get_rng()
            if len(args) == 0:
                return rng.gauss(0.0, 1.0)
            return rng.gauss(float(args[0]), float(args[1]))
        if name == "min":
            return min(args)
        if name == "max":
            return max(args)
        if name == "sum":
            return sum(args)
        if name == "abs":
            if len(args) != 1:
                raise RuntimeErrorNF("abs(x) expects 1 arg")
            return abs(args[0])
        if name == "clamp":
            if len(args) != 3:
                raise RuntimeErrorNF("clamp(x,a,b) expects 3 args")
            x, a, b = args
            return max(a, min(x, b))
        if name == "log":
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "log": args})
            return None
        if name == "thread_id":
            return threading.get_ident()
        if name == "agent_count":
            if len(args) != 1:
                raise RuntimeErrorNF("agent_count(name) expects 1 arg")
            with self._lock:
                return len(self.agents.get(str(args[0]), []))
        if name == "avg_agent":
            if len(args) != 2:
                raise RuntimeErrorNF("avg_agent(type, field) expects 2 args")
            with self._lock:
                items = list(self.agents.get(str(args[0]), []))
            field_name = str(args[1])
            vals = [i.get(field_name) for i in items if isinstance(i.get(field_name), (int, float))]
            return (sum(vals) / len(vals)) if vals else 0
        if name == "sum_agent":
            if len(args) != 2:
                raise RuntimeErrorNF("sum_agent(type, field) expects 2 args")
            with self._lock:
                items = list(self.agents.get(str(args[0]), []))
            field_name = str(args[1])
            vals = [i.get(field_name) for i in items if isinstance(i.get(field_name), (int, float))]
            return sum(vals) if vals else 0
        if name == "count_events":
            if len(args) > 1:
                raise RuntimeErrorNF("count_events([label]) expects 0 or 1 args")
            with self._lock:
                evts = list(self.events)
            if not args:
                return len(evts)
            label = str(args[0])
            return sum(1 for e in evts if e.get("event") == label)
        if name == "send":
            if len(args) != 2:
                raise RuntimeErrorNF("send(channel, value) expects 2 args")
            return self._channel_send(str(args[0]), args[1])
        if name == "recv":
            if len(args) != 1:
                raise RuntimeErrorNF("recv(channel) expects 1 arg")
            return self._channel_recv(str(args[0]))
        if name == "drain":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("drain(channel[, limit]) expects 1 or 2 args")
            channel = str(args[0])
            limit = None if len(args) == 1 or args[1] is None else max(0, _as_int(args[1]))
            with self._lock:
                q = self.channels.setdefault(channel, [])
                if limit is None:
                    items = list(q)
                    q.clear()
                else:
                    items = q[:limit]
                    del q[:limit]
            return items
        if name == "peek":
            if len(args) != 1:
                raise RuntimeErrorNF("peek(channel) expects 1 arg")
            return self._channel_peek(str(args[0]))
        if name == "queue_size":
            if len(args) != 1:
                raise RuntimeErrorNF("queue_size(channel) expects 1 arg")
            return self._channel_size(str(args[0]))
        raise RuntimeErrorNF(f"Unknown function: {name}")

    def exec_stmt(self, stmt: Stmt, local: Optional[Dict[str, Any]]) -> Any:
        if isinstance(stmt, Assign):
            value = self.eval_expr(stmt.expr, local)
            if local is not None and stmt.name in local:
                target = local
            elif stmt.name in self.state:
                target = self.state
            elif local is not None:
                target = local
            else:
                target = self.state

            if target is self.state:
                with self._lock:
                    self._apply_assign(target, stmt.name, stmt.op, value)
            else:
                self._apply_assign(target, stmt.name, stmt.op, value)
            return None

        if isinstance(stmt, If):
            body = stmt.then_body if self.eval_expr(stmt.cond, local) else stmt.else_body
            for sub in body:
                self.exec_stmt(sub, local)
            return None

        if isinstance(stmt, Emit):
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": self.eval_expr(stmt.expr, local)})
            return None

        if isinstance(stmt, ExprStmt):
            return self.eval_expr(stmt.expr, local)

        raise RuntimeErrorNF(f"Unsupported statement {stmt!r}")

    def _apply_assign(self, target: Dict[str, Any], name: str, op: str, value: Any) -> None:
        if op == "=":
            target[name] = value
            return
        cur = target.get(name, 0)
        if op == "+=":
            target[name] = cur + value
        elif op == "-=":
            target[name] = cur - value
        elif op == "*=":
            target[name] = cur * value
        elif op == "/=":
            target[name] = cur / value
        else:
            raise RuntimeErrorNF(f"Unsupported assignment op {op}")

    def simulate(self, ticks: int) -> None:
        if ticks < 0:
            raise RuntimeErrorNF("simulate ticks must be >= 0")
        self.thread_stats["simulate_calls"] += 1
        self._record_thread_event("simulate", mode="single", ticks=ticks)
        agent_decl_map = {a.name: a for a in self.project.agents}
        for _ in range(ticks):
            with self._lock:
                self.tick_count += 1
                tick_now = self.tick_count
            for agent_name, instances in self.agents.items():
                decl = agent_decl_map.get(agent_name)
                if decl is None:
                    continue
                for inst in instances:
                    inst["tick"] = tick_now
                    for stmt in decl.on_tick:
                        self.exec_stmt(stmt, inst)

    def _run_agent_chunk(self, decl: AgentDecl, instances: List[Dict[str, Any]], tick_now: int) -> None:
        for inst in instances:
            inst["tick"] = tick_now
            for stmt in decl.on_tick:
                self.exec_stmt(stmt, inst)

    def simulate_mt(self, ticks: int, workers: int, chunk_size: int = 0) -> None:
        if ticks < 0:
            raise RuntimeErrorNF("simulate_mt ticks must be >= 0")
        if workers < 1:
            raise RuntimeErrorNF("simulate_mt workers must be >= 1")
        with self._lock:
            max_sim_workers = (self.resource_state.get("limits", {}) or {}).get("max_sim_workers")
        if isinstance(max_sim_workers, int):
            workers = max(1, min(int(workers), int(max_sim_workers)))
        if workers == 1:
            self.simulate(ticks)
            return

        self.thread_stats["simulate_mt_calls"] += 1
        self.thread_stats["max_workers_seen"] = max(self.thread_stats["max_workers_seen"], workers)
        self._record_thread_event("simulate", mode="multithread", ticks=ticks, workers=workers)

        agent_decl_map = {a.name: a for a in self.project.agents}
        for _ in range(ticks):
            with self._lock:
                self.tick_count += 1
                tick_now = self.tick_count

            for agent_name, instances in self.agents.items():
                decl = agent_decl_map.get(agent_name)
                if decl is None or not instances:
                    continue
                if len(instances) < 2:
                    self._run_agent_chunk(decl, instances, tick_now)
                    continue

                actual_workers = min(workers, len(instances))
                if actual_workers <= 1:
                    self._run_agent_chunk(decl, instances, tick_now)
                    continue

                if chunk_size <= 0:
                    chunk = max(1, (len(instances) + actual_workers - 1) // actual_workers)
                else:
                    chunk = max(1, chunk_size)

                chunks = [instances[i : i + chunk] for i in range(0, len(instances), chunk)]
                with ThreadPoolExecutor(max_workers=actual_workers, thread_name_prefix=f"nf-{agent_name}") as pool:
                    futures = [pool.submit(self._run_agent_chunk, decl, c, tick_now) for c in chunks]
                    for fut in futures:
                        fut.result()

    def auto_simulate(self, ticks: int) -> None:
        workers_cfg = self.config.get("threads", 1)
        try:
            workers = max(1, _as_int(workers_cfg))
        except RuntimeErrorNF:
            workers = 1
        if workers > 1:
            self.simulate_mt(ticks, workers)
        else:
            self.simulate(ticks)

    def agent_summary(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, instances in self.agents.items():
            summary: Dict[str, Any] = {"count": len(instances)}
            if instances:
                keys = {k for inst in instances for k in inst.keys() if not k.startswith("_")}
                for key in sorted(keys):
                    nums = [inst[key] for inst in instances if isinstance(inst.get(key), (int, float))]
                    if nums:
                        summary[f"avg_{key}"] = sum(nums) / len(nums)
                        summary[f"min_{key}"] = min(nums)
                        summary[f"max_{key}"] = max(nums)
            result[name] = summary
        return result

    def snapshot(self) -> Dict[str, Any]:
        metrics = {}
        for name in self.metric_exprs:
            try:
                metrics[name] = self._compute_metric(name, None)
            except Exception as exc:
                metrics[name] = f"<metric-error: {exc}>"
        with self._lock:
            channel_sizes = {k: len(v) for k, v in self.channels.items()}
            state_copy = copy.deepcopy(self.state)
            models_copy = copy.deepcopy(self.models)
            datasets_copy = copy.deepcopy(self.datasets)
            assets3d_copy = copy.deepcopy(self.assets3d)
            scenes3d_copy = copy.deepcopy(self.scenes3d)
            web_tools_copy = copy.deepcopy(self.web_tools)
            fusion_runs_copy = copy.deepcopy(self.fusion_runs)
            protein_runs_copy = copy.deepcopy(self.protein_runs)
            agents_copy = copy.deepcopy(self.agents)
            events_copy = copy.deepcopy(self.events[-200:])
            training_runs_copy = copy.deepcopy(self.training_runs)
            benchmarks_copy = copy.deepcopy(self.benchmarks)
            windows_ops_copy = copy.deepcopy(self.windows_ops)
            vui_state_copy = copy.deepcopy(self.vui_state)
            http_ops_copy = copy.deepcopy(self.http_ops)
            http_auth_presets_copy = copy.deepcopy(self.http_auth_presets)
            mock_http_servers_copy = copy.deepcopy(self.mock_http_servers)
            lang_modules_copy = copy.deepcopy(self.lang_modules)
            lang_module_runs_copy = copy.deepcopy(self.lang_module_runs)
            python_ml_state_copy = copy.deepcopy(self.python_ml_state)
            python_models_copy = copy.deepcopy(self._python_trained_models)
            sqlite_state_copy = copy.deepcopy(self.sqlite_state)
            archive_state_copy = copy.deepcopy(self.archive_state)
            resource_state_copy = copy.deepcopy(self.resource_state)
            thread_stats_copy = copy.deepcopy(self.thread_stats)
            torch_state_copy = copy.deepcopy(self.torch_state)
            process_state_copy = copy.deepcopy(self.process_state)
            wifi_state_copy = copy.deepcopy(self.wifi_state)
            npu_state_copy = copy.deepcopy(self.npu_state)
            photo_state_copy = copy.deepcopy(self.photo_state)
            graph_state_copy = copy.deepcopy(self.graph_state)
            convert_state_copy = copy.deepcopy(self.convert_state)
            iso_state_copy = copy.deepcopy(self.iso_state)
            exe_state_copy = copy.deepcopy(self.exe_state)
            github_local_state_copy = copy.deepcopy(self.github_local_state)
            unsupported_steps_copy = copy.deepcopy(self.unsupported_steps)
            tick_now = self.tick_count
        return {
            "project": self.project.name,
            "config": self.config,
            "state": state_copy,
            "metrics": metrics,
            "tick": tick_now,
            "models": models_copy,
            "datasets": datasets_copy,
            "assets3d": assets3d_copy,
            "scenes3d": scenes3d_copy,
            "web_tools": web_tools_copy,
            "fusion_runs": fusion_runs_copy,
            "protein_runs": protein_runs_copy,
            "channels": channel_sizes,
            "agents": agents_copy,
            "agent_summary": self.agent_summary(),
            "events": events_copy,
            "training_runs": training_runs_copy,
            "benchmarks": benchmarks_copy,
            "threading": thread_stats_copy,
            "pytorch": torch_state_copy,
            "python_ml": {
                **python_ml_state_copy,
                "models": python_models_copy,
            },
            "languages": {
                "modules": lang_modules_copy,
                "runs": lang_module_runs_copy,
            },
            "windows": {
                "host": self._host_info(),
                "ops": windows_ops_copy,
            },
            "vui": vui_state_copy,
            "http": {
                "ops": http_ops_copy,
                "auth_presets": http_auth_presets_copy,
                "mock_servers": mock_http_servers_copy,
            },
            "sqlite": sqlite_state_copy,
            "archives": archive_state_copy,
            "resources": resource_state_copy,
            "processes": process_state_copy,
            "wifi": wifi_state_copy,
            "npu": npu_state_copy,
            "photos": photo_state_copy,
            "graphs": graph_state_copy,
            "conversion": convert_state_copy,
            "iso": iso_state_copy,
            "executables": exe_state_copy,
            "github_local": github_local_state_copy,
            "unsupported_steps": unsupported_steps_copy,
        }

    def _resolve_output_path(self, rel_path: str) -> Path:
        p = Path(rel_path)
        out = p if p.is_absolute() else (self.out_dir / p).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def export_json(self, rel_path: str) -> Path:
        out = self._resolve_output_path(rel_path)
        out.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return out

    def export_html(self, rel_path: str) -> Path:
        out = self._resolve_output_path(rel_path)
        snap = self.snapshot()
        ui_panels = []
        ui_meta = {"templates": [], "theme": None}
        if self.project.ui:
            ui_meta["templates"] = list(self.project.ui.templates)
            ui_meta["theme"] = self.project.ui.theme
            for panel in self.project.ui.panels:
                widgets = []
                for w in panel.widgets:
                    if isinstance(w, (TextWidget, StatWidget, ProgressWidget, JsonWidget, Scene3DWidget)):
                        try:
                            value = self.eval_expr(w.expr, None)
                        except Exception as exc:
                            value = f"<expr-error: {exc}>"
                        widget_type = (
                            "text"
                            if isinstance(w, TextWidget)
                            else "stat"
                            if isinstance(w, StatWidget)
                            else "progress"
                            if isinstance(w, ProgressWidget)
                            else "json"
                            if isinstance(w, JsonWidget)
                            else "scene3d"
                        )
                        widgets.append({"type": widget_type, "label": w.label, "value": value})
                    else:
                        widgets.append({"type": "button", "label": w.label, "action": w.action})
                ui_panels.append({"title": panel.title, "widgets": widgets})

        html = self._build_html_preview(snap, ui_panels, ui_meta)
        out.write_text(html, encoding="utf-8")
        return out

    def _build_html_preview(self, snap: Dict[str, Any], ui_panels: List[Dict[str, Any]], ui_meta: Dict[str, Any]) -> str:
        snap_json = json.dumps(snap, indent=2)
        ui_json = json.dumps(ui_panels, indent=2)
        ui_meta_json = json.dumps(ui_meta, indent=2)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self.project.name} - NexusFlow Export</title>
  <style>
    :root {{
      --bg:#0f172a; --panel:#111827; --card:#1f2937; --line:#334155;
      --text:#e5eefc; --muted:#94a3b8; --accent:#67e8f9; --accent2:#86efac;
      --glow:rgba(103,232,249,.18);
      --font: 'Segoe UI', system-ui, sans-serif;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:var(--font); color:var(--text);
      background:
        radial-gradient(1000px 420px at 0% -10%, var(--glow), transparent 60%),
        radial-gradient(900px 380px at 100% 0%, rgba(134,239,172,.12), transparent 55%),
        linear-gradient(180deg,#020617,var(--bg)); }}
    body.template-ops_console {{ --font: Consolas, 'Cascadia Code', 'Segoe UI', monospace; }}
    body.template-creator_studio {{ --line:#3b2f53; --card:#1d1630; --accent:#f59e0b; --accent2:#fb7185; --glow:rgba(245,158,11,.16); }}
    body.theme-amber {{ --accent:#f59e0b; --accent2:#fbbf24; --glow:rgba(245,158,11,.16); }}
    body.theme-ocean {{ --accent:#22d3ee; --accent2:#38bdf8; --glow:rgba(34,211,238,.18); }}
    body.theme-lime {{ --accent:#86efac; --accent2:#4ade80; --glow:rgba(134,239,172,.16); }}
    .wrap {{ max-width:1180px; margin:0 auto; padding:22px; }}
    .hero {{ border:1px solid var(--line); background:rgba(255,255,255,.03); border-radius:16px; padding:18px; position:relative; overflow:hidden; }}
    .hero::after {{ content:''; position:absolute; inset:-40% auto auto 60%; width:220px; height:220px; border-radius:999px; background:radial-gradient(circle, rgba(255,255,255,.10), transparent 70%); pointer-events:none; }}
    .hero h1 {{ margin:0 0 8px; font-size:28px; }}
    .muted {{ color:var(--muted); }}
    .badges {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
    .badge {{ border:1px solid rgba(255,255,255,.12); color:var(--accent); background:rgba(255,255,255,.02); border-radius:999px; padding:3px 9px; font-size:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; margin-top:14px; }}
    .card {{ border:1px solid var(--line); border-radius:14px; background:rgba(17,24,39,.75); padding:14px; }}
    .card h2 {{ margin:0 0 10px; font-size:15px; letter-spacing:.02em; }}
    .kv {{ display:grid; grid-template-columns:1fr auto; gap:8px; }}
    .kv > div {{ padding:6px 0; border-bottom:1px dashed rgba(148,163,184,.15); }}
    pre {{ margin:0; font-size:12px; line-height:1.35; background:rgba(2,6,23,.55); border:1px solid rgba(255,255,255,.06); border-radius:10px; padding:10px; overflow:auto; }}
    .pill {{ display:inline-block; padding:4px 8px; border-radius:10px; border:1px solid rgba(255,255,255,.12); margin:4px 4px 0 0; font-size:12px; }}
    .ui-panel {{ margin-bottom:12px; border:1px dashed rgba(148,163,184,.18); border-radius:10px; padding:10px; }}
    .ui-panel-title {{ margin-bottom:8px; font-weight:700; letter-spacing:.02em; }}
    .ui-widget {{ margin-bottom:8px; }}
    .ui-stat {{ border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.02); border-radius:10px; padding:10px; }}
    .ui-stat .label {{ color:var(--muted); font-size:12px; }}
    .ui-stat .value {{ color:var(--accent2); font-weight:700; font-size:18px; margin-top:2px; }}
    .progress-row {{ display:grid; gap:5px; }}
    .progress-track {{ height:8px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; }}
    .progress-fill {{ height:100%; background:linear-gradient(90deg,var(--accent),var(--accent2)); width:0%; }}
    .scene-card {{ border:1px solid rgba(255,255,255,.08); border-radius:10px; padding:10px; background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01)); }}
    .scene-mini {{ margin-top:8px; height:76px; border-radius:8px; border:1px solid rgba(255,255,255,.06); background:
      radial-gradient(circle at 20% 30%, rgba(103,232,249,.16), transparent 42%),
      radial-gradient(circle at 75% 55%, rgba(134,239,172,.12), transparent 45%),
      #060c18; position:relative; overflow:hidden; }}
    .scene-dot {{ position:absolute; width:8px; height:8px; border-radius:999px; box-shadow:0 0 10px rgba(255,255,255,.25); }}
    .ui-json details {{ border:1px solid rgba(255,255,255,.06); border-radius:10px; padding:6px 8px; }}
    .ui-json summary {{ cursor:pointer; color:var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{self.project.name}</h1>
      <div class="muted">Static single-file dashboard export for rapid prototype inspection (with GUI templates + 3D scene metadata widgets).</div>
      <div class="badges">
        <span class="badge">Tick {snap["tick"]}</span>
        <span class="badge">Agent types {len(snap["agent_summary"])}</span>
        <span class="badge">Events {len(snap["events"])}</span>
        <span class="badge">Training runs {len(snap["training_runs"])}</span>
        <span class="badge">3D assets {len(snap.get("assets3d", {}))}</span>
        <span class="badge">3D scenes {len(snap.get("scenes3d", {}))}</span>
      </div>
    </section>
    <section class="grid">
      <div class="card"><h2>World / Metrics</h2><div class="kv" id="world"></div></div>
      <div class="card"><h2>Agent Summary</h2><pre id="agents"></pre></div>
      <div class="card"><h2>Models / Datasets / Training</h2><pre id="train"></pre></div>
      <div class="card"><h2>UI Panels</h2><div id="panels"></div></div>
      <div class="card"><h2>3D Assets / Scenes</h2><pre id="scene_meta"></pre></div>
      <div class="card" style="grid-column:1 / -1;"><h2>Snapshot JSON</h2><pre id="json"></pre></div>
    </section>
  </div>
  <script>
    const snapshot = {snap_json};
    const uiPanels = {ui_json};
    const uiMeta = {ui_meta_json};

    const themes = {{
      amber: {{ '--accent':'#f59e0b', '--accent2':'#fbbf24', '--glow':'rgba(245,158,11,.16)' }},
      ocean: {{ '--accent':'#22d3ee', '--accent2':'#38bdf8', '--glow':'rgba(34,211,238,.18)' }},
      lime: {{ '--accent':'#86efac', '--accent2':'#4ade80', '--glow':'rgba(134,239,172,.16)' }},
    }};
    for (const t of (uiMeta.templates || [])) {{
      document.body.classList.add(`template-${{String(t).replace(/[^a-z0-9_\\-]/gi,'_')}}`);
    }}
    if (uiMeta.theme) {{
      document.body.classList.add(`theme-${{String(uiMeta.theme).replace(/[^a-z0-9_\\-]/gi,'_')}}`);
      const palette = themes[String(uiMeta.theme)];
      if (palette) {{
        for (const [k,v] of Object.entries(palette)) document.documentElement.style.setProperty(k, v);
      }}
    }}

    const world = document.getElementById('world');
    function textOf(v) {{
      if (typeof v === 'string') return v;
      if (v === null || v === undefined) return String(v);
      if (typeof v === 'object') return JSON.stringify(v);
      return String(v);
    }}
    for (const [k,v] of Object.entries(snapshot.state || {{}})) {{
      const a = document.createElement('div'); a.className='muted'; a.textContent = k;
      const b = document.createElement('div'); b.textContent = textOf(v);
      world.append(a,b);
    }}
    for (const [k,v] of Object.entries(snapshot.metrics || {{}})) {{
      const a = document.createElement('div'); a.className='muted'; a.textContent = `metric:${{k}}`;
      const b = document.createElement('div'); b.textContent = textOf(v);
      world.append(a,b);
    }}
    for (const [k,v] of Object.entries(snapshot.channels || {{}})) {{
      const a = document.createElement('div'); a.className='muted'; a.textContent = `channel:${{k}}`;
      const b = document.createElement('div'); b.textContent = `size=${{v}}`;
      world.append(a,b);
    }}
    document.getElementById('agents').textContent = JSON.stringify(snapshot.agent_summary, null, 2);
    document.getElementById('train').textContent = JSON.stringify({{
      models: snapshot.models, datasets: snapshot.datasets,
      training_runs: snapshot.training_runs, threading: snapshot.threading,
      pytorch: snapshot.pytorch, benchmarks: snapshot.benchmarks,
      windows: snapshot.windows, http: snapshot.http,
      web_tools: snapshot.web_tools, fusion_runs: snapshot.fusion_runs, protein_runs: snapshot.protein_runs,
      unsupported_steps: snapshot.unsupported_steps
    }}, null, 2);
    document.getElementById('scene_meta').textContent = JSON.stringify({{
      ui_meta: uiMeta,
      assets3d: snapshot.assets3d || {{}},
      scenes3d: snapshot.scenes3d || {{}}
    }}, null, 2);
    document.getElementById('json').textContent = JSON.stringify(snapshot, null, 2);

    function appendKVRow(parent, label, value) {{
      const row = document.createElement('div'); row.className='kv ui-widget';
      const a = document.createElement('div'); a.className='muted'; a.textContent = label;
      const b = document.createElement('div'); b.textContent = textOf(value);
      row.append(a,b);
      parent.appendChild(row);
    }}

    function normalizedPercent(v) {{
      const n = Number(v);
      if (!Number.isFinite(n)) return 0;
      if (n <= 1 && n >= 0) return Math.max(0, Math.min(100, n * 100));
      return Math.max(0, Math.min(100, n));
    }}

    function renderSceneMini(container, scene) {{
      const box = document.createElement('div');
      box.className = 'scene-mini';
      const nodes = Array.isArray(scene?.nodes) ? scene.nodes : [];
      nodes.slice(0, 12).forEach((n, idx) => {{
        const dot = document.createElement('div');
        dot.className = 'scene-dot';
        const pos = n?.transform?.position || [0,0,0];
        const x = 50 + (Number(pos[0] || 0) * 11) + ((idx * 23) % 140);
        const y = 34 - (Number(pos[1] || 0) * 9) + ((idx * 17) % 24);
        dot.style.left = `${{Math.max(4, Math.min(260, x))}}px`;
        dot.style.top = `${{Math.max(4, Math.min(58, y))}}px`;
        dot.style.background = ['#22d3ee','#f59e0b','#86efac','#fb7185'][idx % 4];
        box.appendChild(dot);
      }});
      container.appendChild(box);
    }}

    const panels = document.getElementById('panels');
    if (!uiPanels.length) {{
      const p = document.createElement('div'); p.className='muted'; p.textContent = 'No ui block defined.'; panels.appendChild(p);
    }}
    for (const panel of uiPanels) {{
      const card = document.createElement('div');
      card.className = 'ui-panel';
      const title = document.createElement('div');
      title.className = 'ui-panel-title';
      title.textContent = panel.title;
      card.appendChild(title);

      for (const w of panel.widgets || []) {{
        if (w.type === 'text') {{
          appendKVRow(card, w.label, w.value);
          continue;
        }}
        if (w.type === 'button') {{
          const p = document.createElement('span'); p.className='pill'; p.textContent = `${{w.label}} -> ${{w.action}}`; card.appendChild(p);
          continue;
        }}
        if (w.type === 'stat') {{
          const el = document.createElement('div'); el.className = 'ui-stat ui-widget';
          const l = document.createElement('div'); l.className = 'label'; l.textContent = w.label;
          const v = document.createElement('div'); v.className = 'value'; v.textContent = textOf(w.value);
          el.append(l, v); card.appendChild(el);
          continue;
        }}
        if (w.type === 'progress') {{
          const wrap = document.createElement('div'); wrap.className = 'progress-row ui-widget';
          const head = document.createElement('div'); head.className = 'kv';
          const a = document.createElement('div'); a.className = 'muted'; a.textContent = w.label;
          const b = document.createElement('div');
          const pct = normalizedPercent(w.value);
          b.textContent = `${{pct.toFixed(1)}}%`;
          head.append(a,b);
          const track = document.createElement('div'); track.className='progress-track';
          const fill = document.createElement('div'); fill.className='progress-fill'; fill.style.width = `${{pct}}%`;
          track.appendChild(fill);
          wrap.append(head, track);
          card.appendChild(wrap);
          continue;
        }}
        if (w.type === 'json') {{
          const wrap = document.createElement('div'); wrap.className = 'ui-json ui-widget';
          const details = document.createElement('details');
          const summary = document.createElement('summary'); summary.textContent = w.label;
          const pre = document.createElement('pre'); pre.textContent = JSON.stringify(w.value, null, 2);
          details.append(summary, pre); wrap.appendChild(details); card.appendChild(wrap);
          continue;
        }}
        if (w.type === 'scene3d') {{
          const sceneName = String(w.value);
          const scene = (snapshot.scenes3d || {{}})[sceneName];
          const wrap = document.createElement('div'); wrap.className = 'scene-card ui-widget';
          const head = document.createElement('div'); head.className = 'kv';
          const a = document.createElement('div'); a.className = 'muted'; a.textContent = w.label;
          const b = document.createElement('div'); b.textContent = sceneName;
          head.append(a,b); wrap.appendChild(head);
          if (scene) {{
            const info = document.createElement('div');
            info.className = 'muted';
            const nodes = Array.isArray(scene.nodes) ? scene.nodes.length : 0;
            const lights = Array.isArray(scene.lights) ? scene.lights.length : 0;
            info.textContent = `template=${{scene.template || 'stage'}} | nodes=${{nodes}} | lights=${{lights}}`;
            wrap.appendChild(info);
            renderSceneMini(wrap, scene);
          }} else {{
            const miss = document.createElement('div'); miss.className='muted'; miss.textContent = 'Scene not found in snapshot.scenes3d';
            wrap.appendChild(miss);
          }}
          card.appendChild(wrap);
          continue;
        }}
        appendKVRow(card, w.label || w.type || 'widget', w.value);
      }}
      panels.appendChild(card);
    }}
  </script>
</body>
</html>
"""

    def _find_pipeline(self, name: Optional[str]) -> PipelineDecl:
        pipeline: Optional[PipelineDecl] = None
        if name is None and self.project.pipelines:
            pipeline = self.project.pipelines[0]
        elif name is not None:
            for p in self.project.pipelines:
                if p.name == name:
                    pipeline = p
                    break
        if pipeline is None:
            raise RuntimeErrorNF("Pipeline not found")
        return pipeline

    def _trace_call(self, call: Call) -> Dict[str, Any]:
        if call.name == "parallel":
            nested = []
            for arg in call.args:
                if isinstance(arg, Call):
                    nested.append(self._trace_call(arg))
                else:
                    nested.append(expr_to_json(arg))
            return {"step": "parallel", "substeps": nested}
        if call.name == "repeat":
            if len(call.args) != 2 or not isinstance(call.args[0], Call):
                return {"step": "repeat", "error": "repeat(stepCall, count) expected"}
            return {
                "step": "repeat",
                "count": expr_to_json(call.args[1]),
                "substep": self._trace_call(call.args[0]),
            }
        if call.name == "when":
            if len(call.args) != 2 or not isinstance(call.args[1], Call):
                return {"step": "when", "error": "when(condition, stepCall) expected"}
            return {
                "step": "when",
                "condition": expr_to_json(call.args[0]),
                "substep": self._trace_call(call.args[1]),
            }
        if call.name == "try":
            if len(call.args) not in {1, 2} or not isinstance(call.args[0], Call):
                return {"step": "try", "error": "try(stepCall[, fallbackStep]) expected"}
            payload: Dict[str, Any] = {"step": "try", "substep": self._trace_call(call.args[0])}
            if len(call.args) == 2:
                if isinstance(call.args[1], Call):
                    payload["fallback"] = self._trace_call(call.args[1])
                else:
                    payload["fallback"] = expr_to_json(call.args[1])
            return payload
        if call.name == "retry":
            if len(call.args) not in {2, 3} or not isinstance(call.args[0], Call):
                return {"step": "retry", "error": "retry(stepCall, attempts[, delay_ms]) expected"}
            out: Dict[str, Any] = {
                "step": "retry",
                "attempts": expr_to_json(call.args[1]),
                "substep": self._trace_call(call.args[0]),
            }
            if len(call.args) == 3:
                out["delay_ms"] = expr_to_json(call.args[2])
            return out
        if call.name == "bench":
            if not call.args:
                return {"step": "bench", "error": "bench(stepCall[, label]) expected"}
            label = None
            subcall = None
            for arg in call.args:
                if isinstance(arg, Call):
                    subcall = arg
                else:
                    if isinstance(arg, Literal):
                        label = arg.value
                    else:
                        label = expr_to_json(arg)
            payload: Dict[str, Any] = {"step": "bench"}
            if label is not None:
                payload["label"] = label
            if subcall is not None:
                payload["substep"] = self._trace_call(subcall)
            return payload
        return {"step": call.name, "args": [expr_to_json(a) for a in call.args]}

    def _exec_pipeline_call_ast(self, call: Call) -> None:
        if call.name == "parallel":
            subcalls = [arg for arg in call.args if isinstance(arg, Call)]
            if len(subcalls) != len(call.args):
                raise RuntimeErrorNF("parallel(...) only accepts pipeline step calls")
            self._exec_parallel_substeps(subcalls)
            return
        if call.name == "repeat":
            if len(call.args) != 2 or not isinstance(call.args[0], Call):
                raise RuntimeErrorNF("repeat(stepCall, count) expects a nested step call and repeat count")
            n = max(0, _as_int(self.eval_expr(call.args[1], None)))
            for _ in range(n):
                self._exec_pipeline_call_ast(call.args[0])
            return
        if call.name == "when":
            if len(call.args) != 2 or not isinstance(call.args[1], Call):
                raise RuntimeErrorNF("when(condition, stepCall) expects a condition and nested step call")
            if bool(self.eval_expr(call.args[0], None)):
                self._exec_pipeline_call_ast(call.args[1])
            return
        if call.name == "try":
            if len(call.args) not in {1, 2} or not isinstance(call.args[0], Call):
                raise RuntimeErrorNF("try(stepCall[, fallbackStep]) expects nested step call")
            try:
                self._exec_pipeline_call_ast(call.args[0])
                with self._lock:
                    tick_now = self.tick_count
                self._append_event({"tick": tick_now, "event": "try_success", "step": call.args[0].name})
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    tick_now = self.tick_count
                self._append_event({"tick": tick_now, "event": "try_failure", "step": call.args[0].name, "error": str(exc)})
                if len(call.args) == 2 and isinstance(call.args[1], Call):
                    self._exec_pipeline_call_ast(call.args[1])
                else:
                    return
            return
        if call.name == "retry":
            if len(call.args) not in {2, 3} or not isinstance(call.args[0], Call):
                raise RuntimeErrorNF("retry(stepCall, attempts[, delay_ms]) expects nested step call + attempts")
            attempts = max(1, _as_int(self.eval_expr(call.args[1], None)))
            delay_ms = max(0, _as_int(self.eval_expr(call.args[2], None))) if len(call.args) == 3 else 0
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    self._exec_pipeline_call_ast(call.args[0])
                    with self._lock:
                        tick_now = self.tick_count
                    self._append_event({"tick": tick_now, "event": "retry_success", "attempt": attempt, "step": call.args[0].name})
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    with self._lock:
                        tick_now = self.tick_count
                    self._append_event(
                        {
                            "tick": tick_now,
                            "event": "retry_failure",
                            "attempt": attempt,
                            "step": call.args[0].name,
                            "error": str(exc),
                        }
                    )
                    if attempt < attempts and delay_ms > 0:
                        time.sleep(delay_ms / 1000.0)
            assert last_exc is not None
            raise last_exc
        if call.name == "bench":
            if not call.args:
                raise RuntimeErrorNF("bench(stepCall[, label]) expects at least one arg")
            label: Optional[str] = None
            subcall: Optional[Call] = None
            for arg in call.args:
                if isinstance(arg, Call):
                    subcall = arg
                else:
                    label = str(self.eval_expr(arg, None))
            if subcall is None:
                raise RuntimeErrorNF("bench(...) requires a nested step call argument")
            start = time.time()
            self._exec_pipeline_call_ast(subcall)
            elapsed_ms = round((time.time() - start) * 1000, 2)
            with self._lock:
                tick_now = self.tick_count
            bench = {
                "label": label or subcall.name,
                "step": subcall.name,
                "duration_ms": elapsed_ms,
                "tick": tick_now,
            }
            with self._lock:
                self.benchmarks.append(bench)
                self.benchmarks = self.benchmarks[-50:]
            self._append_event({"tick": tick_now, "event": f"bench:{bench['label']}", "duration_ms": elapsed_ms})
            return
        args = [self.eval_expr(a, None) for a in call.args]
        self.exec_pipeline_step(call.name, args)

    def _exec_parallel_substeps(self, calls: List[Call]) -> None:
        if not calls:
            return
        workers_cfg = self.config.get("pipeline_threads", self.config.get("threads", len(calls)))
        try:
            workers = max(1, min(len(calls), _as_int(workers_cfg)))
        except RuntimeErrorNF:
            workers = min(len(calls), 4)
        with self._lock:
            max_pipe_workers = (self.resource_state.get("limits", {}) or {}).get("max_pipeline_workers")
        if isinstance(max_pipe_workers, int):
            workers = max(1, min(workers, int(max_pipe_workers), len(calls)))

        self.thread_stats["pipeline_parallel_calls"] += 1
        self.thread_stats["max_workers_seen"] = max(self.thread_stats["max_workers_seen"], workers)
        self._record_thread_event("pipeline_parallel", workers=workers, substeps=[c.name for c in calls])

        if workers <= 1 or len(calls) == 1:
            for c in calls:
                self._exec_pipeline_call_ast(c)
            return

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nf-pipe") as pool:
            futures = [pool.submit(self._exec_pipeline_call_ast, c) for c in calls]
            for fut in futures:
                fut.result()

    def run_pipeline(self, name: Optional[str] = None) -> Dict[str, Any]:
        pipeline = self._find_pipeline(name)
        if pipeline.name in self._pipeline_call_stack:
            chain = " -> ".join(self._pipeline_call_stack + [pipeline.name])
            raise RuntimeErrorNF(f"Pipeline recursion detected: {chain}")
        self._pipeline_call_stack.append(pipeline.name)
        executed: List[Dict[str, Any]] = []
        try:
            for step in pipeline.steps:
                executed.append(self._trace_call(step))
                self._exec_pipeline_call_ast(step)
            return {"pipeline": pipeline.name, "executed": executed}
        finally:
            if self._pipeline_call_stack and self._pipeline_call_stack[-1] == pipeline.name:
                self._pipeline_call_stack.pop()
            elif pipeline.name in self._pipeline_call_stack:
                self._pipeline_call_stack.remove(pipeline.name)

    def _record_training_run(self, run: Dict[str, Any]) -> None:
        with self._lock:
            self.training_runs.append(run)

    def _record_unsupported_step(self, step_name: str, args: List[Any], reason: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"step": step_name, "args": args}
        if reason:
            payload["reason"] = reason
        with self._lock:
            self.unsupported_steps.append(payload)
            tick_now = self.tick_count
        event = f"unsupported_step:{step_name}"
        if reason:
            event = f"{event}:{reason}"
        self._append_event({"tick": tick_now, "event": event})

    def _record_stub_train(self, model: str, dataset: str, epochs: int) -> None:
        with self._lock:
            tick_now = self.tick_count
        run = {
            "backend": "stub",
            "model": model,
            "dataset": dataset,
            "epochs": epochs,
            "tick_at_start": tick_now,
        }
        self._record_training_run(run)
        self._append_event({"tick": tick_now, "event": f"train:{model}:{dataset}"})

    def _python_model_rng(self, model_name: str, dataset_name: str) -> random.Random:
        seed_base = self._base_seed if self._base_seed is not None else 0
        h = hashlib.sha256(f"{model_name}|{dataset_name}|{seed_base}".encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big") & 0xFFFFFFFF
        return random.Random(seed)

    def _python_make_data(self, dataset_name: str, model_name: str) -> Dict[str, Any]:
        ds = self.datasets.get(dataset_name)
        if ds is None:
            raise RuntimeErrorNF(f"Unknown dataset: {dataset_name}")
        model_spec = self.models.get(model_name, {})
        rng = self._python_model_rng(model_name, dataset_name)
        samples = max(8, _as_int(ds.get("samples", 256)))
        inputs = max(1, _as_int(ds.get("inputs", model_spec.get("inputs", model_spec.get("input_dim", 4)))))
        outputs = max(1, _as_int(ds.get("outputs", model_spec.get("outputs", model_spec.get("output_dim", 1)))))
        task = str(ds.get("task", "regression")).lower()
        kind = str(model_spec.get("kind", "")).lower()
        if kind == "kmeans" or task in {"cluster", "clustering", "unsupervised"}:
            task = "clustering"
        elif kind == "logistic_regression" and task == "regression":
            task = "binary_classification"

        x: List[List[float]] = []
        y: Optional[List[Any]] = None

        if task == "clustering":
            clusters = max(2, _as_int(ds.get("clusters", model_spec.get("clusters", 3))))
            centers: List[List[float]] = [[rng.uniform(-3.0, 3.0) for _ in range(inputs)] for _ in range(clusters)]
            labels: List[int] = []
            for i in range(samples):
                cidx = i % clusters
                labels.append(cidx)
                row = [centers[cidx][j] + rng.gauss(0.0, 0.45 + 0.08 * (j % 3)) for j in range(inputs)]
                x.append(row)
            y = labels
        elif task in {"classification", "binary", "binary_classification"}:
            task = "binary_classification"
            w_true = [rng.uniform(-1.5, 1.5) for _ in range(inputs)]
            bias = rng.uniform(-0.6, 0.6)
            labels = []
            for _ in range(samples):
                row = [rng.gauss(0.0, 1.0) for _ in range(inputs)]
                margin = sum(a * b for a, b in zip(row, w_true)) + bias + rng.gauss(0.0, 0.25)
                labels.append(1 if margin >= 0 else 0)
                x.append(row)
            y = labels
        else:
            task = "regression"
            w_true = [[rng.uniform(-1.2, 1.2) for _ in range(inputs)] for _ in range(outputs)]
            bias_true = [rng.uniform(-0.4, 0.4) for _ in range(outputs)]
            targets: List[Any] = []
            for _ in range(samples):
                row = [rng.gauss(0.0, 1.0) for _ in range(inputs)]
                pred = []
                for o in range(outputs):
                    pred.append(sum(row[j] * w_true[o][j] for j in range(inputs)) + bias_true[o] + rng.gauss(0.0, 0.08))
                targets.append(pred[0] if outputs == 1 else pred)
                x.append(row)
            y = targets

        return {
            "task": task,
            "samples": samples,
            "inputs": inputs,
            "outputs": outputs,
            "x": x,
            "y": y,
        }

    def _python_train(self, model_name: str, dataset_name: str, epochs: int, lr: float = 0.03) -> None:
        spec = self.models.get(model_name)
        if spec is None:
            raise RuntimeErrorNF(f"Unknown model: {model_name}")
        kind = str(spec.get("kind", "linear_regression")).lower()
        data = self._python_make_data(dataset_name, model_name)
        x = data["x"]
        y = data["y"]
        samples = int(data["samples"])
        inputs = int(data["inputs"])
        task = str(data["task"])
        epochs_n = max(1, int(epochs))
        lr_f = float(lr)
        start = time.time()

        artifact: Dict[str, Any]
        run_extra: Dict[str, Any]
        loss_history: List[float] = []

        if kind in {"linear_regression", "ridge_regression"}:
            outputs = max(1, int(data["outputs"]))
            if not isinstance(y, list):
                raise RuntimeErrorNF("Regression dataset missing targets")
            targets: List[List[float]] = []
            for t in y:
                if isinstance(t, list):
                    targets.append([float(v) for v in t[:outputs]] + [0.0] * max(0, outputs - len(t)))
                else:
                    targets.append([float(t)])
            weights = [[0.0 for _ in range(inputs)] for _ in range(outputs)]
            bias = [0.0 for _ in range(outputs)]
            l2 = float(spec.get("l2", 0.0 if kind == "linear_regression" else 0.01))
            for _ in range(epochs_n):
                grad_w = [[0.0 for _ in range(inputs)] for _ in range(outputs)]
                grad_b = [0.0 for _ in range(outputs)]
                mse_sum = 0.0
                for i in range(samples):
                    row = x[i]
                    tgt = targets[i]
                    for o in range(outputs):
                        pred = bias[o] + sum(weights[o][j] * row[j] for j in range(inputs))
                        err = pred - tgt[o]
                        mse_sum += err * err
                        grad_b[o] += (2.0 / samples) * err
                        for j in range(inputs):
                            grad_w[o][j] += (2.0 / samples) * err * row[j]
                for o in range(outputs):
                    for j in range(inputs):
                        grad_w[o][j] += (2.0 * l2 / samples) * weights[o][j]
                        weights[o][j] -= lr_f * grad_w[o][j]
                    bias[o] -= lr_f * grad_b[o]
                loss_history.append(mse_sum / max(1, samples * outputs))
            preds = []
            for row in x:
                vals = [bias[o] + sum(weights[o][j] * row[j] for j in range(inputs)) for o in range(outputs)]
                preds.append(vals[0] if outputs == 1 else vals)
            mse = sum(
                (
                    (preds[i] - (targets[i][0] if outputs == 1 else 0.0)) ** 2
                    if outputs == 1
                    else sum((preds[i][o] - targets[i][o]) ** 2 for o in range(outputs)) / outputs
                )
                for i in range(samples)
            ) / max(1, samples)
            artifact = {
                "backend": "python_native",
                "kind": kind,
                "task": task,
                "inputs": inputs,
                "outputs": outputs,
                "weights": [[round(v, 8) for v in row] for row in weights],
                "bias": [round(v, 8) for v in bias],
                "l2": l2,
            }
            run_extra = {
                "metric": "mse",
                "final_metric": round(float(mse), 6),
                "loss_history": [round(float(v), 6) for v in loss_history[-50:]],
            }
        elif kind in {"logistic_regression", "perceptron"}:
            if not isinstance(y, list):
                raise RuntimeErrorNF("Classification dataset missing labels")
            labels = [1 if _as_int(v) else 0 for v in y]
            weights = [0.0 for _ in range(inputs)]
            bias = 0.0
            l2 = float(spec.get("l2", 0.0))
            for _ in range(epochs_n):
                grad_w = [0.0 for _ in range(inputs)]
                grad_b = 0.0
                ce = 0.0
                for i in range(samples):
                    row = x[i]
                    z = bias + sum(weights[j] * row[j] for j in range(inputs))
                    p = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))
                    t = float(labels[i])
                    err = p - t
                    ce += -(t * math.log(max(p, 1e-9)) + (1.0 - t) * math.log(max(1.0 - p, 1e-9)))
                    grad_b += err / samples
                    for j in range(inputs):
                        grad_w[j] += (err * row[j]) / samples
                for j in range(inputs):
                    grad_w[j] += (2.0 * l2 / samples) * weights[j]
                    weights[j] -= lr_f * grad_w[j]
                bias -= lr_f * grad_b
                loss_history.append(ce / max(1, samples))
            probs = []
            preds = []
            for row in x:
                z = bias + sum(weights[j] * row[j] for j in range(inputs))
                p = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))
                probs.append(p)
                preds.append(1 if p >= 0.5 else 0)
            acc = sum(1 for a, b in zip(preds, labels) if a == b) / max(1, samples)
            artifact = {
                "backend": "python_native",
                "kind": kind,
                "task": "binary_classification",
                "inputs": inputs,
                "weights": [round(v, 8) for v in weights],
                "bias": round(bias, 8),
                "l2": l2,
            }
            run_extra = {
                "metric": "accuracy",
                "final_metric": round(float(acc), 6),
                "final_loss": round(float(loss_history[-1]), 6) if loss_history else None,
                "loss_history": [round(float(v), 6) for v in loss_history[-50:]],
            }
        elif kind == "kmeans":
            clusters = max(2, _as_int(spec.get("clusters", self.datasets.get(dataset_name, {}).get("clusters", 3))))
            max_iters = epochs_n
            tol = float(spec.get("tol", 1e-4))
            if samples < clusters:
                raise RuntimeErrorNF("kmeans requires samples >= clusters")
            init_idx = list(range(samples))
            rng = self._python_model_rng(model_name, dataset_name + ":kmeans")
            rng.shuffle(init_idx)
            centers = [list(x[init_idx[i % samples]]) for i in range(clusters)]
            assignments = [0 for _ in range(samples)]
            for _ in range(max_iters):
                changed = 0
                inertia = 0.0
                for i, row in enumerate(x):
                    best_k = 0
                    best_d = float("inf")
                    for k in range(clusters):
                        d = sum((row[j] - centers[k][j]) ** 2 for j in range(inputs))
                        if d < best_d:
                            best_d = d
                            best_k = k
                    inertia += best_d
                    if assignments[i] != best_k:
                        assignments[i] = best_k
                        changed += 1
                new_centers = [[0.0 for _ in range(inputs)] for _ in range(clusters)]
                counts = [0 for _ in range(clusters)]
                for i, row in enumerate(x):
                    k = assignments[i]
                    counts[k] += 1
                    for j in range(inputs):
                        new_centers[k][j] += row[j]
                for k in range(clusters):
                    if counts[k] == 0:
                        new_centers[k] = list(x[rng.randrange(samples)])
                        counts[k] = 1
                    else:
                        new_centers[k] = [v / counts[k] for v in new_centers[k]]
                shift = sum(
                    math.sqrt(sum((centers[k][j] - new_centers[k][j]) ** 2 for j in range(inputs)))
                    for k in range(clusters)
                )
                centers = new_centers
                loss_history.append(inertia / max(1, samples))
                if changed == 0 or shift <= tol:
                    break
            counts = [0 for _ in range(clusters)]
            for a in assignments:
                counts[a] += 1
            artifact = {
                "backend": "python_native",
                "kind": "kmeans",
                "task": "clustering",
                "inputs": inputs,
                "clusters": clusters,
                "centers": [[round(v, 8) for v in c] for c in centers],
                "cluster_counts": counts,
            }
            run_extra = {
                "metric": "inertia",
                "final_metric": round(float(loss_history[-1]), 6) if loss_history else None,
                "iterations": len(loss_history),
                "loss_history": [round(float(v), 6) for v in loss_history[-50:]],
            }
        else:
            raise RuntimeErrorNF(f"Python backend unsupported model kind: {kind}")

        duration_ms = round((time.time() - start) * 1000, 2)
        with self._lock:
            tick_now = self.tick_count
            self._python_trained_models[model_name] = {
                "model": model_name,
                "dataset": dataset_name,
                "spec": copy.deepcopy(spec),
                "artifact": copy.deepcopy(artifact),
                "trained_at_tick": tick_now,
                "trained_at": self._now_iso(),
                "metrics": {
                    "samples": samples,
                    "inputs": inputs,
                    **{k: v for k, v in run_extra.items() if k not in {"loss_history"}},
                },
            }
            self.python_ml_state["trained_models"] = sorted(self._python_trained_models.keys())
        run = {
            "backend": "python_native",
            "model": model_name,
            "dataset": dataset_name,
            "kind": kind,
            "epochs": epochs_n,
            "lr": lr_f,
            "samples": samples,
            "inputs": inputs,
            "duration_ms": duration_ms,
            "tick_at_start": tick_now,
            **run_extra,
        }
        self._record_training_run(run)
        self._append_event({"tick": tick_now, "event": f"python_train:{model_name}:{dataset_name}", "kind": kind})

    def _python_export(self, model_name: str, rel_path: str) -> None:
        with self._lock:
            model_rec = copy.deepcopy(self._python_trained_models.get(model_name))
            tick_now = self.tick_count
        if not isinstance(model_rec, dict):
            raise RuntimeErrorNF(f"Python model not trained: {model_name}")
        out = self._resolve_output_path(rel_path)
        payload = {
            "project": self.project.name,
            "tick": tick_now,
            "python_model": model_rec,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._lock:
            self.python_ml_state["exports"].append({"model": model_name, "path": str(out)})
            self.python_ml_state["exports"] = self.python_ml_state["exports"][-20:]
        self._append_event({"tick": tick_now, "event": f"python_export:{model_name}"})

    def _torch_activation(self, name: str):
        key = (name or "relu").lower()
        if not TORCH_AVAILABLE:
            raise RuntimeErrorNF("PyTorch is not available")
        mapping = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "elu": nn.ELU,
        }
        return mapping.get(key, nn.ReLU)

    def _torch_build_model(self, model_name: str):
        if not TORCH_AVAILABLE:
            raise RuntimeErrorNF("PyTorch is not available")
        spec = self.models.get(model_name)
        if spec is None:
            raise RuntimeErrorNF(f"Unknown model: {model_name}")
        kind = str(spec.get("kind", "mlp")).lower()
        if kind != "mlp":
            raise RuntimeErrorNF(f"PyTorch backend currently supports kind='mlp' only (got {kind})")
        inputs = _as_int(spec.get("inputs", spec.get("input_dim", 8)))
        outputs = _as_int(spec.get("outputs", spec.get("output_dim", 1)))
        hidden_cfg = spec.get("hidden", [64, 64])
        if isinstance(hidden_cfg, list):
            hidden_layers = [_as_int(v) for v in hidden_cfg]
        elif isinstance(hidden_cfg, (int, float, bool)):
            hidden_layers = [_as_int(hidden_cfg)]
        else:
            hidden_layers = [64, 64]
        act_cls = self._torch_activation(str(spec.get("activation", "relu")))

        layers = []
        in_dim = inputs
        for h in hidden_layers:
            if h <= 0:
                continue
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, outputs))
        model = nn.Sequential(*layers)
        self._torch_models[model_name] = model
        return model

    def _torch_make_data(self, dataset_name: str, model_name: str):
        if not TORCH_AVAILABLE:
            raise RuntimeErrorNF("PyTorch is not available")
        ds = self.datasets.get(dataset_name)
        if ds is None:
            raise RuntimeErrorNF(f"Unknown dataset: {dataset_name}")
        model_spec = self.models.get(model_name, {})
        samples = max(8, _as_int(ds.get("samples", 256)))
        inputs = max(1, _as_int(ds.get("inputs", model_spec.get("inputs", 8))))
        outputs = max(1, _as_int(ds.get("outputs", model_spec.get("outputs", 1))))
        task = str(ds.get("task", "classification" if outputs > 1 else "regression")).lower()
        batch_size = max(1, _as_int(ds.get("batch_size", min(64, samples))))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._base_seed or 0)
        x = torch.randn(samples, inputs, generator=gen)
        w = torch.randn(inputs, outputs, generator=gen)
        logits = x @ w

        if task.startswith("class"):
            # Classification: labels are argmax of synthetic logits
            y = torch.argmax(logits, dim=1)
            criterion = nn.CrossEntropyLoss()
        else:
            noise = 0.05 * torch.randn(samples, outputs, generator=gen)
            y = logits + noise
            criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
        return loader, criterion, task, batch_size

    def _torch_pick_device(self, requested: Optional[str] = None) -> Dict[str, Any]:
        req = str(requested or "auto").strip().lower()
        if not TORCH_AVAILABLE or torch is None:
            return {"device": "cpu", "requested": req, "used_fallback": True, "fallback_reason": "torch_missing"}
        cuda_avail = False
        mps_avail = False
        xpu_avail = False
        try:
            cuda_avail = bool(getattr(torch.cuda, "is_available", lambda: False)())
        except Exception:
            cuda_avail = False
        try:
            mps_backend = getattr(torch.backends, "mps", None)
            mps_avail = bool(mps_backend and mps_backend.is_available())
        except Exception:
            mps_avail = False
        try:
            xpu_mod = getattr(torch, "xpu", None)
            xpu_avail = bool(xpu_mod and xpu_mod.is_available())
        except Exception:
            xpu_avail = False

        if req in {"cpu"}:
            return {"device": "cpu", "requested": req, "used_fallback": False, "fallback_reason": None}
        if req in {"cuda", "gpu"}:
            if cuda_avail:
                return {"device": "cuda", "requested": req, "used_fallback": False, "fallback_reason": None}
            return {"device": "cpu", "requested": req, "used_fallback": True, "fallback_reason": "cuda_unavailable"}
        if req == "mps":
            if mps_avail:
                return {"device": "mps", "requested": req, "used_fallback": False, "fallback_reason": None}
            return {"device": "cpu", "requested": req, "used_fallback": True, "fallback_reason": "mps_unavailable"}
        if req in {"xpu", "npu"}:
            if xpu_avail:
                return {"device": "xpu", "requested": req, "logical_device": "npu" if req == "npu" else "xpu", "used_fallback": False, "fallback_reason": None}
            return {"device": "cpu", "requested": req, "logical_device": "npu" if req == "npu" else None, "used_fallback": True, "fallback_reason": "xpu_unavailable"}

        # auto
        if cuda_avail:
            return {"device": "cuda", "requested": req, "used_fallback": False, "fallback_reason": None}
        if mps_avail:
            return {"device": "mps", "requested": req, "used_fallback": False, "fallback_reason": None}
        if xpu_avail:
            return {"device": "xpu", "requested": req, "used_fallback": False, "fallback_reason": None}
        return {"device": "cpu", "requested": req, "used_fallback": False, "fallback_reason": None}

    def _torch_train(self, model_name: str, dataset_name: str, epochs: int, lr: float = 1e-3, batch_size: Optional[int] = None, device_pref: Optional[str] = None) -> None:
        if not TORCH_AVAILABLE:
            self._record_unsupported_step("torch_train", [model_name, dataset_name, epochs, lr, batch_size, device_pref], TORCH_IMPORT_ERROR or "torch_missing")
            return
        model = self._torch_models.get(model_name) or self._torch_build_model(model_name)
        loader, criterion, task, ds_batch_size = self._torch_make_data(dataset_name, model_name)
        if batch_size and ds_batch_size != batch_size:
            # Rebuild loader with requested batch size over same synthetic tensors
            dataset = loader.dataset  # type: ignore[assignment]
            loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=True)
            ds_batch_size = max(1, int(batch_size))

        device_sel = self._torch_pick_device(device_pref)
        device_name = str(device_sel.get("device", "cpu"))
        try:
            model = model.to(torch.device(device_name))
        except Exception:
            device_sel = {"device": "cpu", "requested": device_sel.get("requested", device_pref or "auto"), "used_fallback": True, "fallback_reason": "model_to_device_failed"}
            device_name = "cpu"
            model = model.to(torch.device("cpu"))

        optimizer = optim.Adam(model.parameters(), lr=float(lr))
        model.train()
        loss_history: List[float] = []
        start = time.time()
        for _ in range(max(1, epochs)):
            running = 0.0
            batches = 0
            for xb, yb in loader:
                if device_name != "cpu":
                    try:
                        xb = xb.to(device_name)
                        yb = yb.to(device_name)
                    except Exception:
                        xb = xb.to("cpu")
                        yb = yb.to("cpu")
                        model = model.to(torch.device("cpu"))
                        device_name = "cpu"
                        device_sel = {"device": "cpu", "requested": device_sel.get("requested", device_pref or "auto"), "used_fallback": True, "fallback_reason": "batch_to_device_failed"}
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                if task.startswith("class"):
                    loss = criterion(pred, yb.long())
                else:
                    if yb.ndim == 1:
                        yb = yb.unsqueeze(1)
                    loss = criterion(pred, yb.float())
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu().item())
                batches += 1
            loss_history.append(running / max(1, batches))

        with self._lock:
            tick_now = self.tick_count
        run = {
            "backend": "pytorch",
            "model": model_name,
            "dataset": dataset_name,
            "epochs": max(1, epochs),
            "lr": float(lr),
            "batch_size": ds_batch_size,
            "device": device_name,
            "device_requested": device_sel.get("requested", device_pref or "auto"),
            "device_used_fallback": bool(device_sel.get("used_fallback", False)),
            "device_fallback_reason": device_sel.get("fallback_reason"),
            "loss_history": [round(v, 6) for v in loss_history],
            "final_loss": round(loss_history[-1], 6) if loss_history else None,
            "duration_ms": round((time.time() - start) * 1000, 2),
            "tick_at_start": tick_now,
        }
        self._record_training_run(run)
        with self._lock:
            self.torch_state["trained_models"] = sorted(set(list(self.torch_state["trained_models"]) + [model_name]))
        self._append_event({"tick": tick_now, "event": f"torch_train:{model_name}:{dataset_name}"})

    def _torch_export(self, model_name: str, rel_path: str) -> None:
        if not TORCH_AVAILABLE:
            self._record_unsupported_step("torch_export", [model_name, rel_path], TORCH_IMPORT_ERROR or "torch_missing")
            return
        model = self._torch_models.get(model_name)
        if model is None:
            model = self._torch_build_model(model_name)
        out = self._resolve_output_path(rel_path)
        payload = {
            "project": self.project.name,
            "model_name": model_name,
            "spec": self.models.get(model_name, {}),
            "tick": self.tick_count,
            "state_dict": model.state_dict(),
        }
        torch.save(payload, out)
        meta_path = out.with_suffix(out.suffix + ".json")
        meta_path.write_text(
            json.dumps(
                {
                    "project": self.project.name,
                    "model_name": model_name,
                    "path": str(out),
                    "tick": self.tick_count,
                    "spec": self.models.get(model_name, {}),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with self._lock:
            self.torch_state["exports"].append({"model": model_name, "path": str(out), "meta": str(meta_path)})
            self.torch_state["exports"] = self.torch_state["exports"][-20:]
            tick_now = self.tick_count
        self._append_event({"tick": tick_now, "event": f"torch_export:{model_name}"})

    def exec_pipeline_step(self, step_name: str, args: List[Any]) -> None:
        if step_name == "simulate":
            if len(args) != 1:
                raise RuntimeErrorNF("simulate(ticks) expects 1 arg")
            self.simulate(_as_int(args[0]))
            return
        if step_name == "auto_simulate":
            if len(args) != 1:
                raise RuntimeErrorNF("auto_simulate(ticks) expects 1 arg")
            self.auto_simulate(_as_int(args[0]))
            return
        if step_name == "simulate_mt":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("simulate_mt(ticks, workers[, chunk_size]) expects 2 or 3 args")
            ticks = _as_int(args[0])
            workers = _as_int(args[1])
            chunk_size = _as_int(args[2]) if len(args) == 3 else 0
            self.simulate_mt(ticks, workers, chunk_size)
            return
        if step_name == "summary":
            return
        if step_name == "run_pipeline":
            if len(args) != 1:
                raise RuntimeErrorNF("run_pipeline(name) expects 1 arg")
            self.run_pipeline(str(args[0]))
            return
        if step_name == "emit_event":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("emit_event(label[, payload]) expects 1 or 2 args")
            with self._lock:
                tick_now = self.tick_count
            evt: Dict[str, Any] = {"tick": tick_now, "event": str(args[0])}
            if len(args) == 2:
                evt["payload"] = args[1]
            self._append_event(evt)
            return
        if step_name == "set_state":
            if len(args) != 2:
                raise RuntimeErrorNF("set_state(name, value) expects 2 args")
            with self._lock:
                self.state[str(args[0])] = args[1]
            return
        if step_name == "inc_state":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("inc_state(name[, delta]) expects 1 or 2 args")
            key = str(args[0])
            delta = args[1] if len(args) == 2 else 1
            with self._lock:
                cur = self.state.get(key, 0)
                self.state[key] = cur + delta
            return
        if step_name == "assert":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("assert(condition[, message]) expects 1 or 2 args")
            condition = bool(args[0])
            if not condition:
                msg = str(args[1]) if len(args) == 2 else "assertion failed"
                raise RuntimeErrorNF(msg)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "assert_ok"})
            return
        if step_name == "sleep":
            if len(args) != 1:
                raise RuntimeErrorNF("sleep(ms) expects 1 arg")
            ms = max(0, _as_int(args[0]))
            time.sleep(ms / 1000.0)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "sleep", "ms": ms})
            return
        if step_name == "proc_profile":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("proc_profile(name[, config]) expects 1 or 2 args")
            cfg = args[1] if len(args) == 2 and isinstance(args[1], dict) else None
            self._proc_profile_register(str(args[0]), cfg=cfg)
            return
        if step_name == "proc_exec":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("proc_exec(command[, args_or_cfg[, config]]) expects 1-3 args")
            command = str(args[0])
            cmd_args = None
            cfg = None
            if len(args) >= 2:
                if isinstance(args[1], dict):
                    cfg = args[1]
                else:
                    cmd_args = args[1]
            if len(args) == 3 and isinstance(args[2], dict):
                cfg = args[2]
            self._proc_exec(command, args=cmd_args, cfg=cfg)
            return
        if step_name == "proc_profile_run":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("proc_profile_run(profile, command[, args_or_cfg[, config]]) expects 2-4 args")
            profile = str(args[0])
            command = str(args[1])
            cmd_args = None
            cfg = None
            if len(args) >= 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    cmd_args = args[2]
            if len(args) == 4 and isinstance(args[3], dict):
                cfg = args[3]
            self._proc_exec(command, args=cmd_args, cfg=cfg, profile_name=profile)
            return
        if step_name == "proc_spawn":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("proc_spawn(name, command[, args_or_cfg[, config]]) expects 2-4 args")
            proc_name = str(args[0])
            command = str(args[1])
            cmd_args = None
            cfg = None
            if len(args) >= 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    cmd_args = args[2]
            if len(args) == 4 and isinstance(args[3], dict):
                cfg = args[3]
            self._proc_spawn(proc_name, command, args=cmd_args, cfg=cfg)
            return
        if step_name == "proc_wait":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("proc_wait(name[, timeout_sec]) expects 1 or 2 args")
            timeout_sec = float(args[1]) if len(args) == 2 else 30.0
            self._proc_wait(str(args[0]), timeout_sec=timeout_sec)
            return
        if step_name == "proc_kill":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("proc_kill(name[, force]) expects 1 or 2 args")
            force = bool(args[1]) if len(args) == 2 else False
            self._proc_kill(str(args[0]), force=force)
            return
        if step_name == "proc_history_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("proc_history_json(path[, limit]) expects 1 or 2 args")
            limit = _as_int(args[1]) if len(args) == 2 else None
            out = self._proc_history_json(str(args[0]), limit=limit)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "proc_history_json", "path": str(out)})
            return
        if step_name == "proc_managed_json":
            if len(args) != 1:
                raise RuntimeErrorNF("proc_managed_json(path) expects 1 arg")
            out = self._proc_managed_json(str(args[0]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "proc_managed_json", "path": str(out)})
            return
        if step_name == "export_json":
            if len(args) != 1:
                raise RuntimeErrorNF("export_json(path) expects 1 arg")
            self.export_json(str(args[0]))
            return
        if step_name == "export_html":
            if len(args) != 1:
                raise RuntimeErrorNF("export_html(path) expects 1 arg")
            self.export_html(str(args[0]))
            return
        if step_name == "load_3d":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("load_3d(alias, path[, kind]) expects 2 or 3 args")
            kind = str(args[2]) if len(args) == 3 and args[2] is not None else None
            self._load_3d_asset(str(args[0]), str(args[1]), kind)
            return
        if step_name == "scene_new":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("scene_new(name[, template]) expects 1 or 2 args")
            template = str(args[1]) if len(args) == 2 else "stage"
            self._new_scene3d(str(args[0]), template=template)
            return
        if step_name == "scene_add":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("scene_add(scene, asset[, node_name[, transform]]) expects 2-4 args")
            node_name = str(args[2]) if len(args) >= 3 and args[2] is not None else None
            transform = args[3] if len(args) == 4 else None
            self._scene3d_add_node(str(args[0]), str(args[1]), node_name=node_name, transform=transform if isinstance(transform, dict) else None)
            return
        if step_name == "scene_light":
            if len(args) not in {1, 2, 3, 4}:
                raise RuntimeErrorNF("scene_light(scene[, kind[, intensity[, color]]]) expects 1-4 args")
            kind = str(args[1]) if len(args) >= 2 else "directional"
            intensity = float(args[2]) if len(args) >= 3 else 1.0
            color = str(args[3]) if len(args) >= 4 else "#ffffff"
            self._scene3d_add_light(str(args[0]), kind=kind, intensity=intensity, color=color)
            return
        if step_name == "scene_camera":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("scene_camera(scene, position, target[, fov]) expects 3 or 4 args")
            fov = float(args[3]) if len(args) == 4 else 50.0
            self._scene3d_set_camera(str(args[0]), args[1], args[2], fov=fov)
            return
        if step_name == "export_scene_json":
            if len(args) != 2:
                raise RuntimeErrorNF("export_scene_json(scene, path) expects 2 args")
            self._export_scene_json(str(args[0]), str(args[1]))
            return
        if step_name == "export_scene_html":
            if len(args) != 2:
                raise RuntimeErrorNF("export_scene_html(scene, path) expects 2 args")
            self._export_scene_html(str(args[0]), str(args[1]))
            return
        if step_name == "web_tool_generate":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("web_tool_generate(kind, path[, title]) expects 2 or 3 args")
            title = str(args[2]) if len(args) == 3 else None
            self._web_tool_generate(str(args[0]), str(args[1]), title=title)
            return
        if step_name == "web_tool_suite":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("web_tool_suite(dir[, preset]) expects 1 or 2 args")
            preset = str(args[1]) if len(args) == 2 else "lab"
            self._web_tool_suite(str(args[0]), preset=preset)
            return
        if step_name == "web_live_tool_suite":
            if len(args) != 1:
                raise RuntimeErrorNF("web_live_tool_suite(dir) expects 1 arg")
            self._web_tool_suite(str(args[0]), preset="live_api")
            return
        if step_name == "web_interaction_tool":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("web_interaction_tool(path[, title]) expects 1 or 2 args")
            title = str(args[1]) if len(args) == 2 else "NexusFlow Interaction Lab"
            self._web_tool_generate("interaction_lab", str(args[0]), title=title)
            return
        if step_name == "nexus_ide":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("nexus_ide(path[, title]) expects 1 or 2 args")
            title = str(args[1]) if len(args) == 2 else "NexusFlow IDE"
            self._web_tool_generate("nexus_ide", str(args[0]), title=title)
            return
        if step_name == "nexus_idle":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("nexus_idle(path[, title]) expects 1 or 2 args")
            title = str(args[1]) if len(args) == 2 else "NexusFlow IDLE"
            self._web_tool_generate("nexus_idle", str(args[0]), title=title)
            return
        if step_name == "nexus_dev_suite":
            if len(args) != 1:
                raise RuntimeErrorNF("nexus_dev_suite(dir) expects 1 arg")
            self._web_tool_suite(str(args[0]), preset="nexus_dev")
            return
        if step_name == "lang_module":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("lang_module(name, language, path[, config]) expects 3 or 4 args")
            cfg = args[3] if len(args) == 4 and isinstance(args[3], dict) else None
            self._register_lang_module(str(args[0]), str(args[1]), str(args[2]), cfg=cfg)
            return
        if step_name == "py_module":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("py_module(name, path[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._register_lang_module(str(args[0]), "python", str(args[1]), cfg=cfg)
            return
        if step_name == "js_module":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("js_module(name, path[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._register_lang_module(str(args[0]), "javascript", str(args[1]), cfg=cfg)
            return
        if step_name == "cpp_module":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("cpp_module(name, path[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._register_lang_module(str(args[0]), "cpp", str(args[1]), cfg=cfg)
            return
        if step_name in {"rust_module", "rs_module"}:
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF(f"{step_name}(name, path[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._register_lang_module(str(args[0]), "rust", str(args[1]), cfg=cfg)
            return
        if step_name in {"csharp_module", "cs_module"}:
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF(f"{step_name}(name, path[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._register_lang_module(str(args[0]), "csharp", str(args[1]), cfg=cfg)
            return
        if step_name == "cpp_build":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("cpp_build(name[, out_path[, config]]) expects 1-3 args")
            out_path = str(args[1]) if len(args) >= 2 and isinstance(args[1], str) else None
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            result = self._cpp_build_module(str(args[0]), out_path=out_path, cfg=cfg)
            if not result.get("ok") and str(result.get("reason", "")).startswith("compiler_missing"):
                self._record_unsupported_step("cpp_build", args, "compiler_missing")
            return
        if step_name in {"csharp_build", "cs_build"}:
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF(f"{step_name}(name[, out_path[, config]]) expects 1-3 args")
            out_path = str(args[1]) if len(args) >= 2 and isinstance(args[1], str) else None
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            result = self._csharp_build_module(str(args[0]), out_path=out_path, cfg=cfg)
            if not result.get("ok") and str(result.get("reason", "")) in {"compiler_missing", "dotnet_requires_csproj"}:
                self._record_unsupported_step(step_name, args, str(result.get("reason")))
            return
        if step_name == "rust_build":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("rust_build(name[, out_path[, config]]) expects 1-3 args")
            out_path = str(args[1]) if len(args) >= 2 and isinstance(args[1], str) else None
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            result = self._rust_build_module(str(args[0]), out_path=out_path, cfg=cfg)
            if not result.get("ok") and str(result.get("reason", "")) in {"toolchain_missing", "cargo_missing", "rustc_missing"}:
                self._record_unsupported_step("rust_build", args, str(result.get("reason")))
            return
        if step_name == "lang_run":
            if len(args) not in {1, 2, 3, 4}:
                raise RuntimeErrorNF("lang_run(name[, args_or_single_arg[, timeout[, stdin_text]]]) expects 1-4 args")
            run_args = None
            timeout_sec = 30.0
            stdin_text = None
            if len(args) >= 2:
                run_args = args[1]
            if len(args) >= 3:
                timeout_sec = float(args[2])
            if len(args) == 4:
                stdin_text = self._safe_text(args[3])
            result = self._lang_module_run(str(args[0]), run_args, timeout_sec=timeout_sec, stdin_text=stdin_text)
            reason = str(result.get("reason", ""))
            if not result.get("ok") and (
                reason.startswith("runtime_missing")
                or reason in {"compiler_missing", "cpp_build_failed", "csharp_build_failed", "dotnet_requires_csproj"}
            ):
                self._record_unsupported_step("lang_run", args, reason)
            return
        if step_name == "cpp_run":
            if len(args) not in {1, 2, 3, 4}:
                raise RuntimeErrorNF("cpp_run(name[, args_or_single_arg[, timeout[, stdin_text]]]) expects 1-4 args")
            run_args = None
            timeout_sec = 30.0
            stdin_text = None
            if len(args) >= 2:
                run_args = args[1]
            if len(args) >= 3:
                timeout_sec = float(args[2])
            if len(args) == 4:
                stdin_text = self._safe_text(args[3])
            result = self._lang_module_run(str(args[0]), run_args, timeout_sec=timeout_sec, stdin_text=stdin_text)
            if not result.get("ok") and str(result.get("reason", "")) in {"compiler_missing", "cpp_build_failed"}:
                self._record_unsupported_step("cpp_run", args, str(result.get("reason")))
            return
        if step_name in {"csharp_run", "cs_run"}:
            if len(args) not in {1, 2, 3, 4}:
                raise RuntimeErrorNF(f"{step_name}(name[, args_or_single_arg[, timeout[, stdin_text]]]) expects 1-4 args")
            run_args = None
            timeout_sec = 30.0
            stdin_text = None
            if len(args) >= 2:
                run_args = args[1]
            if len(args) >= 3:
                timeout_sec = float(args[2])
            if len(args) == 4:
                stdin_text = self._safe_text(args[3])
            result = self._lang_module_run(str(args[0]), run_args, timeout_sec=timeout_sec, stdin_text=stdin_text)
            reason = str(result.get("reason", ""))
            if not result.get("ok") and (
                reason.startswith("runtime_missing")
                or reason in {"compiler_missing", "csharp_build_failed", "dotnet_requires_csproj"}
            ):
                self._record_unsupported_step(step_name, args, reason)
            return
        if step_name == "rust_run":
            if len(args) not in {1, 2, 3, 4}:
                raise RuntimeErrorNF("rust_run(name[, args_or_single_arg[, timeout[, stdin_text]]]) expects 1-4 args")
            run_args = None
            timeout_sec = 30.0
            stdin_text = None
            if len(args) >= 2:
                run_args = args[1]
            if len(args) >= 3:
                timeout_sec = float(args[2])
            if len(args) == 4:
                stdin_text = self._safe_text(args[3])
            result = self._lang_module_run(str(args[0]), run_args, timeout_sec=timeout_sec, stdin_text=stdin_text)
            if not result.get("ok") and str(result.get("reason", "")) in {"toolchain_missing", "cargo_missing", "rustc_missing", "rust_build_failed"}:
                self._record_unsupported_step("rust_run", args, str(result.get("reason")))
            return
        if step_name == "http_auth_preset":
            if len(args) != 2 or not isinstance(args[1], dict):
                raise RuntimeErrorNF("http_auth_preset(name, headers_map) expects 2 args with dict headers")
            self._set_http_auth_preset(str(args[0]), args[1])
            return
        if step_name == "http_clear_auth_preset":
            if len(args) != 1:
                raise RuntimeErrorNF("http_clear_auth_preset(name) expects 1 arg")
            self._clear_http_auth_preset(str(args[0]))
            return
        if step_name == "mock_http_server_start":
            if len(args) not in {1, 2, 3, 4}:
                raise RuntimeErrorNF("mock_http_server_start(name[, port_or_routes[, routes_or_host_or_cfg[, host_or_cfg]]]) expects 1-4 args")
            port = 0
            routes = None
            host = "127.0.0.1"
            cfg = None
            tail = list(args[1:])
            for item in tail:
                if isinstance(item, (int, float)):
                    port = _as_int(item)
                elif isinstance(item, dict):
                    if routes is None:
                        routes = item
                    else:
                        cfg = item
                elif isinstance(item, str):
                    host = str(item)
            self._mock_http_server_start(str(args[0]), port=port, routes=routes, host=host, cfg=cfg)
            return
        if step_name == "mock_http_server_stop":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("mock_http_server_stop(name[, timeout_sec]) expects 1 or 2 args")
            timeout_sec = float(args[1]) if len(args) == 2 else 2.0
            self._mock_http_server_stop(str(args[0]), timeout_sec=timeout_sec)
            return
        if step_name == "export_http_history_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("export_http_history_json(path[, limit]) expects 1 or 2 args")
            limit = _as_int(args[1]) if len(args) == 2 else None
            self._export_http_history_json(str(args[0]), limit=limit)
            return
        if step_name == "http_download":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("http_download(url, path[, timeout_or_preset_or_headers[, timeout]]) expects 2-4 args")
            timeout_sec = 30.0
            auth_preset = None
            headers = None
            if len(args) >= 3:
                if isinstance(args[2], (int, float)):
                    timeout_sec = float(args[2])
                elif isinstance(args[2], str):
                    auth_preset = str(args[2])
                elif isinstance(args[2], dict):
                    headers = args[2]
            if len(args) == 4:
                timeout_sec = float(args[3])
            self._http_download(str(args[0]), str(args[1]), timeout_sec=timeout_sec, headers=headers, auth_preset=auth_preset)
            return
        if step_name == "http_fetch_json":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("http_fetch_json(url, path[, timeout_or_preset_or_headers[, timeout]]) expects 2-4 args")
            timeout_sec = 15.0
            auth_preset = None
            headers = None
            if len(args) >= 3:
                if isinstance(args[2], (int, float)):
                    timeout_sec = float(args[2])
                elif isinstance(args[2], str):
                    auth_preset = str(args[2])
                elif isinstance(args[2], dict):
                    headers = args[2]
            if len(args) == 4:
                timeout_sec = float(args[3])
            self._http_fetch_json_to_file(str(args[0]), str(args[1]), timeout_sec=timeout_sec, headers=headers, auth_preset=auth_preset)
            return
        if step_name == "fusion_sim":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("fusion_sim(name[, steps[, config]]) expects 1-3 args")
            steps = _as_int(args[1]) if len(args) >= 2 and args[1] is not None else 120
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._fusion_sim(str(args[0]), steps=steps, cfg=cfg)
            return
        if step_name == "fusion_sim_multizone":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("fusion_sim_multizone(name[, steps[, config]]) expects 1-3 args")
            steps = _as_int(args[1]) if len(args) >= 2 and args[1] is not None else 120
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else {}
            if not isinstance(cfg, dict):
                cfg = {}
            cfg = {**cfg, "mode": "multizone"}
            self._fusion_sim_multizone(str(args[0]), steps=steps, cfg=cfg)
            return
        if step_name == "fusion_control_sim":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("fusion_control_sim(name[, steps[, config]]) expects 1-3 args")
            steps = _as_int(args[1]) if len(args) >= 2 and args[1] is not None else 180
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._fusion_control_sim(str(args[0]), steps=steps, cfg=cfg)
            return
        if step_name == "fusion_sweep":
            if len(args) not in {2, 3, 4} or not isinstance(args[1], dict):
                raise RuntimeErrorNF("fusion_sweep(name, grid[, steps[, config]]) expects 2-4 args with grid dict")
            steps = _as_int(args[2]) if len(args) >= 3 and args[2] is not None else 80
            cfg = args[3] if len(args) == 4 and isinstance(args[3], dict) else None
            self._fusion_sweep(str(args[0]), args[1], steps=steps, cfg=cfg)
            return
        if step_name == "export_fusion_json":
            if len(args) != 2:
                raise RuntimeErrorNF("export_fusion_json(run, path) expects 2 args")
            self._export_fusion_json(str(args[0]), str(args[1]))
            return
        if step_name == "export_fusion_html":
            if len(args) != 2:
                raise RuntimeErrorNF("export_fusion_html(run, path) expects 2 args")
            self._export_fusion_html(str(args[0]), str(args[1]))
            return
        if step_name == "protein_fold_sim":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("protein_fold_sim(name, sequence[, steps[, config]]) expects 2-4 args")
            steps = _as_int(args[2]) if len(args) >= 3 and args[2] is not None else 200
            cfg = args[3] if len(args) == 4 and isinstance(args[3], dict) else None
            self._protein_fold_sim(str(args[0]), str(args[1]), steps=steps, cfg=cfg)
            return
        if step_name == "protein_fold_sim_3d":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("protein_fold_sim_3d(name, sequence[, steps[, config]]) expects 2-4 args")
            steps = _as_int(args[2]) if len(args) >= 3 and args[2] is not None else 250
            cfg = args[3] if len(args) == 4 and isinstance(args[3], dict) else {}
            if not isinstance(cfg, dict):
                cfg = {}
            cfg = {**cfg, "mode": "lattice3d"}
            self._protein_fold_sim_3d(str(args[0]), str(args[1]), steps=steps, cfg=cfg)
            return
        if step_name == "export_protein_json":
            if len(args) != 2:
                raise RuntimeErrorNF("export_protein_json(run, path) expects 2 args")
            self._export_protein_json(str(args[0]), str(args[1]))
            return
        if step_name == "export_protein_html":
            if len(args) != 2:
                raise RuntimeErrorNF("export_protein_html(run, path) expects 2 args")
            self._export_protein_html(str(args[0]), str(args[1]))
            return
        if step_name == "export_csv":
            if len(args) != 1:
                raise RuntimeErrorNF("export_csv(path) expects 1 arg")
            self._export_csv(str(args[0]))
            return
        if step_name == "export_events_jsonl":
            if len(args) != 1:
                raise RuntimeErrorNF("export_events_jsonl(path) expects 1 arg")
            self._export_events_jsonl(str(args[0]))
            return
        if step_name == "write_text":
            if len(args) != 2:
                raise RuntimeErrorNF("write_text(path, text) expects 2 args")
            self._write_text(str(args[0]), args[1], append=False)
            return
        if step_name == "append_text":
            if len(args) != 2:
                raise RuntimeErrorNF("append_text(path, text) expects 2 args")
            self._write_text(str(args[0]), args[1], append=True)
            return
        if step_name == "save_state":
            if len(args) != 1:
                raise RuntimeErrorNF("save_state(path) expects 1 arg")
            self._save_state_file(str(args[0]))
            return
        if step_name == "load_state":
            if len(args) != 1:
                raise RuntimeErrorNF("load_state(path) expects 1 arg")
            self._load_state_file(str(args[0]))
            return
        if step_name == "resource_set_limits":
            if len(args) != 1 or not isinstance(args[0], dict):
                raise RuntimeErrorNF("resource_set_limits(config_map) expects 1 dict arg")
            self._resource_set_limits(args[0])
            return
        if step_name == "resource_snapshot":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("resource_snapshot([label]) expects 0 or 1 args")
            label = str(args[0]) if len(args) == 1 else None
            self._resource_snapshot(label=label)
            return
        if step_name == "resource_gc":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("resource_gc([generation]) expects 0 or 1 args")
            gen = _as_int(args[0]) if len(args) == 1 else None
            self._resource_gc(gen)
            return
        if step_name == "resource_trim":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("resource_trim([config]) expects 0 or 1 args")
            cfg = args[0] if len(args) == 1 and isinstance(args[0], dict) else None
            self._resource_trim(cfg)
            return
        if step_name == "resource_optimize":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("resource_optimize([config]) expects 0 or 1 args")
            cfg = args[0] if len(args) == 1 and isinstance(args[0], dict) else None
            self._resource_optimize(cfg)
            return
        if step_name == "export_resource_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("export_resource_json(path[, limit]) expects 1 or 2 args")
            limit = _as_int(args[1]) if len(args) == 2 else None
            self._export_resource_json(str(args[0]), limit=limit)
            return
        if step_name == "render_template":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("render_template(path, template, data[, strict]) expects 3 or 4 args")
            strict = bool(args[3]) if len(args) == 4 else False
            rendered = self._template_fill(str(args[1]), args[2], strict=strict)
            out = self._write_text(str(args[0]), rendered, append=False)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "render_template", "path": str(out)})
            return
        if step_name == "sqlite_exec":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("sqlite_exec(db, sql[, params]) expects 2 or 3 args")
            params = args[2] if len(args) == 3 else None
            self._sqlite_exec(str(args[0]), str(args[1]), params=params)
            return
        if step_name == "sqlite_query_json":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("sqlite_query_json(db, sql, path[, params]) expects 3 or 4 args")
            params = args[3] if len(args) == 4 else None
            self._sqlite_query_json(str(args[0]), str(args[1]), str(args[2]), params=params)
            return
        if step_name == "sqlite_export_csv":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("sqlite_export_csv(db, sql, path[, params]) expects 3 or 4 args")
            params = args[3] if len(args) == 4 else None
            self._sqlite_export_csv(str(args[0]), str(args[1]), str(args[2]), params=params)
            return
        if step_name == "sqlite_import_csv":
            if len(args) not in {3, 4, 5}:
                raise RuntimeErrorNF("sqlite_import_csv(db, table, csv_path[, header_or_cfg[, cfg]]) expects 3-5 args")
            header = True
            cfg = None
            if len(args) >= 4:
                if isinstance(args[3], dict):
                    cfg = args[3]
                else:
                    header = bool(args[3])
            if len(args) == 5 and isinstance(args[4], dict):
                cfg = args[4]
            self._sqlite_import_csv(str(args[0]), str(args[1]), str(args[2]), header=header, cfg=cfg)
            return
        if step_name == "zip_pack":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("zip_pack(source, archive[, include_root]) expects 2 or 3 args")
            include_root = bool(args[2]) if len(args) == 3 else False
            self._zip_pack(str(args[0]), str(args[1]), include_root=include_root)
            return
        if step_name == "zip_unpack":
            if len(args) != 2:
                raise RuntimeErrorNF("zip_unpack(archive, dest_dir) expects 2 args")
            self._zip_unpack(str(args[0]), str(args[1]))
            return
        if step_name == "tar_pack":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("tar_pack(source, archive[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._tar_pack(str(args[0]), str(args[1]), cfg=cfg)
            return
        if step_name == "tar_unpack":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("tar_unpack(archive, dest_dir[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._tar_unpack(str(args[0]), str(args[1]), cfg=cfg)
            return
        if step_name == "hash_file_json":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("hash_file_json(path, out[, algo_or_cfg[, cfg]]) expects 2-4 args")
            algo = None
            cfg = None
            if len(args) >= 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    algo = str(args[2]) if args[2] is not None else None
            if len(args) == 4:
                if not isinstance(args[3], dict):
                    raise RuntimeErrorNF("hash_file_json(..., cfg) final arg must be a dict")
                cfg = args[3]
            self._hash_file_json(str(args[0]), str(args[1]), algo=algo, cfg=cfg)
            return
        if step_name == "dir_manifest_json":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("dir_manifest_json(path, out[, recursive_or_cfg[, cfg]]) expects 2-4 args")
            recursive = True
            cfg = None
            if len(args) >= 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    recursive = bool(args[2])
            if len(args) == 4:
                if not isinstance(args[3], dict):
                    raise RuntimeErrorNF("dir_manifest_json(..., cfg) final arg must be a dict")
                cfg = args[3]
            self._dir_manifest_json(str(args[0]), str(args[1]), recursive=recursive, cfg=cfg)
            return
        if step_name == "dir_diff_json":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("dir_diff_json(left, right, out[, cfg]) expects 3 or 4 args")
            cfg = args[3] if len(args) == 4 and isinstance(args[3], dict) else None
            self._dir_diff_json(str(args[0]), str(args[1]), str(args[2]), cfg=cfg)
            return
        if step_name == "exe_build":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("exe_build(source_or_module, out_path[, name_or_cfg[, cfg]]) expects 2-4 args")
            name = None
            cfg = None
            if len(args) >= 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    name = str(args[2]) if args[2] is not None else None
            if len(args) == 4:
                if not isinstance(args[3], dict):
                    raise RuntimeErrorNF("exe_build(..., cfg) final arg must be a dict")
                cfg = args[3]
            result = self._exe_build(str(args[0]), str(args[1]), name=name, cfg=cfg)
            reason = str(result.get("reason", ""))
            if not result.get("ok") and (
                reason.startswith("tool_missing")
                or reason in {
                    "unsupported_module_language",
                    "unsupported_source_type",
                    "toolchain_missing",
                    "cargo_missing",
                    "rustc_missing",
                    "compiler_missing",
                    "dotnet_requires_csproj",
                }
            ):
                self._record_unsupported_step("exe_build", args, reason or "build_unavailable")
            return
        if step_name == "file_convert":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("file_convert(src, dst[, mode_or_cfg[, cfg]]) expects 2-4 args")
            mode = None
            cfg = None
            if len(args) >= 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    mode = str(args[2]) if args[2] is not None else None
            if len(args) == 4:
                if not isinstance(args[3], dict):
                    raise RuntimeErrorNF("file_convert(..., cfg) final arg must be a dict")
                cfg = args[3]
            self._file_convert(str(args[0]), str(args[1]), mode=mode, cfg=cfg)
            return
        if step_name == "iso_manifest_json":
            if len(args) != 2:
                raise RuntimeErrorNF("iso_manifest_json(source_dir, out_path) expects 2 args")
            out = self._iso_manifest_json(str(args[0]), str(args[1]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "iso_manifest_json", "path": str(out)})
            return
        if step_name == "photo_generate":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("photo_generate(path, width, height[, prompt_or_cfg]) expects 3 or 4 args")
            self._photo_generate(str(args[0]), _as_int(args[1]), _as_int(args[2]), args[3] if len(args) == 4 else None)
            return
        if step_name == "photo_batch_generate":
            if len(args) not in {4, 5}:
                raise RuntimeErrorNF("photo_batch_generate(dir, width, height, prompts[, cfg]) expects 4 or 5 args")
            cfg = args[4] if len(args) == 5 and isinstance(args[4], dict) else None
            if len(args) == 5 and args[4] is not None and not isinstance(args[4], dict):
                raise RuntimeErrorNF("photo_batch_generate(..., cfg) final arg must be a dict")
            self._photo_batch_generate(str(args[0]), _as_int(args[1]), _as_int(args[2]), args[3], cfg=cfg)
            return
        if step_name == "photo_poster":
            if len(args) not in {3, 4, 5}:
                raise RuntimeErrorNF("photo_poster(path, width, height[, title_or_cfg[, cfg]]) expects 3-5 args")
            title_or_cfg = args[3] if len(args) >= 4 else None
            cfg = None
            if len(args) == 5:
                if not isinstance(args[4], dict):
                    raise RuntimeErrorNF("photo_poster(..., cfg) final arg must be a dict")
                cfg = args[4]
            self._photo_poster(str(args[0]), _as_int(args[1]), _as_int(args[2]), title_or_cfg=title_or_cfg, cfg=cfg)
            return
        if step_name in {"photo_collage", "photo_contact_sheet"}:
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF(f"{step_name}(path, photos_or_batch[, cfg]) expects 2 or 3 args")
            cfg = None
            if len(args) == 3:
                if not isinstance(args[2], dict):
                    raise RuntimeErrorNF(f"{step_name}(..., cfg) final arg must be a dict")
                cfg = copy.deepcopy(args[2])
            photos_arg = args[1]
            if isinstance(photos_arg, (str, Path)):
                batch_meta = self._photo_batch_find(photos_arg)
                if isinstance(batch_meta, dict):
                    photos_arg = [it.get("path", "") for it in (batch_meta.get("items") or []) if isinstance(it, dict)]
                    if step_name == "photo_contact_sheet":
                        cfg = cfg or {}
                        cfg.setdefault("title", f"Contact Sheet: {batch_meta.get('name', 'batch')}")
                        cfg.setdefault("subtitle", f"{len(photos_arg)} generated photos")
            if step_name == "photo_contact_sheet":
                cfg = cfg or {}
                cfg.setdefault("title", "NexusFlow Contact Sheet")
                cfg.setdefault("gap", 12)
                cfg.setdefault("padding", 16)
                cfg.setdefault("label_height", 38)
                cfg.setdefault("tile_width", 200)
                cfg.setdefault("tile_height", 132)
            self._photo_collage(str(args[0]), photos_arg, cfg=cfg)
            return
        if step_name == "photo_filter":
            if len(args) not in {3, 4}:
                raise RuntimeErrorNF("photo_filter(source, dest, mode[, cfg]) expects 3 or 4 args")
            cfg = None
            if len(args) == 4:
                if not isinstance(args[3], dict):
                    raise RuntimeErrorNF("photo_filter(..., cfg) final arg must be a dict")
                cfg = args[3]
            self._photo_filter(str(args[0]), str(args[1]), str(args[2]), cfg=cfg)
            return
        if step_name == "data_chart_svg":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("data_chart_svg(path, values[, cfg]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._data_chart_svg(str(args[0]), args[1], cfg=cfg)
            return
        if step_name == "graph_create":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("graph_create(name, edges_or_graph[, directed_or_cfg]) expects 2 or 3 args")
            directed = False
            if len(args) == 3 and isinstance(args[2], (bool, int, float)):
                directed = bool(args[2])
            elif len(args) == 3 and isinstance(args[2], dict):
                directed = bool(args[2].get("directed", False))
                if isinstance(args[1], dict):
                    graph_obj = copy.deepcopy(args[1])
                    graph_obj["directed"] = directed
                    self._graph_store(str(args[0]), graph_obj, directed=directed)
                    return
            self._graph_store(str(args[0]), args[1], directed=directed)
            return
        if step_name == "graph_from_csv":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("graph_from_csv(name, csv_path[, cfg]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._graph_from_csv(str(args[0]), str(args[1]), cfg=cfg)
            return
        if step_name == "graph_export_svg":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("graph_export_svg(name, path[, cfg_or_title]) expects 2 or 3 args")
            cfg = None
            if len(args) == 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    cfg = {"title": str(args[2])}
            self._graph_export_svg(str(args[0]), str(args[1]), cfg=cfg)
            return
        if step_name == "graph_metrics_json":
            if len(args) != 2:
                raise RuntimeErrorNF("graph_metrics_json(name, path) expects 2 args")
            out = self._graph_metrics_json(str(args[0]), str(args[1]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "graph_metrics_json", "path": str(out)})
            return
        if step_name == "iso_build":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("iso_build(source, iso_path[, label_or_cfg[, cfg]]) expects 2-4 args")
            label = None
            cfg = None
            if len(args) >= 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    label = str(args[2]) if args[2] is not None else None
            if len(args) == 4:
                if not isinstance(args[3], dict):
                    raise RuntimeErrorNF("iso_build(..., cfg) final arg must be a dict")
                cfg = args[3]
            result = self._iso_build(str(args[0]), str(args[1]), label=label, cfg=cfg)
            if not result.get("ok") and str(result.get("reason", "")).startswith("tool_missing"):
                self._record_unsupported_step("iso_build", args, "tool_missing")
            return
        if step_name == "iso_list_json":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("iso_list_json(iso_path, out_path[, limit_or_cfg[, cfg]]) expects 2-4 args")
            limit = None
            cfg = None
            if len(args) >= 3:
                if isinstance(args[2], dict):
                    cfg = args[2]
                else:
                    limit = _as_int(args[2])
            if len(args) == 4:
                if not isinstance(args[3], dict):
                    raise RuntimeErrorNF("iso_list_json(..., cfg) final arg must be a dict")
                cfg = args[3]
            result_path = self._iso_list_json(str(args[0]), str(args[1]), limit=limit, cfg=cfg)
            # If listing failed, iso_list_json still writes a payload; mark unsupported when tool is missing.
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict) and (not payload.get("ok")) and str(payload.get("reason", "")).startswith("tool_missing"):
                self._record_unsupported_step("iso_list_json", args, "tool_missing")
            return
        if step_name == "iso_extract":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("iso_extract(iso_path, dest[, cfg]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            result = self._iso_extract(str(args[0]), str(args[1]), cfg=cfg)
            if not result.get("ok") and str(result.get("reason", "")).startswith("tool_missing"):
                self._record_unsupported_step("iso_extract", args, "tool_missing")
            return
        if step_name == "wifi_interfaces_json":
            if len(args) != 1:
                raise RuntimeErrorNF("wifi_interfaces_json(path) expects 1 arg")
            out = self._wifi_export_json("interfaces", str(args[0]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "wifi_interfaces_json", "path": str(out)})
            return
        if step_name == "wifi_profiles_json":
            if len(args) != 1:
                raise RuntimeErrorNF("wifi_profiles_json(path) expects 1 arg")
            out = self._wifi_export_json("profiles", str(args[0]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "wifi_profiles_json", "path": str(out)})
            return
        if step_name == "wifi_scan_json":
            if len(args) != 1:
                raise RuntimeErrorNF("wifi_scan_json(path) expects 1 arg")
            out = self._wifi_export_json("scan", str(args[0]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "wifi_scan_json", "path": str(out)})
            return
        if step_name == "wifi_connect":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("wifi_connect(profile[, interface]) expects 1 or 2 args")
            result = self._wifi_connect(str(args[0]), str(args[1]) if len(args) == 2 else None)
            if not result.get("ok") and str(result.get("reason")) in {"not_windows", "netsh_missing"}:
                self._record_unsupported_step("wifi_connect", args, str(result.get("reason")))
            return
        if step_name == "wifi_disconnect":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("wifi_disconnect([interface]) expects 0 or 1 arg")
            result = self._wifi_disconnect(str(args[0]) if len(args) == 1 else None)
            if not result.get("ok") and str(result.get("reason")) in {"not_windows", "netsh_missing"}:
                self._record_unsupported_step("wifi_disconnect", args, str(result.get("reason")))
            return
        if step_name == "npu_probe_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("npu_probe_json(path[, deep]) expects 1 or 2 args")
            out = self._npu_probe_json(str(args[0]), deep=bool(args[1]) if len(args) == 2 else False)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "npu_probe_json", "path": str(out)})
            return
        if step_name == "npu_profile":
            if len(args) != 2 or not isinstance(args[1], dict):
                raise RuntimeErrorNF("npu_profile(name, config) expects 2 args with dict config")
            self._npu_profile_set(str(args[0]), args[1])
            return
        if step_name == "npu_plan_json":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("npu_plan_json(path, workload_or_model[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            out = self._npu_plan_json(str(args[0]), args[1], cfg=cfg)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "npu_plan_json", "path": str(out)})
            return
        if step_name == "npu_benchmark_json":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("npu_benchmark_json(path, workload_or_model[, config]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            out = self._npu_benchmark_json(str(args[0]), args[1], cfg=cfg)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "npu_benchmark_json", "path": str(out)})
            return
        if step_name == "npu_runtime_export_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("npu_runtime_export_json(path[, config]) expects 1 or 2 args")
            cfg = args[1] if len(args) == 2 and isinstance(args[1], dict) else None
            out = self._npu_runtime_export_json(str(args[0]), cfg=cfg)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "npu_runtime_export_json", "path": str(out)})
            return
        if step_name == "npu_torch_train":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("npu_torch_train(model, dataset[, epochs[, config]]) expects 2-4 args")
            epochs = _as_int(args[2]) if len(args) >= 3 and not isinstance(args[2], dict) else 1
            cfg = None
            if len(args) == 3 and isinstance(args[2], dict):
                cfg = args[2]
            if len(args) == 4 and isinstance(args[3], dict):
                cfg = args[3]
            self._npu_torch_train(str(args[0]), str(args[1]), epochs=epochs, cfg=cfg)
            return
        if step_name == "github_portfolio_report":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("github_portfolio_report(meta_path, out_path[, cfg]) expects 2 or 3 args")
            cfg = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._github_portfolio_report(str(args[0]), str(args[1]), cfg=cfg)
            return
        if step_name == "idea_forge_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("idea_forge_json(path[, theme]) expects 1 or 2 args")
            out = self._idea_forge_json(str(args[0]), theme=str(args[1]) if len(args) == 2 else None)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "idea_forge_json", "path": str(out)})
            return
        if step_name == "win_powershell":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("win_powershell(command[, timeout_sec]) expects 1 or 2 args")
            timeout_sec = float(args[1]) if len(args) == 2 else 30.0
            result = self._run_powershell(str(args[0]), timeout_sec=timeout_sec)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_powershell", "ok": result.get("ok")})
            return
        if step_name == "win_cmd":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("win_cmd(command[, timeout_sec]) expects 1 or 2 args")
            timeout_sec = float(args[1]) if len(args) == 2 else 30.0
            result = self._win_cmd(str(args[0]), timeout_sec=timeout_sec)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_cmd", "ok": result.get("ok")})
            return
        if step_name == "win_beep":
            if len(args) not in {0, 1, 2}:
                raise RuntimeErrorNF("win_beep([freq[, duration_ms]]) expects 0-2 args")
            freq = _as_int(args[0]) if len(args) >= 1 else 880
            dur = _as_int(args[1]) if len(args) >= 2 else 120
            result = self._win_beep(freq, dur)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_beep", "ok": result.get("ok")})
            return
        if step_name == "win_processes_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("win_processes_json(path[, top_n]) expects 1 or 2 args")
            top_n = _as_int(args[1]) if len(args) == 2 else 20
            out = self._win_processes_json(str(args[0]), top_n=top_n)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_processes_json", "path": str(out)})
            return
        if step_name == "win_services_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("win_services_json(path[, top_n]) expects 1 or 2 args")
            top_n = _as_int(args[1]) if len(args) == 2 else 50
            out = self._win_services_json(str(args[0]), top_n=top_n)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_services_json", "path": str(out)})
            return
        if step_name == "win_clipboard_set":
            if len(args) != 1:
                raise RuntimeErrorNF("win_clipboard_set(text) expects 1 arg")
            result = self._win_clipboard_set(str(args[0]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_clipboard_set", "ok": result.get("ok")})
            return
        if step_name == "win_notify":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("win_notify(title, message[, seconds]) expects 2 or 3 args")
            seconds = _as_int(args[2]) if len(args) == 3 else 4
            result = self._win_notify(str(args[0]), str(args[1]), seconds=max(1, seconds))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_notify", "ok": result.get("ok")})
            return
        if step_name == "win_open":
            if len(args) != 1:
                raise RuntimeErrorNF("win_open(target) expects 1 arg")
            result = self._win_open(str(args[0]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_open", "ok": result.get("ok")})
            return
        if step_name == "win_reveal":
            if len(args) != 1:
                raise RuntimeErrorNF("win_reveal(path) expects 1 arg")
            result = self._win_reveal(str(args[0]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_reveal", "ok": result.get("ok")})
            return
        if step_name == "win_windows_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("win_windows_json(path[, top_n]) expects 1 or 2 args")
            top_n = _as_int(args[1]) if len(args) == 2 else 20
            out = self._win_windows_json(str(args[0]), top_n=top_n)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_windows_json", "path": str(out)})
            return
        if step_name == "win_activate_window":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("win_activate_window(target[, timeout_sec]) expects 1 or 2 args")
            timeout_sec = float(args[1]) if len(args) == 2 else 5.0
            result = self._win_activate_window(args[0], timeout_sec=timeout_sec)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_activate_window", "ok": result.get("ok")})
            return
        if step_name == "win_mouse_move":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("win_mouse_move(x, y[, relative]) expects 2 or 3 args")
            relative = bool(args[2]) if len(args) == 3 else False
            result = self._win_mouse_move(_as_int(args[0]), _as_int(args[1]), relative=relative)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_mouse_move", "ok": result.get("ok")})
            return
        if step_name == "win_mouse_click":
            if len(args) not in {0, 1, 2, 3, 4}:
                raise RuntimeErrorNF("win_mouse_click([button[, clicks[, x[, y]]]]) expects 0-4 args")
            button = str(args[0]) if len(args) >= 1 else "left"
            clicks = _as_int(args[1]) if len(args) >= 2 else 1
            x = _as_int(args[2]) if len(args) >= 3 and args[2] is not None else None
            y = _as_int(args[3]) if len(args) >= 4 and args[3] is not None else None
            result = self._win_mouse_click(button=button, clicks=clicks, x=x, y=y)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_mouse_click", "ok": result.get("ok")})
            return
        if step_name == "win_mouse_scroll":
            if len(args) not in {0, 1}:
                raise RuntimeErrorNF("win_mouse_scroll([delta]) expects 0 or 1 args")
            delta = _as_int(args[0]) if len(args) == 1 else 120
            result = self._win_mouse_scroll(delta=delta)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_mouse_scroll", "ok": result.get("ok")})
            return
        if step_name in {"win_key_send", "win_key_chord"}:
            if len(args) != 1:
                raise RuntimeErrorNF(f"{step_name}(keys) expects 1 arg")
            result = self._win_key_send(str(args[0]))
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": step_name, "ok": result.get("ok")})
            return
        if step_name == "win_type_text":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("win_type_text(text[, interval_ms]) expects 1 or 2 args")
            interval_ms = _as_int(args[1]) if len(args) == 2 else 0
            result = self._win_type_text(str(args[0]), interval_ms=interval_ms)
            with self._lock:
                tick_now = self.tick_count
            self._append_event({"tick": tick_now, "event": "win_type_text", "ok": result.get("ok")})
            return
        if step_name == "win_input_sequence":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("win_input_sequence(actions[, dry_run]) expects 1 or 2 args")
            dry_run = bool(args[1]) if len(args) == 2 else False
            self._win_input_sequence(args[0], dry_run=dry_run)
            return
        if step_name == "vui_profile":
            if len(args) != 2 or not isinstance(args[1], dict):
                raise RuntimeErrorNF("vui_profile(name, config) expects 2 args with dict config")
            self._vui_profile_set(str(args[0]), args[1])
            return
        if step_name == "vui_say":
            if len(args) not in {1, 2, 3}:
                raise RuntimeErrorNF("vui_say(text[, profile_or_cfg[, cfg_or_dry_run]]) expects 1-3 args")
            profile_or_cfg = args[1] if len(args) >= 2 else None
            cfg_map: Optional[Dict[str, Any]] = None
            dry_run = False
            if len(args) == 3:
                if isinstance(args[2], dict):
                    cfg_map = args[2]
                else:
                    dry_run = bool(args[2])
            self._vui_say(str(args[0]), profile_or_cfg, cfg=cfg_map, dry_run=dry_run)
            return
        if step_name == "vui_log":
            if len(args) not in {2, 3}:
                raise RuntimeErrorNF("vui_log(role, text[, meta]) expects 2 or 3 args")
            meta = args[2] if len(args) == 3 and isinstance(args[2], dict) else None
            self._vui_log(str(args[0]), str(args[1]), meta=meta)
            return
        if step_name == "vui_voices_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("vui_voices_json(path[, refresh]) expects 1 or 2 args")
            refresh = bool(args[1]) if len(args) == 2 else False
            self._vui_voices_json(str(args[0]), refresh=refresh)
            return
        if step_name == "vui_export_json":
            if len(args) not in {1, 2}:
                raise RuntimeErrorNF("vui_export_json(path[, limit]) expects 1 or 2 args")
            limit = _as_int(args[1]) if len(args) == 2 else None
            self._vui_export_json(str(args[0]), limit=limit)
            return
        if step_name == "torch_train":
            if len(args) < 2:
                raise RuntimeErrorNF("torch_train(model, dataset[, epochs[, lr[, batch_size]]]) expects at least 2 args")
            epochs = _as_int(args[2]) if len(args) >= 3 else 1
            lr = float(args[3]) if len(args) >= 4 else 1e-3
            batch_size = _as_int(args[4]) if len(args) >= 5 else None
            self._torch_train(str(args[0]), str(args[1]), epochs, lr, batch_size)
            return
        if step_name == "torch_export":
            if len(args) != 2:
                raise RuntimeErrorNF("torch_export(model, path) expects 2 args")
            self._torch_export(str(args[0]), str(args[1]))
            return
        if step_name == "python_train":
            if len(args) not in {2, 3, 4}:
                raise RuntimeErrorNF("python_train(model, dataset[, epochs[, lr]]) expects 2-4 args")
            epochs = _as_int(args[2]) if len(args) >= 3 else 1
            lr = float(args[3]) if len(args) == 4 else 0.03
            self._python_train(str(args[0]), str(args[1]), epochs, lr)
            return
        if step_name == "python_export":
            if len(args) != 2:
                raise RuntimeErrorNF("python_export(model, path) expects 2 args")
            self._python_export(str(args[0]), str(args[1]))
            return
        if step_name == "train":
            if len(args) < 2:
                raise RuntimeErrorNF("train(model, dataset[, epochs]) expects at least 2 args")
            model_name = str(args[0])
            dataset_name = str(args[1])
            epochs = _as_int(args[2]) if len(args) >= 3 else 1
            backend = str(self.models.get(model_name, {}).get("backend", "")).lower()
            if backend == "pytorch":
                self._torch_train(model_name, dataset_name, epochs)
            elif backend in {"npu", "npu_torch", "npu_pytorch"}:
                self._npu_torch_train(model_name, dataset_name, epochs=epochs, cfg={"task": "training"})
            elif backend in {"python", "python_native", "py"}:
                py_lr = float(self.models.get(model_name, {}).get("lr", 0.03))
                self._python_train(model_name, dataset_name, epochs, py_lr)
            else:
                self._record_stub_train(model_name, dataset_name, epochs)
            return

        self._record_unsupported_step(step_name, args)


def run_program(path: str | Path, pipeline: Optional[str] = None, out_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    src_path = Path(path).resolve()
    project = parse_file(src_path)
    executor = Executor(project, source_path=src_path, out_dir=Path(out_dir) if out_dir else None)
    pipeline_result = executor.run_pipeline(pipeline) if project.pipelines else None
    return {"pipeline": pipeline_result, "snapshot": executor.snapshot()}

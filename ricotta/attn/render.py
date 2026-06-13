"""HTML assembly for notebook-inline or standalone rendering."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from .extract import AttentionData
from .serialize import serialize

_STATIC = Path(__file__).parent / "static"


def _build_html(payload: dict, height: int) -> str:
    js = (_STATIC / "headview.js").read_text()
    css = (_STATIC / "headview.css").read_text()
    div_id = "attnviz-" + uuid.uuid4().hex[:10]
    # "</" would terminate the script tag early if it ever appeared in a token
    data_json = json.dumps(payload).replace("</", "<\\/")
    return f"""
<style>{css}</style>
<div id="{div_id}"></div>
<script>
(function() {{
{js}
var data = {data_json};
var el = document.getElementById("{div_id}");
el.querySelectorAll && attnvizHeadView(el, data);
el.querySelector(".av-main").style.height = "{height}px";
}})();
</script>
"""


def head_view(
    data: AttentionData,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
    top_k: int = 32,
    height: int = 640,
    html_file: str | None = None,
):
    """Render the interactive head view.

    In a notebook this displays inline and returns the IPython HTML object.
    With ``html_file`` it also writes a self-contained standalone page.
    For long sequences pass a ``layers=[...]`` subset to keep the payload small.
    """
    payload = serialize(data, layers=layers, heads=heads, top_k=top_k)
    fragment = _build_html(payload, height)

    if html_file:
        page = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>attnviz · {data.model_name}</title></head>"
            f"<body style='background:#f4f4f8'>{fragment}</body></html>"
        )
        Path(html_file).write_text(page)

    try:
        from IPython.display import HTML, display

        obj = HTML(fragment)
        display(obj)
        return obj
    except ImportError:
        if not html_file:
            raise RuntimeError(
                "not in an IPython environment; pass html_file= to write a standalone page"
            ) from None
        return None

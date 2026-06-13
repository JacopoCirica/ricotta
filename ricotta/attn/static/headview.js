/* attnviz head view.
 *
 * Renders two virtualized token columns with attention edges drawn on a
 * canvas between them. Designed for long sequences: only visible rows get
 * DOM nodes, edges come from top-k sparsified data, and per-(layer,head)
 * arrays are decoded from base64 lazily and cached.
 *
 * Expects `data` in the format produced by attnviz.serialize.serialize().
 */
function attnvizHeadView(root, data) {
  "use strict";

  var ROW = 22;            // px per token row
  var COL_W = 150;         // token column width
  var CANVAS_W = 260;      // gap between columns where edges are drawn
  var MIN_DRAW = 0.02;     // never draw edges fainter than this

  var nLayers = data.layers.length;
  var nHeads = data.heads.length;
  var N = data.seqLen;
  var K = data.topK;

  // ---- decoding ----------------------------------------------------------
  var cache = {};
  function decode(li, hi) {
    var key = li + ":" + hi;
    if (cache[key]) return cache[key];
    var entry = data.attn[li][hi];
    var ib = atob(entry.i), vb = atob(entry.v);
    var idx = new Uint32Array(ib.length / 4);
    var dv = new DataView(new ArrayBuffer(ib.length));
    for (var b = 0; b < ib.length; b++) dv.setUint8(b, ib.charCodeAt(b));
    for (var u = 0; u < idx.length; u++) idx[u] = dv.getUint32(u * 4, true);
    var val = new Uint8Array(vb.length);
    for (var c = 0; c < vb.length; c++) val[c] = vb.charCodeAt(c);
    cache[key] = { idx: idx, val: val };
    return cache[key];
  }

  function headColor(hi) {
    var hue = (data.heads[hi] * 137.508) % 360;
    return "hsla(" + hue + ", 72%, 48%, ";
  }

  // ---- state -------------------------------------------------------------
  var state = {
    layer: 0,                       // index into data.layers
    enabled: data.heads.map(function () { return true; }),
    minW: 0.05,
    hover: null,                    // {side: 'q'|'k', i: tokenIdx} or null
    pinned: null
  };

  // ---- skeleton ----------------------------------------------------------
  root.classList.add("av-root");
  root.innerHTML =
    '<div class="av-toolbar">' +
    '  <span class="av-title">attnviz</span>' +
    '  <label>layer <select class="av-layer"></select></label>' +
    '  <div class="av-heads"></div>' +
    '  <button class="av-btn av-all">all</button>' +
    '  <button class="av-btn av-none">none</button>' +
    '  <label class="av-minw">min w <input type="range" min="0" max="50" value="5"></label>' +
    '  <span class="av-info"></span>' +
    '</div>' +
    '<div class="av-main">' +
    '  <div class="av-scroll">' +
    '    <div class="av-spacer"></div>' +
    '    <div class="av-tokens av-q"></div>' +
    '    <div class="av-tokens av-k"></div>' +
    '  </div>' +
    '  <canvas class="av-canvas"></canvas>' +
    '</div>';

  var $ = function (sel) { return root.querySelector(sel); };
  var scroll = $(".av-scroll"), spacer = $(".av-spacer");
  var qCol = $(".av-tokens.av-q"), kCol = $(".av-tokens.av-k");
  var canvas = $(".av-canvas"), ctx = canvas.getContext("2d");
  var info = $(".av-info");

  var totalW = COL_W * 2 + CANVAS_W;
  $(".av-main").style.width = totalW + "px";
  spacer.style.height = N * ROW + "px";
  kCol.style.left = COL_W + CANVAS_W + "px";

  // layer dropdown
  var layerSel = $(".av-layer");
  data.layers.forEach(function (l, li) {
    var o = document.createElement("option");
    o.value = li; o.textContent = l;
    layerSel.appendChild(o);
  });
  layerSel.addEventListener("change", function () {
    state.layer = +layerSel.value; render();
  });

  // head chips
  var chipBox = $(".av-heads");
  var chips = data.heads.map(function (h, hi) {
    var chip = document.createElement("span");
    chip.className = "av-chip";
    chip.textContent = h;
    chip.style.background = headColor(hi) + "1)";
    chip.addEventListener("click", function () {
      state.enabled[hi] = !state.enabled[hi]; syncChips(); render();
    });
    chipBox.appendChild(chip);
    return chip;
  });
  function syncChips() {
    chips.forEach(function (chip, hi) {
      chip.classList.toggle("av-off", !state.enabled[hi]);
    });
  }
  $(".av-all").addEventListener("click", function () {
    state.enabled = state.enabled.map(function () { return true; }); syncChips(); render();
  });
  $(".av-none").addEventListener("click", function () {
    state.enabled = state.enabled.map(function () { return false; }); syncChips(); render();
  });
  $(".av-minw input").addEventListener("input", function (e) {
    state.minW = e.target.value / 100; render();
  });

  // ---- virtualized token rows --------------------------------------------
  function visibleRange() {
    var top = scroll.scrollTop, h = scroll.clientHeight;
    var i0 = Math.max(0, Math.floor(top / ROW) - 2);
    var i1 = Math.min(N - 1, Math.ceil((top + h) / ROW) + 2);
    return [i0, i1];
  }

  function focusToken() { return state.pinned || state.hover; }

  // weights into / out of the focused token, for tinting the opposite column
  function focusWeights() {
    var f = focusToken();
    if (!f) return null;
    var out = new Float32Array(N);
    for (var hi = 0; hi < nHeads; hi++) {
      if (!state.enabled[hi]) continue;
      var d = decode(state.layer, hi);
      if (f.side === "q") {
        for (var t = 0; t < K; t++) {
          var v = d.val[f.i * K + t] / 255;
          if (v <= 0) break;
          var j = d.idx[f.i * K + t];
          if (v > out[j]) out[j] = v;
        }
      } else {
        for (var q = 0; q < N; q++) {
          for (var t2 = 0; t2 < K; t2++) {
            var v2 = d.val[q * K + t2] / 255;
            if (v2 <= 0) break;
            if (d.idx[q * K + t2] === f.i && v2 > out[q]) out[q] = v2;
          }
        }
      }
    }
    return out;
  }

  function makeRow(i, side) {
    var el = document.createElement("div");
    el.className = "av-tok";
    el.style.top = i * ROW + "px";
    el.textContent = data.tokens[i];
    el.title = i + ": " + data.tokens[i];
    el.addEventListener("mouseenter", function () {
      state.hover = { side: side, i: i }; render();
    });
    el.addEventListener("mouseleave", function () {
      state.hover = null; render();
    });
    el.addEventListener("click", function () {
      var f = state.pinned;
      state.pinned = (f && f.side === side && f.i === i) ? null : { side: side, i: i };
      render();
    });
    return el;
  }

  function renderTokens() {
    var r = visibleRange(), i0 = r[0], i1 = r[1];
    qCol.textContent = ""; kCol.textContent = "";
    var f = focusToken();
    var fw = focusWeights();
    for (var i = i0; i <= i1; i++) {
      var qe = makeRow(i, "q"), ke = makeRow(i, "k");
      if (f) {
        if (f.side === "q" && f.i === i) qe.classList.add("av-focus");
        if (f.side === "k" && f.i === i) ke.classList.add("av-focus");
        // tint the opposite column by attention weight
        var w = fw ? fw[i] : 0;
        if (w > 0.01) {
          var tint = "rgba(255, 140, 0, " + (0.15 + 0.6 * w).toFixed(3) + ")";
          if (f.side === "q") ke.style.background = tint;
          else qe.style.background = tint;
        }
      }
      qCol.appendChild(qe); kCol.appendChild(ke);
    }
  }

  // ---- canvas ------------------------------------------------------------
  function sizeCanvas() {
    var dpr = window.devicePixelRatio || 1;
    var h = scroll.clientHeight;
    canvas.width = CANVAS_W * dpr; canvas.height = h * dpr;
    canvas.style.width = CANVAS_W + "px"; canvas.style.height = h + "px";
    canvas.style.left = COL_W + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function edgeY(i) { return i * ROW + ROW / 2 - scroll.scrollTop; }

  function drawEdge(yq, yk, color, alpha) {
    ctx.strokeStyle = color + alpha.toFixed(3) + ")";
    ctx.lineWidth = 1 + alpha * 1.5;
    ctx.beginPath();
    ctx.moveTo(0, yq);
    ctx.bezierCurveTo(CANVAS_W * 0.4, yq, CANVAS_W * 0.6, yk, CANVAS_W, yk);
    ctx.stroke();
  }

  function drawEdgesForQuery(qi, minW) {
    for (var hi = 0; hi < nHeads; hi++) {
      if (!state.enabled[hi]) continue;
      var d = decode(state.layer, hi), color = headColor(hi);
      for (var t = 0; t < K; t++) {
        var v = d.val[qi * K + t] / 255;
        if (v < Math.max(minW, MIN_DRAW)) break;  // values sorted desc
        drawEdge(edgeY(qi), edgeY(d.idx[qi * K + t]), color, v);
      }
    }
  }

  function renderCanvas() {
    ctx.clearRect(0, 0, CANVAS_W, scroll.clientHeight);
    var f = focusToken();
    if (f && f.side === "q") {
      drawEdgesForQuery(f.i, 0);
    } else if (f && f.side === "k") {
      for (var hi = 0; hi < nHeads; hi++) {
        if (!state.enabled[hi]) continue;
        var d = decode(state.layer, hi), color = headColor(hi);
        for (var q = 0; q < N; q++) {
          for (var t = 0; t < K; t++) {
            var v = d.val[q * K + t] / 255;
            if (v <= 0) break;
            if (d.idx[q * K + t] === f.i) drawEdge(edgeY(q), edgeY(f.i), color, v);
          }
        }
      }
    } else {
      var r = visibleRange();
      for (var i = r[0]; i <= r[1]; i++) drawEdgesForQuery(i, state.minW);
    }
  }

  function renderInfo() {
    var f = focusToken();
    info.textContent = data.modelName + " · " + N + " tokens · layer " +
      data.layers[state.layer] +
      (f ? " · " + (f.side === "q" ? "from" : "into") + " [" + f.i + "] " +
        JSON.stringify(data.tokens[f.i]) : "");
  }

  var raf = null;
  function render() {
    if (raf) return;
    raf = requestAnimationFrame(function () {
      raf = null;
      renderTokens(); renderCanvas(); renderInfo();
    });
  }

  scroll.addEventListener("scroll", render);
  window.addEventListener("resize", function () { sizeCanvas(); render(); });

  sizeCanvas();
  syncChips();
  render();
}

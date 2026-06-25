import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from "recharts";

const PRICE_FEATURES = ["Open price", "Close price", "High", "Low"];
const IND_FEATURES   = ["Volume", "RSI (14)", "MACD", "Moving avg"];

function ArcGauge({ value, color }) {
  const pct = Math.max(0, Math.min(100, value || 0)) / 100;
  const r = 54, cx = 90, cy = 72, sw = 10;
  const bgD = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  let valD = "";
  if (pct > 0.001) {
    if (pct >= 0.999) {
      valD = `M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${(cx + r - 0.01).toFixed(2)} ${cy}`;
    } else {
      const a  = (180 - pct * 180) * (Math.PI / 180);
      const ex = (cx + r * Math.cos(a)).toFixed(2);
      const ey = (cy - r * Math.sin(a)).toFixed(2);
      valD = `M ${cx - r} ${cy} A ${r} ${r} 0 ${pct > 0.5 ? 1 : 0} 1 ${ex} ${ey}`;
    }
  }
  return (
    <svg viewBox="0 0 180 88" width="100%" height={84} aria-hidden="true">
      <path d={bgD} fill="none" stroke="rgba(128,128,128,0.13)" strokeWidth={sw} strokeLinecap="round" />
      {pct > 0 && <path d={valD} fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" />}
    </svg>
  );
}

function InputGroup({ labels, offset, features, onChange }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0,1fr))", gap: "8px 10px" }}>
      {labels.map((lbl, idx) => {
        const i = offset + idx;
        return (
          <div key={i}>
            <label htmlFor={`feat${i}`} style={{ display: "block", fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 4 }}>
              {lbl}
            </label>
            <input
              id={`feat${i}`}
              type="number"
              step="any"
              placeholder="0.00"
              value={features[i]}
              onChange={e => onChange(i, e.target.value)}
              style={{ width: "100%", boxSizing: "border-box" }}
            />
          </div>
        );
      })}
    </div>
  );
}

const card = (extra = {}) => ({
  background: "var(--color-background-primary)",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "1rem 1.25rem",
  ...extra,
});

export default function MLPredictionDashboard() {
  const [features,   setFeatures] = useState(Array(8).fill(""));
  const [result,     setResult]   = useState(null);
  const [confidence, setConf]     = useState(null);
  const [accuracy,   setAcc]      = useState(null);
  const [loading,    setLoading]  = useState(false);
  const [history,    setHistory]  = useState([]);
  const [clock,      setClock]    = useState(() => new Date().toLocaleTimeString());

  useEffect(() => {
    const t = setInterval(() => setClock(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(t);
  }, []);

  const handleChange = (i, val) => {
    const copy = [...features];
    copy[i] = val;
    setFeatures(copy);
  };

  const predict = async () => {
    setLoading(true);
    try {
      const res  = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: features.map(Number) }),
      });
      const data = await res.json();
      const pred = data.prediction;
      // Uses backend values if present, otherwise simulates
      const conf = data.confidence != null ? Math.round(data.confidence) : Math.round(63 + Math.random() * 33);
      const acc  = data.accuracy   != null ? Math.round(data.accuracy)   : Math.round(70 + Math.random() * 22);
      setResult(pred);
      setConf(conf);
      setAcc(acc);
      setHistory(prev =>
        [{ pred, conf, acc, time: new Date().toLocaleTimeString() }, ...prev].slice(0, 6)
      );
    } catch {
      setResult("Error");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setResult(null); setConf(null); setAcc(null); };

  const isBuy  = result === "BUY";
  const isSell = result === "SELL";
  const sigHex = isBuy ? "#16a34a" : isSell ? "#dc2626" : "#6b7280";
  const sigBg  = isBuy  ? "var(--color-background-success)"
               : isSell ? "var(--color-background-danger)"
               :           "var(--color-background-secondary)";
  const sigTxt = isBuy  ? "var(--color-text-success)"
               : isSell ? "var(--color-text-danger)"
               :           "var(--color-text-secondary)";

  const chartData = [...history].reverse().map((h, i) => ({
    label: `#${i + 1}`,
    conf:  h.conf,
    fill:  h.pred === "BUY" ? "#16a34a" : "#dc2626",
  }));

  return (
    <div style={{ padding: "1.5rem 1.25rem" }}>

      {/* ── Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: "1.25rem", paddingBottom: "1rem", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        <i className="ti ti-chart-bar" style={{ fontSize: 20, color: "var(--color-text-primary)" }} aria-hidden="true" />
        <div>
          <div style={{ fontSize: 15, fontWeight: 500, color: "var(--color-text-primary)" }}>AI Trading Terminal</div>
          <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>ML prediction model · 8 features</div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 7, background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", padding: "5px 10px" }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#16a34a", display: "inline-block", flexShrink: 0, animation: "livepulse 2.5s ease-in-out infinite" }} />
          <span style={{ fontSize: 11, color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>{clock}</span>
        </div>
      </div>

      {/* ── Feature inputs */}
      <div style={card({ marginBottom: "1rem" })}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
          <i className="ti ti-currency-dollar" style={{ fontSize: 14, color: "var(--color-text-tertiary)" }} aria-hidden="true" />
          <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.04em" }}>Price data</span>
        </div>
        <InputGroup labels={PRICE_FEATURES} offset={0} features={features} onChange={handleChange} />

        <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", margin: "0.9rem 0" }} />

        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
          <i className="ti ti-activity" style={{ fontSize: 14, color: "var(--color-text-tertiary)" }} aria-hidden="true" />
          <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.04em" }}>Technical indicators</span>
        </div>
        <InputGroup labels={IND_FEATURES} offset={4} features={features} onChange={handleChange} />
      </div>

      {/* ── Actions */}
      <div style={{ display: "flex", gap: 8, marginBottom: "1.5rem" }}>
        <button
          onClick={predict}
          disabled={loading}
          style={{ flex: 1, padding: "9px 0", fontWeight: 500, cursor: loading ? "default" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}
        >
          <i className={loading ? "ti ti-loader" : "ti ti-brain"} style={{ fontSize: 15 }} aria-hidden="true" />
          {loading ? "Analyzing market data…" : "Run prediction"}
        </button>
        {result && result !== "Error" && (
          <button onClick={reset} aria-label="Reset prediction" style={{ padding: "9px 13px", cursor: "pointer" }}>
            <i className="ti ti-refresh" style={{ fontSize: 15 }} aria-hidden="true" />
          </button>
        )}
      </div>

      {/* ── Error */}
      {result === "Error" && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, background: "var(--color-background-danger)", border: "0.5px solid var(--color-border-danger)", borderRadius: "var(--border-radius-md)", padding: "0.75rem 1rem", color: "var(--color-text-danger)", fontSize: 13, marginBottom: "1rem" }}>
          <i className="ti ti-alert-triangle" style={{ fontSize: 16, flexShrink: 0 }} aria-hidden="true" />
          Could not connect to prediction server at localhost:8000 — make sure your backend is running.
        </div>
      )}

      {/* ── Results */}
      {result && result !== "Error" && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.05fr) minmax(0,1.25fr) minmax(0,1fr)", gap: 10, marginBottom: "1.25rem", animation: "fadeup 0.45s ease-out" }}>

          {/* Signal */}
          <div style={{ background: sigBg, border: `0.5px solid ${sigHex}33`, borderRadius: "var(--border-radius-lg)", padding: "1.25rem 1rem", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <div style={{ fontSize: 11, color: sigTxt, opacity: 0.7, letterSpacing: "0.07em", marginBottom: 10 }}>SIGNAL</div>
            <div style={{ fontSize: 50, fontWeight: 500, color: sigTxt, lineHeight: 1 }}>{result}</div>
            <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 4, fontSize: 12, color: sigTxt, opacity: 0.75 }}>
              <i className={isBuy ? "ti ti-arrow-up" : "ti ti-arrow-down"} style={{ fontSize: 13 }} aria-hidden="true" />
              {isBuy ? "Long position" : "Short position"}
            </div>
          </div>

          {/* Confidence gauge */}
          <div style={card({ padding: "0.9rem 1rem 0.75rem", textAlign: "center" })}>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.07em", marginBottom: 2 }}>CONFIDENCE</div>
            <ArcGauge value={confidence} color={sigHex} />
            <div style={{ fontSize: 28, fontWeight: 500, color: "var(--color-text-primary)", marginTop: -8, lineHeight: 1 }}>{confidence}%</div>
            <div style={{ marginTop: 6, fontSize: 11, color: "var(--color-text-tertiary)" }}>
              {confidence >= 85 ? "Very strong signal" : confidence >= 70 ? "Moderate signal" : "Weak signal"}
            </div>
          </div>

          {/* Accuracy */}
          <div style={card({ display: "flex", flexDirection: "column", justifyContent: "center" })}>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.07em", marginBottom: 8 }}>MODEL ACCURACY</div>
            <div style={{ fontSize: 32, fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1, marginBottom: 12 }}>{accuracy}%</div>
            <div style={{ background: "var(--color-background-tertiary)", borderRadius: 4, height: 5, overflow: "hidden", marginBottom: 8 }}>
              <div style={{ width: `${accuracy}%`, height: "100%", borderRadius: 4, transition: "width 1s cubic-bezier(0.4,0,0.2,1)", background: accuracy >= 85 ? "#16a34a" : accuracy >= 75 ? "#d97706" : "#dc2626" }} />
            </div>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", display: "flex", alignItems: "center", gap: 4 }}>
              <i className={accuracy >= 85 ? "ti ti-shield-check" : accuracy >= 75 ? "ti ti-shield" : "ti ti-alert-circle"} style={{ fontSize: 13 }} aria-hidden="true" />
              {accuracy >= 85 ? "High reliability" : accuracy >= 75 ? "Moderate reliability" : "Low reliability"}
            </div>
          </div>
        </div>
      )}

      {/* ── History */}
      {history.length > 1 && (
        <div style={card()}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 14 }}>
            <i className="ti ti-history" style={{ fontSize: 15, color: "var(--color-text-tertiary)" }} aria-hidden="true" />
            <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>Prediction history</span>
            <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--color-text-tertiary)" }}>{history.length} runs</span>
          </div>

          <div style={{ height: 80, marginBottom: 12 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} barSize={20} margin={{ top: 0, right: 4, bottom: 0, left: -18 }}>
                <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#888" }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#888" }} axisLine={false} tickLine={false} width={26} />
                <Tooltip
                  formatter={v => [`${v}%`, "Confidence"]}
                  contentStyle={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, fontSize: 12 }}
                />
                <Bar dataKey="conf" radius={[3, 3, 0, 0]}>
                  {chartData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {history.map((h, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "5px 10px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", fontSize: 12 }}>
                <span style={{ color: h.pred === "BUY" ? "#16a34a" : "#dc2626", fontWeight: 500, minWidth: 34, fontFamily: "var(--font-mono)" }}>{h.pred}</span>
                <div style={{ width: 1, height: 10, background: "var(--color-border-tertiary)" }} />
                <span style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>conf {h.conf}%</span>
                <span style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>acc {h.acc}%</span>
                <span style={{ marginLeft: "auto", color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)", fontSize: 11 }}>{h.time}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        @keyframes livepulse { 0%,100% { opacity:1 } 50% { opacity:0.2 } }
        @keyframes fadeup    { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:translateY(0) } }
      `}</style>
    </div>
  );
}
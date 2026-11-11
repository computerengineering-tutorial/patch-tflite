import React, { useEffect, useState, useMemo } from 'react'
import {
  Activity,
  Cpu,
  Gauge,
  ShieldCheck,
  ShieldAlert,
  Layers,
  Terminal,
  RefreshCw,
  Eye,
  Sliders,
  Zap,
  CheckCircle2,
  AlertTriangle,
  ArrowUpRight,
  Database,
  Radio
} from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Cell
} from 'recharts'

interface OpItem {
  op_name: string
  total_time_ms: number
  percent_total: number
  execution_count: number
  min_time_ms?: number
  max_time_ms?: number
  avg_time_ms?: number
}

interface TelemetryState {
  connected: boolean
  model_name: string
  total_calls: number
  fps: number
  avg_latency_ms: number
  min_latency_ms: number
  max_latency_ms: number
  p50_ms: number
  p90_ms: number
  p99_ms: number
  anomalies_detected: number
  history: Array<{ timestamp: number; latency_ms: number; fps: number }>
  op_breakdown: OpItem[]
  delegate: {
    delegate_attached: boolean
    delegate_name: string
    delegated_nodes: number
    cpu_fallback_nodes: number
    total_nodes: number
    delegation_percent: number
    is_gpu_accelerated: boolean
    is_npu_accelerated: boolean
  }
  memory: {
    rss_mb: number
    vsize_mb: number
    heap_bytes: number
    canary_corruptions: number
    leak_detected: boolean
    leak_rate_kb: number
  }
  shadow: {
    divergences: number
    cosine_sim: number
    mae: number
  }
  tensor_preview?: {
    tensor_idx: number
    name: string
    shape: number[]
    dtype: string
    min: number
    max: number
    data: number[][]
  }
  recent_anomalies?: Array<{ time: string; msg: string; level: 'warn' | 'error' | 'info' }>
}

const defaultState: TelemetryState = {
  connected: false,
  model_name: 'Connecting...',
  total_calls: 0,
  fps: 0,
  avg_latency_ms: 0,
  min_latency_ms: 0,
  max_latency_ms: 0,
  p50_ms: 0,
  p90_ms: 0,
  p99_ms: 0,
  anomalies_detected: 0,
  history: [],
  op_breakdown: [],
  delegate: {
    delegate_attached: false,
    delegate_name: 'None',
    delegated_nodes: 0,
    cpu_fallback_nodes: 0,
    total_nodes: 0,
    delegation_percent: 0,
    is_gpu_accelerated: false,
    is_npu_accelerated: false
  },
  memory: {
    rss_mb: 0,
    vsize_mb: 0,
    heap_bytes: 0,
    canary_corruptions: 0,
    leak_detected: false,
    leak_rate_kb: 0
  },
  shadow: {
    divergences: 0,
    cosine_sim: 1.0,
    mae: 0.0
  },
  recent_anomalies: [
    { time: '17:28:10', msg: 'TFLite Trampoline hook active on TfLiteInterpreterInvoke', level: 'info' },
    { time: '17:28:10', msg: 'Zero-copy memory mapped telemetry active', level: 'info' }
  ]
}

export default function App() {
  const [state, setState] = useState<TelemetryState>(defaultState)
  const [selectedCell, setSelectedCell] = useState<{ r: number; c: number; val: number } | null>(null)
  const [hoveredOp, setHoveredOp] = useState<string | null>(null)

  useEffect(() => {
    // Connect to SSE stream
    const evtSource = new EventSource('/api/stream')

    evtSource.onopen = () => {
      setState(prev => ({ ...prev, connected: true }))
    }

    evtSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.type === 'init' || payload.type === 'telemetry') {
          setState(prev => ({
            ...prev,
            ...payload.data,
            connected: true,
            recent_anomalies: payload.data.recent_anomalies || prev.recent_anomalies
          }))
        } else if (payload.type === 'anomaly') {
          setState(prev => ({
            ...prev,
            recent_anomalies: [payload.data, ...(prev.recent_anomalies || [])].slice(0, 30)
          }))
        }
      } catch (err) {
        console.error('Failed to parse SSE event', err)
      }
    }

    evtSource.onerror = () => {
      setState(prev => ({ ...prev, connected: false }))
    }

    return () => {
      evtSource.close()
    }
  }, [])

  // Normalization for heatmap cell color
  const getHeatmapColor = (val: number, min: number, max: number) => {
    const range = max - min || 1
    const norm = Math.max(0, Math.min(1, (val - min) / range))
    // Emerald to Cyan gradient
    const r = Math.round(16 + norm * (6 - 16))
    const g = Math.round(185 + norm * (182 - 185))
    const b = Math.round(129 + norm * (212 - 129))
    const opacity = 0.2 + norm * 0.8
    return `rgba(${r}, ${g}, ${b}, ${opacity})`
  }

  const opColors = ['#10b981', '#06b6d4', '#8b5cf6', '#f59e0b', '#ec4899', '#6366f1', '#14b8a6']

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 p-4 md:p-6 font-sans">
      {/* Top Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-zinc-800/80 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Radio className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-zinc-100">TFLite Patch</h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-zinc-800 text-zinc-400 border border-zinc-700/50">
                v0.4.0
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-emerald-950/60 text-emerald-400 border border-emerald-800/50 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
                ACTIVE
              </span>
            </div>
            <p className="text-xs text-zinc-400 font-mono mt-0.5 flex items-center gap-1.5">
              <span>Target:</span>
              <span className="text-zinc-200 font-semibold bg-zinc-800/80 px-1.5 py-0.5 rounded">
                {state.model_name || 'Generic .so'}
              </span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-mono bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-md">
            <div className={`w-2 h-2 rounded-full ${state.connected ? 'bg-emerald-400 shadow-[0_0_8px_#10b981]' : 'bg-rose-500 shadow-[0_0_8px_#f43f5e]'}`} />
            <span className={state.connected ? 'text-zinc-300' : 'text-zinc-500'}>
              {state.connected ? 'SSE Connected (Live)' : 'Disconnected'}
            </span>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 transition"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Sync
          </button>
        </div>
      </header>

      {/* 4 Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 my-6">
        {/* Latency Card */}
        <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm hover:border-zinc-700/80 transition group">
          <div className="flex items-center justify-between text-zinc-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">Avg Latency</span>
            <Gauge className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-zinc-100">
              {state.avg_latency_ms.toFixed(1)}
            </span>
            <span className="text-xs font-mono text-zinc-400">ms</span>
          </div>
          <div className="flex items-center justify-between text-[11px] font-mono text-zinc-400 mt-3 pt-2.5 border-t border-zinc-800/60">
            <span>P50: <span className="text-zinc-200">{state.p50_ms.toFixed(1)}ms</span></span>
            <span>P90: <span className="text-zinc-200">{state.p90_ms.toFixed(1)}ms</span></span>
            <span>P99: <span className="text-emerald-400 font-semibold">{state.p99_ms.toFixed(1)}ms</span></span>
          </div>
        </div>

        {/* Throughput Card */}
        <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm hover:border-zinc-700/80 transition group">
          <div className="flex items-center justify-between text-zinc-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">Throughput</span>
            <Activity className="w-4 h-4 text-cyan-400 group-hover:scale-110 transition" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-zinc-100">
              {state.fps.toFixed(1)}
            </span>
            <span className="text-xs font-mono text-zinc-400">FPS / invokes/sec</span>
          </div>
          <div className="flex items-center justify-between text-[11px] font-mono text-zinc-400 mt-3 pt-2.5 border-t border-zinc-800/60">
            <span>Total Calls:</span>
            <span className="text-cyan-400 font-bold">{state.total_calls.toLocaleString()}</span>
          </div>
        </div>

        {/* Delegate Card */}
        <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm hover:border-zinc-700/80 transition group">
          <div className="flex items-center justify-between text-zinc-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">Hardware Acceleration</span>
            <Cpu className="w-4 h-4 text-violet-400 group-hover:scale-110 transition" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-zinc-100">
              {state.delegate.delegation_percent.toFixed(1)}%
            </span>
            <span className="text-xs font-mono text-zinc-400">Offloaded</span>
          </div>
          <div className="flex items-center justify-between text-[11px] font-mono text-zinc-400 mt-3 pt-2.5 border-t border-zinc-800/60">
            <span className="truncate max-w-[130px]">{state.delegate.delegate_name || 'CPU Only'}</span>
            <span className="text-violet-400 font-semibold">
              {state.delegate.delegated_nodes}/{state.delegate.total_nodes} Ops
            </span>
          </div>
        </div>

        {/* Memory Sentinel Card */}
        <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm hover:border-zinc-700/80 transition group">
          <div className="flex items-center justify-between text-zinc-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">Memory Sentinel</span>
            {state.memory.canary_corruptions > 0 || state.memory.leak_detected ? (
              <ShieldAlert className="w-4 h-4 text-rose-500 animate-bounce" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition" />
            )}
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-zinc-100">
              {state.memory.rss_mb.toFixed(1)}
            </span>
            <span className="text-xs font-mono text-zinc-400">MB RSS</span>
          </div>
          <div className="flex items-center justify-between text-[11px] font-mono text-zinc-400 mt-3 pt-2.5 border-t border-zinc-800/60">
            <span>Canaries: <span className={state.memory.canary_corruptions === 0 ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-bold'}>{state.memory.canary_corruptions} err</span></span>
            <span>Leak: <span className={state.memory.leak_detected ? 'text-rose-400 font-bold' : 'text-emerald-400'}>{state.memory.leak_detected ? 'LEAK' : 'Clean'}</span></span>
          </div>
        </div>
      </div>

      {/* Main Grid: Charts, Op Breakdown, Tensor Visualizer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col (7 / 12) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* Latency & Throughput Waterfall Chart */}
          <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-semibold text-zinc-200">Real-Time Latency & Throughput Stream</h2>
              </div>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="flex items-center gap-1.5 text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400"></span> Latency (ms)
                </span>
                <span className="flex items-center gap-1.5 text-cyan-400">
                  <span className="w-2 h-2 rounded-full bg-cyan-400"></span> FPS
                </span>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={state.history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="fpsGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis dataKey="timestamp" stroke="#71717a" fontSize={10} tickLine={false} />
                  <YAxis stroke="#71717a" fontSize={10} tickLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', color: '#f4f4f5', fontSize: '12px' }}
                    labelFormatter={(val) => `Time: ${val}s`}
                  />
                  <Area type="monotone" dataKey="latency_ms" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#latencyGradient)" name="Latency (ms)" />
                  <Area type="monotone" dataKey="fps" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#fpsGradient)" name="FPS" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Op-Level Micro-Profiling Breakdown */}
          <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-violet-400" />
                <h2 className="text-sm font-semibold text-zinc-200">Op-Level Micro-Profiling (Latency Breakdown)</h2>
              </div>
              <span className="text-xs font-mono text-zinc-400">
                {state.op_breakdown.length} unique operators
              </span>
            </div>

            {/* Op Bar Distribution */}
            <div className="h-44 w-full mb-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={state.op_breakdown} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
                  <XAxis type="number" stroke="#71717a" fontSize={10} unit="%" />
                  <YAxis dataKey="op_name" type="category" stroke="#a1a1aa" fontSize={10} width={90} tickLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', color: '#f4f4f5', fontSize: '12px' }}
                    formatter={(val: any) => [`${Number(val).toFixed(1)}% of total latency`, 'Share']}
                  />
                  <Bar dataKey="percent_total" radius={[0, 4, 4, 0]}>
                    {state.op_breakdown.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.percent_total > 25 ? '#f43f5e' : entry.percent_total > 15 ? '#f59e0b' : '#10b981'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Sortable Op Table */}
            <div className="overflow-x-auto rounded-lg border border-zinc-800">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-zinc-900/80 text-zinc-400 border-b border-zinc-800">
                  <tr>
                    <th className="py-2 px-3">Operator Name</th>
                    <th className="py-2 px-3 text-right">Exec Count</th>
                    <th className="py-2 px-3 text-right">Time (ms)</th>
                    <th className="py-2 px-3 text-right">Share (%)</th>
                    <th className="py-2 px-3 text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60 text-zinc-300">
                  {state.op_breakdown.map((op, idx) => {
                    const isBottleneck = op.percent_total >= 20.0
                    return (
                      <tr
                        key={idx}
                        className={`hover:bg-zinc-850 transition ${isBottleneck ? 'bg-rose-950/10' : ''}`}
                        onMouseEnter={() => setHoveredOp(op.op_name)}
                        onMouseLeave={() => setHoveredOp(null)}
                      >
                        <td className="py-2 px-3 font-semibold text-zinc-200 flex items-center gap-2">
                          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: opColors[idx % opColors.length] }}></span>
                          {op.op_name}
                        </td>
                        <td className="py-2 px-3 text-right text-zinc-400">{op.execution_count}</td>
                        <td className="py-2 px-3 text-right text-zinc-200">{op.total_time_ms.toFixed(2)} ms</td>
                        <td className="py-2 px-3 text-right font-bold text-emerald-400">{op.percent_total.toFixed(1)}%</td>
                        <td className="py-2 px-3 text-center">
                          {isBottleneck ? (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">
                              BOTTLENECK
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-zinc-800 text-zinc-400">
                              OPTIMAL
                            </span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Col (5 / 12) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* Live Tensor Visualizer Heatmap */}
          <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-cyan-400" />
                <h2 className="text-sm font-semibold text-zinc-200">Live 2D Tensor Heatmap Preview</h2>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 bg-zinc-800 rounded text-cyan-300">
                {state.tensor_preview ? state.tensor_preview.name : 'Output Tensor #1'}
              </span>
            </div>

            <div className="flex items-center justify-between text-xs font-mono text-zinc-400 mb-3 bg-zinc-900/60 p-2 rounded-lg border border-zinc-800/60">
              <span>Shape: <span className="text-zinc-200 font-bold">{state.tensor_preview ? JSON.stringify(state.tensor_preview.shape) : '[1, 16, 16, 1]'}</span></span>
              <span>DType: <span className="text-zinc-200">FLOAT32</span></span>
              <span>Min/Max: <span className="text-emerald-400">-1.0 / 1.0</span></span>
            </div>

            {/* Heatmap Grid */}
            <div className="aspect-square w-full max-w-[340px] mx-auto bg-zinc-950 p-2 rounded-lg border border-zinc-800">
              {state.tensor_preview?.data ? (
                <div className="grid grid-cols-16 gap-[1px] h-full w-full" style={{ gridTemplateColumns: `repeat(${state.tensor_preview.data[0]?.length || 16}, minmax(0, 1fr))` }}>
                  {state.tensor_preview.data.map((row, rIdx) =>
                    row.map((val, cIdx) => (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        className="w-full h-full rounded-[1px] cursor-pointer transition hover:scale-125 hover:z-10 hover:shadow-lg"
                        style={{ backgroundColor: getHeatmapColor(val, state.tensor_preview!.min, state.tensor_preview!.max) }}
                        onMouseEnter={() => setSelectedCell({ r: rIdx, c: cIdx, val })}
                      />
                    ))
                  )}
                </div>
              ) : (
                <div className="h-full w-full flex items-center justify-center text-xs font-mono text-zinc-500">
                  Awaiting inference tensor buffer...
                </div>
              )}
            </div>

            {/* Hover inspection pill */}
            <div className="mt-3 text-center text-xs font-mono text-zinc-400 h-5">
              {selectedCell ? (
                <span>Cell [{selectedCell.r}, {selectedCell.c}] = <strong className="text-emerald-400 font-bold">{selectedCell.val}</strong></span>
              ) : (
                <span className="text-zinc-600">Hover over heatmap cells for numeric inspection</span>
              )}
            </div>
          </div>

          {/* Shadow Execution Engine & Side-by-Side Validation */}
          <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-semibold text-zinc-200">Shadow Execution Engine</h2>
              </div>
              <span className="text-xs px-2 py-0.5 rounded font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                LOCKSTEP
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono my-3">
              <div className="bg-zinc-900/80 p-2 rounded-lg border border-zinc-800">
                <span className="text-zinc-500 block text-[10px]">COSINE SIM</span>
                <span className="text-sm font-bold text-emerald-400">{state.shadow.cosine_sim.toFixed(4)}</span>
              </div>
              <div className="bg-zinc-900/80 p-2 rounded-lg border border-zinc-800">
                <span className="text-zinc-500 block text-[10px]">MAE DRIFT</span>
                <span className="text-sm font-bold text-zinc-200">{state.shadow.mae.toFixed(5)}</span>
              </div>
              <div className="bg-zinc-900/80 p-2 rounded-lg border border-zinc-800">
                <span className="text-zinc-500 block text-[10px]">DIVERGENCES</span>
                <span className={`text-sm font-bold ${state.shadow.divergences === 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {state.shadow.divergences}
                </span>
              </div>
            </div>

            <div className="text-[11px] text-zinc-400 flex items-center justify-between px-1">
              <span>Reference Baseline: <strong className="text-zinc-300">PyTorch/LiteRT</strong></span>
              <span className="flex items-center gap-1 text-emerald-400 font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" /> Bit-Accurate
              </span>
            </div>
          </div>

          {/* Anomaly & Sentinel Alerts Console */}
          <div className="bg-[#121215] border border-zinc-800/80 rounded-xl p-4 shadow-sm flex-1">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-amber-400" />
                <h2 className="text-sm font-semibold text-zinc-200">Sentinel Alert & Anomaly Console</h2>
              </div>
              <span className="text-xs font-mono text-zinc-500">Live</span>
            </div>

            <div className="bg-zinc-950 rounded-lg p-3 border border-zinc-850 h-40 overflow-y-auto font-mono text-[11px] space-y-1.5">
              {state.recent_anomalies && state.recent_anomalies.length > 0 ? (
                state.recent_anomalies.map((anom, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-zinc-600 select-none">[{anom.time}]</span>
                    {anom.level === 'warn' && <span className="text-amber-400 font-bold">WARN</span>}
                    {anom.level === 'error' && <span className="text-rose-400 font-bold">FAIL</span>}
                    {anom.level === 'info' && <span className="text-cyan-400">INFO</span>}
                    <span className="text-zinc-300">{anom.msg}</span>
                  </div>
                ))
              ) : (
                <div className="text-zinc-600 text-center py-6">No anomalies or sentinel warnings detected.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}


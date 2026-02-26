import { createChart, ColorType, type IChartApi, type ISeriesApi } from 'lightweight-charts'
import { useCallback, useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/api/v1/stream'
const API_BASE = 'http://localhost:8000'

/** Wire format: t (unix s), p (p_hat), lo, hi, m (market_mid or null), seg (segment_id) */
type Point = { t: number; p: number; lo: number; hi: number; m: number | null; seg?: number }

/** BO3 live match from /api/v1/bo3/live_matches */
type Bo3Match = { id: number; team1_name: string; team2_name: string; bo_type: number }

function App() {
  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>('connecting')
  const [current, setCurrent] = useState<{ state?: unknown; derived?: { p_hat?: number } } | null>(null)
  const [snapshotHistory, setSnapshotHistory] = useState<Point[]>([])
  const [chartReady, setChartReady] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [wsReconnectTrigger, setWsReconnectTrigger] = useState(0)
  const [liveMatches, setLiveMatches] = useState<Bo3Match[]>([])
  const [selectedMatchId, setSelectedMatchId] = useState<string>('')
  const [teamAIsTeamOne, setTeamAIsTeamOne] = useState(true)
  const [configError, setConfigError] = useState<string | null>(null)

  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<IChartApi | null>(null)
  const pSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const loSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const hiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const pausedRef = useRef(false)
  const pendingPointsRef = useRef<Point[]>([])
  const currentSegRef = useRef<number>(0)

  useEffect(() => {
    pausedRef.current = isPaused
  }, [isPaused])

  const applyPointToChart = useCallback((point: Point) => {
    const time = point.t as import('lightweight-charts').UTCTimestamp
    pSeriesRef.current?.update({ time, value: point.p })
    loSeriesRef.current?.update({ time, value: point.lo })
    hiSeriesRef.current?.update({ time, value: point.hi })
  }, [])

  const setDataFromHistory = useCallback((history: Point[]) => {
    if (!pSeriesRef.current || !loSeriesRef.current || !hiSeriesRef.current) return
    if (history.length === 0) {
      pSeriesRef.current.setData([])
      loSeriesRef.current.setData([])
      hiSeriesRef.current.setData([])
      return
    }
    const pData = history.map((pt) => ({ time: pt.t as import('lightweight-charts').UTCTimestamp, value: pt.p }))
    const loData = history.map((pt) => ({ time: pt.t as import('lightweight-charts').UTCTimestamp, value: pt.lo }))
    const hiData = history.map((pt) => ({ time: pt.t as import('lightweight-charts').UTCTimestamp, value: pt.hi }))
    pSeriesRef.current.setData(pData)
    loSeriesRef.current.setData(loData)
    hiSeriesRef.current.setData(hiData)
    chartInstanceRef.current?.timeScale().fitContent()
  }, [])

  // Flush pending points when chart becomes ready
  useEffect(() => {
    if (!chartReady || !pSeriesRef.current || pendingPointsRef.current.length === 0) return
    const pending = pendingPointsRef.current
    pending.forEach((pt) => applyPointToChart(pt))
    pendingPointsRef.current = []
  }, [chartReady, applyPointToChart])

  // Apply snapshot history when chart is ready and we have history
  useEffect(() => {
    if (snapshotHistory.length === 0 || !chartReady || !pSeriesRef.current) return
    setDataFromHistory(snapshotHistory)
  }, [snapshotHistory, chartReady, setDataFromHistory])

  // Chart + series created exactly once
  useEffect(() => {
    if (!chartRef.current) return
    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1a1a1a' },
        textColor: '#d1d5db',
      },
      width: chartRef.current.clientWidth,
      height: 300,
    })
    chartInstanceRef.current = chart
    pSeriesRef.current = chart.addLineSeries({ color: '#3b82f6', title: 'p_hat' })
    loSeriesRef.current = chart.addLineSeries({ color: '#22c55e', lineWidth: 1, title: 'bound_low' })
    hiSeriesRef.current = chart.addLineSeries({ color: '#ef4444', lineWidth: 1, title: 'bound_high' })
    setChartReady(true)

    const handleResize = () => {
      if (chartRef.current && chartInstanceRef.current)
        chartInstanceRef.current.applyOptions({ width: chartRef.current!.clientWidth })
    }
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartInstanceRef.current = null
      pSeriesRef.current = null
      loSeriesRef.current = null
      hiSeriesRef.current = null
      setChartReady(false)
    }
  }, [])

  // WebSocket: only reconnect when wsReconnectTrigger changes
  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    ws.onopen = () => setWsStatus('open')
    ws.onclose = () => setWsStatus('closed')
    ws.onerror = () => setWsStatus('error')
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string)
        if (msg.type === 'snapshot') {
          setCurrent(msg.current ?? null)
          const seg = (msg.current as { state?: { segment_id?: number } })?.state?.segment_id ?? 0
          currentSegRef.current = seg
          const hist = Array.isArray(msg.history) ? (msg.history as Point[]) : []
          const lastSegmentOnly = hist.filter((p) => p.seg === undefined || p.seg === seg)
          setSnapshotHistory(lastSegmentOnly)
          if (pSeriesRef.current && lastSegmentOnly.length > 0) setDataFromHistory(lastSegmentOnly)
        } else if (msg.type === 'point' && msg.point) {
          const pt = msg.point as Point
          setCurrent((prev) => ({ ...prev, state: prev?.state, derived: { ...prev?.derived, p_hat: pt.p } }))
          if (!pSeriesRef.current) {
            pendingPointsRef.current.push(pt)
            return
          }
          if (!pausedRef.current) {
            if (pt.seg !== undefined && pt.seg !== currentSegRef.current) {
              currentSegRef.current = pt.seg
              setDataFromHistory([])
              setSnapshotHistory([])
            }
            applyPointToChart(pt)
          }
        }
      } catch {
        // ignore parse errors
      }
    }
    return () => ws.close()
  }, [wsReconnectTrigger])

  return (
    <div style={{ padding: 16, maxWidth: 900, margin: '0 auto' }}>
      <h1>ESports Fair Odds</h1>
      <p>
        <strong>Connection:</strong>{' '}
        <span
          style={{
            color:
              wsStatus === 'open' ? 'green' : wsStatus === 'error' ? 'red' : wsStatus === 'closed' ? 'orange' : 'gray',
          }}
        >
          {wsStatus}
        </span>
        {' — '}
        <code>{WS_URL}</code>
        {' · '}
        <label>
          <input type="checkbox" checked={isPaused} onChange={(e) => setIsPaused(e.target.checked)} /> Pause
        </label>
        {' '}
        <button type="button" onClick={() => { setIsPaused(false); setWsReconnectTrigger((n) => n + 1) }}>
          Resume (catch up)
        </button>
      </p>
      {current?.derived?.p_hat != null && (
        <p style={{ fontSize: 14, color: '#9ca3af' }}>
          Current p_hat: <strong>{current.derived.p_hat.toFixed(4)}</strong>
        </p>
      )}
      <section style={{ marginTop: 16, padding: 12, border: '1px solid #374151', borderRadius: 4 }}>
        <h3 style={{ marginTop: 0 }}>BO3</h3>
        <p>
          <button
            type="button"
            onClick={async () => {
              try {
                const r = await fetch(`${API_BASE}/api/v1/bo3/live_matches`)
                const data = await r.json()
                setLiveMatches(Array.isArray(data) ? data : [])
              } catch {
                setLiveMatches([])
              }
            }}
          >
            Load BO3 live matches
          </button>
        </p>
        <p>
          <label>
            Match:{' '}
            <select
              value={selectedMatchId}
              onChange={(e) => setSelectedMatchId(e.target.value)}
              style={{ minWidth: 200 }}
            >
              <option value="">—</option>
              {liveMatches.map((m) => (
                <option key={m.id} value={String(m.id)}>
                  {m.team1_name} vs {m.team2_name} (bo{m.bo_type})
                </option>
              ))}
            </select>
          </label>
        </p>
        <p>
          <label>
            Team A is:{' '}
            <select
              value={teamAIsTeamOne ? 'team1' : 'team2'}
              onChange={(e) => setTeamAIsTeamOne(e.target.value === 'team1')}
            >
              <option value="team1">Team 1</option>
              <option value="team2">Team 2</option>
            </select>
          </label>
        </p>
        <p>
          <button
            type="button"
            disabled={
              !selectedMatchId ||
              !Number.isFinite(Number(selectedMatchId)) ||
              !Number.isInteger(Number(selectedMatchId))
            }
            onClick={async () => {
              if (!selectedMatchId) return
              const n = Number(selectedMatchId)
              if (!Number.isFinite(n) || !Number.isInteger(n)) return
              setConfigError(null)
              try {
                const r = await fetch(`${API_BASE}/api/v1/config`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    source: 'BO3',
                    match_id: n,
                    team_a_is_team_one: teamAIsTeamOne,
                  }),
                })
                if (!r.ok) {
                  const body = await r.json().catch(() => ({}))
                  const msg =
                    (body as { detail?: string })?.detail ??
                    (body as { message?: string })?.message ??
                    r.statusText
                  setConfigError(String(msg))
                  return
                }
                setConfigError(null)
                setWsReconnectTrigger((prev) => prev + 1)
              } catch (e) {
                setConfigError(e instanceof Error ? e.message : String(e))
              }
            }}
          >
            Activate BO3
          </button>
        </p>
        {configError && (
          <p style={{ color: '#ef4444', fontSize: 14 }}>{configError}</p>
        )}
        {current?.state && (
          <p style={{ fontSize: 14, color: '#9ca3af' }}>
            Source: <strong>{(current.state as { config?: { source?: string } })?.config?.source ?? '—'}</strong>
            {' · '}
            Match: <strong>{(current.state as { config?: { match_id?: number | null } })?.config?.match_id ?? '—'}</strong>
            {(current.state as { last_frame?: { teams?: string[]; scores?: number[] } })?.last_frame?.teams && (
              <>
                {' · '}
                {(current.state as { last_frame?: { teams?: string[] } }).last_frame?.teams?.join(' vs ')}
                {' '}
                ({(current.state as { last_frame?: { scores?: number[] } }).last_frame?.scores?.[0] ?? 0}
                –
                {(current.state as { last_frame?: { scores?: number[] } }).last_frame?.scores?.[1] ?? 0})
              </>
            )}
          </p>
        )}
      </section>
      <div ref={chartRef} style={{ marginTop: 16 }} />
    </div>
  )
}

export default App

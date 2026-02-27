import { createChart, ColorType, LineStyle, type IChartApi, type ISeriesApi } from 'lightweight-charts'
import { useCallback, useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/api/v1/stream'
const API_BASE = 'http://localhost:8000'

/** Wire format:
 * - t (unix s), p (p_hat)
 * - series_low/series_high: SERIES corridor (preferred)
 * - map_low/map_high: MAP corridor (preferred)
 * - lo/hi: legacy series corridor
 * - rail_low/rail_high: legacy map corridor
 * - m (market_mid or null), seg (segment_id)
 */
type Point = {
  t: number
  p: number
  lo: number
  hi: number
  m: number | null
  seg?: number
  rail_low?: number
  rail_high?: number
  series_low?: number
  series_high?: number
  map_low?: number
  map_high?: number
}

/** BO3 live match from /api/v1/bo3/live_matches */
type Bo3Match = { id: number; team1_name: string; team2_name: string; bo_type: number }

const filterHistoryToSeg = (history: Point[] | undefined, seg: number): Point[] => {
  if (!history || history.length === 0) return []
  const segValues = history
    .map((p) => p.seg)
    .filter((v): v is number => v !== undefined && v !== null)
  if (segValues.length === 0) return history
  const direct = history.filter((p) => p.seg === seg)
  if (direct.length > 0) return direct
  const latestSeg = Math.max(...segValues)
  const latest = history.filter((p) => p.seg === latestSeg)
  if (latest.length > 0) return latest
  return history
}

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
  const [replayPath, setReplayPath] = useState('logs/bo3_pulls.jsonl')
  const [replayMatchId, setReplayMatchId] = useState('')
  const [replaySpeed, setReplaySpeed] = useState(1)
  const [replayLoop, setReplayLoop] = useState(true)
  const [replayError, setReplayError] = useState<string | null>(null)
  const [replayMatches, setReplayMatches] = useState<Array<{ match_id: number; team1: string; team2: string; count: number }>>([])
  const [crosshairT, setCrosshairT] = useState<string | number | null>(null)
  const [kalshiUrl, setKalshiUrl] = useState('')
  const [marketOptions, setMarketOptions] = useState<Array<{ key: string; label: string; ticker_yes: string }>>([])
  const [selectedMarketKey, setSelectedMarketKey] = useState('')
  const [marketError, setMarketError] = useState<string | null>(null)
  const [prematchSeriesInput, setPrematchSeriesInput] = useState('')
  const [rawSnapshotJson, setRawSnapshotJson] = useState<string | null>(null)
  const [rawSnapshotError, setRawSnapshotError] = useState<string | null>(null)
  const [rawSnapshotOpen, setRawSnapshotOpen] = useState(false)
  const [breaches, setBreaches] = useState<Array<{
    ts_epoch: number
    match_id: number | null
    seg: number
    scores: number[]
    series_score: number[]
    map_index: number
    market_mid: number | null
    p_hat: number
    series_low: number
    series_high: number
    map_low: number
    map_high: number
    breach_type: string
    breach_mag: number | null
  }>>([])

  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<IChartApi | null>(null)
  const pSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const loSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const hiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const railLoSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const railHiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const marketSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const pausedRef = useRef(false)
  const pendingPointsRef = useRef<Point[]>([])
  const currentSegRef = useRef<number>(0)

  useEffect(() => {
    pausedRef.current = isPaused
  }, [isPaused])

  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/market/breaches?limit=100`)
        if (!r.ok) return
        const data = await r.json()
        setBreaches(Array.isArray(data) ? data : [])
      } catch {
        // ignore
      }
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
  }, [])

  const applyPointToChart = useCallback((point: Point) => {
    const time = point.t as any
    const seriesLo = point.series_low ?? point.lo
    const seriesHi = point.series_high ?? point.hi
    const mapLo = point.map_low ?? point.rail_low ?? seriesLo
    const mapHi = point.map_high ?? point.rail_high ?? seriesHi
    pSeriesRef.current?.update({ time, value: point.p })
    loSeriesRef.current?.update({ time, value: seriesLo })
    hiSeriesRef.current?.update({ time, value: seriesHi })
    railLoSeriesRef.current?.update({ time, value: mapLo })
    railHiSeriesRef.current?.update({ time, value: mapHi })
    if (point.m != null && marketSeriesRef.current) {
      marketSeriesRef.current.update({ time, value: point.m })
    }
  }, [])

  const setDataFromHistory = useCallback((history: Point[]) => {
    if (!pSeriesRef.current || !loSeriesRef.current || !hiSeriesRef.current) return
    if (history.length === 0) {
      pSeriesRef.current.setData([])
      loSeriesRef.current.setData([])
      hiSeriesRef.current.setData([])
      railLoSeriesRef.current?.setData([])
      railHiSeriesRef.current?.setData([])
      marketSeriesRef.current?.setData([])
      return
    }
    const utc = (t: number) => t as import('lightweight-charts').UTCTimestamp
    const pData = history.map((pt) => ({ time: utc(pt.t), value: pt.p }))
    const loData = history.map((pt) => ({
      time: utc(pt.t),
      value: pt.series_low ?? pt.lo,
    }))
    const hiData = history.map((pt) => ({
      time: utc(pt.t),
      value: pt.series_high ?? pt.hi,
    }))
    const railLoData = history.map((pt) => {
      const seriesLo = pt.series_low ?? pt.lo
      const mapLo = pt.map_low ?? pt.rail_low ?? seriesLo
      return { time: utc(pt.t), value: mapLo }
    })
    const railHiData = history.map((pt) => {
      const seriesHi = pt.series_high ?? pt.hi
      const mapHi = pt.map_high ?? pt.rail_high ?? seriesHi
      return { time: utc(pt.t), value: mapHi }
    })
    const marketData = history.filter((pt) => pt.m != null).map((pt) => ({ time: utc(pt.t), value: pt.m as number }))
    pSeriesRef.current.setData(pData)
    loSeriesRef.current.setData(loData)
    hiSeriesRef.current.setData(hiData)
    railLoSeriesRef.current?.setData(railLoData)
    railHiSeriesRef.current?.setData(railHiData)
    marketSeriesRef.current?.setData(marketData)
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
      grid: {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
      width: chartRef.current.clientWidth,
      height: chartRef.current.clientHeight || 300,
    })
    chartInstanceRef.current = chart
    pSeriesRef.current = chart.addLineSeries({
      color: '#3b82f6',
      title: 'p_hat',
      priceLineVisible: false,
      lastValueVisible: true,
    })
    loSeriesRef.current = chart.addLineSeries({
      color: '#22c55e',
      lineWidth: 1,
      title: 'series_low',
      priceLineVisible: false,
      lastValueVisible: false,
    })
    hiSeriesRef.current = chart.addLineSeries({
      color: '#ef4444',
      lineWidth: 1,
      title: 'series_high',
      priceLineVisible: false,
      lastValueVisible: false,
    })
    railLoSeriesRef.current = chart.addLineSeries({
      color: '#facc15',
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      title: 'map_low',
      priceLineVisible: false,
      lastValueVisible: false,
    })
    railHiSeriesRef.current = chart.addLineSeries({
      color: '#facc15',
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      title: 'map_high',
      priceLineVisible: false,
      lastValueVisible: false,
    })
    marketSeriesRef.current = chart.addLineSeries({
      color: '#a78bfa',
      lineWidth: 1,
      title: 'market_mid',
      priceLineVisible: false,
      lastValueVisible: false,
    })
    setChartReady(true)

    const unsubCrosshair = chart.subscribeCrosshairMove((param) => {
      if (param.time === undefined) {
        setCrosshairT(null)
        return
      }
      if (typeof param.time === 'number') {
        setCrosshairT(param.time)
        return
      }
      if (typeof param.time === 'object' && param.time !== null) {
        setCrosshairT(JSON.stringify(param.time))
        return
      }
      setCrosshairT(String(param.time))
    })

    const handleResize = () => {
      if (chartRef.current && chartInstanceRef.current)
        chartInstanceRef.current.applyOptions({
          width: chartRef.current.clientWidth,
          height: chartRef.current.clientHeight || 300,
        })
    }
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartInstanceRef.current = null
      pSeriesRef.current = null
      loSeriesRef.current = null
      hiSeriesRef.current = null
      railLoSeriesRef.current = null
      railHiSeriesRef.current = null
      setChartReady(false)
    }
  }, [])

  const refreshSegmentFromBackend = useCallback(
    async (newSeg: number) => {
      try {
        const oldSeg = currentSegRef.current
        // Temporary debug for seg transitions
        // eslint-disable-next-line no-console
        console.debug('seg-change: refreshing from backend', { oldSeg, newSeg })
        const r = await fetch(`${API_BASE}/api/v1/state/current`)
        if (!r.ok) {
          // eslint-disable-next-line no-console
          console.debug('seg-change: refresh failed', { status: r.status })
          return
        }
        const cur = await r.json()
        const history = (cur.history ?? []) as Point[]
        const filtered = filterHistoryToSeg(history, newSeg)
         const segValues = history
          .map((p) => p.seg)
          .filter((v): v is number => v !== undefined && v !== null)
        const latestSeg = segValues.length > 0 ? Math.max(...segValues) : null
        setCurrent(cur)
        setSnapshotHistory(filtered)
        if (pSeriesRef.current && filtered.length > 0) {
          setDataFromHistory(filtered)
        }
        // eslint-disable-next-line no-console
        console.debug('seg-change: refresh ok', {
          oldSeg,
          newSeg,
          historyLength: history.length,
          filteredLength: filtered.length,
            latestSeg,
        })
      } catch (e) {
        // eslint-disable-next-line no-console
        console.debug('seg-change: refresh error', e)
      }
    },
    [setDataFromHistory],
  )

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
          const lastSegmentOnly = filterHistoryToSeg(hist, seg)
          setSnapshotHistory(lastSegmentOnly)
          if (pSeriesRef.current && lastSegmentOnly.length > 0) setDataFromHistory(lastSegmentOnly)
        } else if (msg.type === 'point' && msg.point) {
          const pt = msg.point as Point
          if (msg.current != null) {
            setCurrent(msg.current as typeof current)
          } else {
            setCurrent((prev) => ({ ...prev, state: prev?.state, derived: { ...prev?.derived, p_hat: pt.p } }))
          }
          if (!pSeriesRef.current) {
            pendingPointsRef.current.push(pt)
            return
          }
          if (!pausedRef.current) {
            if (pt.seg !== undefined && pt.seg !== currentSegRef.current) {
              const newSeg = pt.seg
              const oldSeg = currentSegRef.current
              currentSegRef.current = newSeg
              // eslint-disable-next-line no-console
              console.debug('seg-change: detected', { oldSeg, newSeg })
              void refreshSegmentFromBackend(newSeg)
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
        {(() => {
          const debug = (current?.derived as {
            debug?: {
              bo3_health?: string
              bo3_health_reason?: string | null
              bo3_snapshot_status?: string
              bo3_feed_error?: string | null
              bo3_buffer_age_s?: number | null
              bo3_buffer_consecutive_failures?: number
            }
          } | undefined)?.debug
          const health = debug?.bo3_health
          const healthReason = debug?.bo3_health_reason
          const status = debug?.bo3_snapshot_status
          const err = debug?.bo3_feed_error
          const bufferAgeS = debug?.bo3_buffer_age_s
          const consecutiveFailures = debug?.bo3_buffer_consecutive_failures
          if (health != null) {
            const reason = healthReason != null && healthReason !== '' ? ` (${healthReason})` : ''
            const agePart = bufferAgeS != null ? `age ${Math.round(bufferAgeS)}s` : ''
            const failsPart = consecutiveFailures != null && consecutiveFailures > 0 ? `fails ${consecutiveFailures}` : ''
            const extra = [agePart, failsPart].filter(Boolean).length > 0 ? ` (${[agePart, failsPart].filter(Boolean).join(', ')})` : ''
            return (
              <p style={{ fontSize: 13, color: '#9ca3af' }}>
                BO3 health: <strong>{health}</strong>{reason}{extra}
              </p>
            )
          }
          if (status == null) return null
          return (
            <p style={{ fontSize: 13, color: '#9ca3af' }}>
              BO3 status: <strong>{status}</strong>
              {err != null && err !== '' && <> ({err})</>}
            </p>
          )
        })()}
      </section>
      <section style={{ marginTop: 16, padding: 12, border: '1px solid #374151', borderRadius: 4 }}>
        <h3 style={{ marginTop: 0 }}>Prematch</h3>
        <p>
          <label>
            prematch_series (0–1):{' '}
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={prematchSeriesInput}
              onChange={(e) => setPrematchSeriesInput(e.target.value)}
              placeholder="0.5"
              style={{ width: 80 }}
            />
          </label>
          {' '}
          <button
            type="button"
            onClick={async () => {
              const v = parseFloat(prematchSeriesInput)
              if (Number.isNaN(v) || v <= 0 || v >= 1) {
                setConfigError('prematch_series must be between 0.01 and 0.99')
                return
              }
              setConfigError(null)
              try {
                const r = await fetch(`${API_BASE}/api/v1/prematch/set`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ prematch_series: v, prematch_locked: true }),
                })
                if (!r.ok) {
                  const body = await r.json().catch(() => ({}))
                  setConfigError((body as { detail?: string })?.detail ?? r.statusText)
                  return
                }
                const data = await r.json()
                setCurrent(data)
              } catch (e) {
                setConfigError(e instanceof Error ? e.message : String(e))
              }
            }}
          >
            Set prematch (lock)
          </button>
          {' '}
          <button
            type="button"
            onClick={async () => {
              setConfigError(null)
              try {
                const r = await fetch(`${API_BASE}/api/v1/prematch/unlock`, { method: 'POST' })
                if (!r.ok) return
                const data = await r.json()
                setCurrent(data)
              } catch {
                // ignore
              }
            }}
          >
            Unlock prematch
          </button>
          {' '}
          <button
            type="button"
            onClick={async () => {
              setConfigError(null)
              try {
                const r = await fetch(`${API_BASE}/api/v1/prematch/clear`, { method: 'POST' })
                if (!r.ok) return
                const data = await r.json()
                setCurrent(data)
              } catch {
                // ignore
              }
            }}
          >
            Clear
          </button>
        </p>
        {(() => {
          const config = (current?.state as { config?: { prematch_series?: number | null; prematch_map?: number | null; prematch_locked?: boolean } })?.config
          const ps = config?.prematch_series
          const pm = config?.prematch_map
          const locked = config?.prematch_locked ?? false
          if (ps == null && pm == null) {
            return <p style={{ fontSize: 13, color: '#9ca3af' }}>No prematch set. Use 0.01–0.99 and Set prematch (lock).</p>
          }
          return (
            <p style={{ fontSize: 13, color: '#9ca3af' }}>
              prematch_series: <strong>{ps != null ? ps.toFixed(4) : '—'}</strong>
              {' · '}
              prematch_map (derived): <strong>{pm != null ? pm.toFixed(4) : '—'}</strong>
              {' · '}
              locked: <strong>{locked ? 'yes' : 'no'}</strong>
            </p>
          )
        })()}
      </section>
      <section style={{ marginTop: 16, padding: 12, border: '1px solid #374151', borderRadius: 4 }}>
        <h3 style={{ marginTop: 0 }}>Market (Kalshi)</h3>
        <p>
          <label>
            Kalshi URL:{' '}
            <input
              type="text"
              value={kalshiUrl}
              onChange={(e) => setKalshiUrl(e.target.value)}
              placeholder="https://kalshi.com/..."
              style={{ width: 320 }}
            />
          </label>
          {' '}
          <button
            type="button"
            onClick={async () => {
              setMarketError(null)
              if (!kalshiUrl.trim()) {
                setMarketError('Enter a Kalshi URL')
                return
              }
              try {
                const r = await fetch(`${API_BASE}/api/v1/market/resolve`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ kalshi_url: kalshiUrl.trim() }),
                })
                if (!r.ok) {
                  const body = await r.json().catch(() => ({}))
                  setMarketError((body as { detail?: string })?.detail ?? r.statusText)
                  setMarketOptions([])
                  return
                }
                const data = await r.json()
                const opts = Array.isArray(data.options) ? data.options : []
                setMarketOptions(opts)
                if (opts.length > 0 && data.suggested) setSelectedMarketKey(data.suggested)
                else if (opts.length > 0) setSelectedMarketKey(opts[0].key)
              } catch (e) {
                setMarketError(e instanceof Error ? e.message : String(e))
                setMarketOptions([])
              }
            }}
          >
            Load markets
          </button>
        </p>
        {marketOptions.length > 0 && (
          <p>
            <label>
              Team / side:{' '}
              <select
                value={selectedMarketKey}
                onChange={(e) => setSelectedMarketKey(e.target.value)}
                style={{ minWidth: 280 }}
              >
                {marketOptions.map((opt) => (
                  <option key={opt.key} value={opt.key}>
                    {opt.label || opt.key}
                  </option>
                ))}
              </select>
            </label>
            {' '}
            <button
              type="button"
              onClick={async () => {
                setMarketError(null)
                try {
                  const r = await fetch(`${API_BASE}/api/v1/market/select`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      kalshi_url: kalshiUrl.trim() || undefined,
                      market_side_key: selectedMarketKey,
                    }),
                  })
                  if (!r.ok) {
                    const body = await r.json().catch(() => ({}))
                    setMarketError((body as { detail?: string })?.detail ?? r.statusText)
                    return
                  }
                  const data = await r.json()
                  setCurrent(data)
                  setWsReconnectTrigger((prev) => prev + 1)
                } catch (e) {
                  setMarketError(e instanceof Error ? e.message : String(e))
                }
              }}
            >
              Track selected
            </button>
          </p>
        )}
        {marketError && <p style={{ color: '#ef4444', fontSize: 14 }}>{marketError}</p>}
      </section>
      <section style={{ marginTop: 16, padding: 12, border: '1px solid #374151', borderRadius: 4 }}>
        <h3 style={{ marginTop: 0 }}>Breaches</h3>
        <p>
          <button
            type="button"
            onClick={async () => {
              try {
                const r = await fetch(`${API_BASE}/api/v1/market/breaches?limit=100`)
                if (!r.ok) return
                const data = await r.json()
                setBreaches(Array.isArray(data) ? data : [])
              } catch {
                setBreaches([])
              }
            }}
          >
            Refresh breaches
          </button>
          {' '}
          <span style={{ fontSize: 12, color: '#9ca3af' }}>Last 20 shown</span>
        </p>
        {breaches.length === 0 && <p style={{ fontSize: 13, color: '#9ca3af' }}>No breach events. Click Refresh or wait for auto-poll.</p>}
        {breaches.slice(-20).reverse().map((evt, i) => (
          <div key={i} style={{ fontSize: 12, marginBottom: 6, padding: 6, background: '#1f2937', borderRadius: 4 }}>
            <span style={{ color: '#9ca3af' }}>{new Date((evt.ts_epoch ?? 0) * 1000).toISOString().replace('T', ' ').slice(0, 19)}</span>
            {' · '}
            <strong>{evt.breach_type}</strong>
            {evt.breach_mag != null && <> mag={evt.breach_mag.toFixed(4)}</>}
            {' · '}
            score {evt.scores?.[0] ?? 0}-{evt.scores?.[1] ?? 0}
            {evt.market_mid != null && (
              <> · market_mid={evt.market_mid.toFixed(4)} vs [{evt.map_low?.toFixed(4) ?? '?'}, {evt.map_high?.toFixed(4) ?? '?'}]</>
            )}
          </div>
        ))}
      </section>
      <section style={{ marginTop: 16, padding: 12, border: '1px solid #374151', borderRadius: 4 }}>
        <h3 style={{ marginTop: 0 }}>PHAT / Model Debug</h3>
        {(() => {
          const debug = (current?.derived as { debug?: Record<string, unknown> } | undefined)?.debug
          if (!debug || typeof debug !== 'object') {
            return <p style={{ fontSize: 12, color: '#9ca3af' }}>No debug (connect and run BO3/Replay).</p>
          }
          const d = debug as Record<string, unknown>
          const MAX_JSON_LEN = 800
          const formatVal = (val: unknown): string => {
            if (val === undefined || val === null) return '—'
            if (typeof val === 'object') {
              try {
                const json = JSON.stringify(val, null, 2)
                return json.length > MAX_JSON_LEN ? json.slice(0, MAX_JSON_LEN) + '…' : json
              } catch {
                return String(val)
              }
            }
            if (typeof val === 'boolean') return val ? 'true' : 'false'
            if (typeof val === 'number') return Number.isInteger(val) ? String(val) : (val as number).toFixed(4)
            return String(val)
          }
          const v = (key: string) => {
            const val = d[key]
            if (val === undefined || val === null) return '—'
            if (typeof val === 'object' && val !== null) return formatVal(val)
            if (typeof val === 'boolean') return val ? 'true' : 'false'
            if (typeof val === 'number') return Number.isInteger(val) ? String(val) : (val as number).toFixed(4)
            return String(val)
          }
          const vNested = (obj: unknown, key: string) => {
            if (obj == null || typeof obj !== 'object') return '—'
            const o = obj as Record<string, unknown>
            const val = o[key]
            if (val === undefined || val === null) return '—'
            if (typeof val === 'boolean') return val ? 'true' : 'false'
            if (typeof val === 'number') return Number.isInteger(val) ? String(val) : (val as number).toFixed(4)
            return String(val)
          }
          const truncateJson = (obj: unknown): string => {
            try {
              const json = JSON.stringify(obj, null, 2)
              return json.length > MAX_JSON_LEN ? json.slice(0, MAX_JSON_LEN) + '…' : json
            } catch {
              return String(obj)
            }
          }
          const scalarStr = (val: unknown): string => {
            if (val === undefined || val === null) return '—'
            if (typeof val === 'number') return Number.isInteger(val) ? String(val) : (val as number).toFixed(4)
            return String(val)
          }
          const seriesLow = d.series_low ?? d.bound_low
          const seriesHigh = d.series_high ?? d.bound_high
          const mapLow = d.map_low ?? d.rail_low
          const mapHigh = d.map_high ?? d.rail_high
          const mid = (d.midround_v2 != null && typeof d.midround_v2 === 'object') ? (d.midround_v2 as Record<string, unknown>) : d
          const mv = (key: string) => {
            const val = mid[key]
            if (val === undefined || val === null) return '—'
            if (typeof val === 'boolean') return val ? 'true' : 'false'
            if (typeof val === 'number') return Number.isInteger(val) ? String(val) : (val as number).toFixed(4)
            return String(val)
          }
          const ap = d.active_points as Record<string, unknown> | undefined
          const lines: string[] = [
            'Prematch:',
            `  prematch_series_used=${v('prematch_series_used')}  prematch_map_used=${v('prematch_map_used')}  prematch_locked=${v('prematch_locked')}`,
            'Corridors:',
            `  series_low=${scalarStr(seriesLow)}  series_high=${scalarStr(seriesHigh)}  map_low=${scalarStr(mapLow)}  map_high=${scalarStr(mapHigh)}`,
            'Resolver:',
            `  p_hat_old=${v('p_hat_old')}  p_hat_final=${v('p_hat_final')}  bo3_health=${v('bo3_health')}`,
            'Midround V2:',
            `  q_intra=${mv('q_intra')}  raw_score=${mv('raw_score')}  urgency=${mv('urgency')}  time_progress=${mv('time_progress')}`,
            `  used_time=${mv('used_time')}  used_loadout=${mv('used_loadout')}  used_bomb_direction=${mv('used_bomb_direction')}  used_armor=${mv('used_armor')}  used_econ=${mv('used_econ')}`,
            'Endpoints:',
            `  canonical_if_a_round=${v('canonical_if_a_round')}  canonical_if_b_round=${v('canonical_if_b_round')}  base_span=${v('base_span')}  k=${v('k')}  a_active=${ap ? vNested(ap, 'a_active') : v('a_active')}  b_active=${ap ? vNested(ap, 'b_active') : v('b_active')}`,
          ]
          if (d.raw != null) {
            lines.push('', 'Raw:', truncateJson(d.raw))
          }
          if (d.fragility != null) {
            lines.push('', 'Fragility:', truncateJson(d.fragility))
          }
          return (
            <pre style={{ fontSize: 11, fontFamily: 'ui-monospace, monospace', margin: 0, color: '#9ca3af', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {lines.join('\n')}
            </pre>
          )
        })()}
      </section>
      <section style={{ marginTop: 16, padding: 12, border: '1px solid #374151', borderRadius: 4 }}>
        <h3 style={{ marginTop: 0 }}>Debug: Raw BO3 Snapshot</h3>
        <p>
          <button
            type="button"
            onClick={async () => {
              setRawSnapshotError(null)
              try {
                const r = await fetch(`${API_BASE}/api/v1/debug/bo3/last_snapshot`)
                if (!r.ok) {
                  const body = await r.json().catch(() => ({}))
                  setRawSnapshotJson(null)
                  setRawSnapshotError((body as { detail?: string })?.detail ?? r.statusText)
                  return
                }
                const data = await r.json()
                const str = JSON.stringify(data, null, 2)
                const maxLen = 120000
                setRawSnapshotJson(str.length > maxLen ? str.slice(0, maxLen) + '…' : str)
                setRawSnapshotOpen(true)
              } catch (e) {
                setRawSnapshotJson(null)
                setRawSnapshotError(e instanceof Error ? e.message : String(e))
              }
            }}
          >
            Fetch raw snapshot
          </button>
          {rawSnapshotError && <span style={{ marginLeft: 8, color: '#ef4444', fontSize: 12 }}>{rawSnapshotError}</span>}
        </p>
        {rawSnapshotJson != null && (
          <div>
            <button type="button" onClick={() => setRawSnapshotOpen((o) => !o)} style={{ fontSize: 12, marginBottom: 4 }}>
              {rawSnapshotOpen ? 'Collapse' : 'Expand'}
            </button>
            {rawSnapshotOpen && (
              <pre style={{ fontSize: 10, fontFamily: 'ui-monospace, monospace', margin: 0, color: '#9ca3af', whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: 400, overflow: 'auto', border: '1px solid #374151', padding: 8, borderRadius: 4 }}>
                {rawSnapshotJson}
              </pre>
            )}
          </div>
        )}
      </section>
      <section style={{ marginTop: 16, padding: 12, border: '1px solid #374151', borderRadius: 4 }}>
        <h3 style={{ marginTop: 0 }}>Replay (JSONL)</h3>
        <p>
          <label>
            Path:{' '}
            <input
              type="text"
              value={replayPath}
              onChange={(e) => setReplayPath(e.target.value)}
              style={{ width: 280 }}
            />
          </label>
        </p>
        <p>
          <button
            type="button"
            onClick={async () => {
              setReplayError(null)
              try {
                const r = await fetch(`${API_BASE}/api/v1/replay/matches?path=${encodeURIComponent(replayPath)}`)
                if (!r.ok) {
                  const body = await r.json().catch(() => ({}))
                  setReplayError((body as { detail?: string })?.detail ?? r.statusText)
                  setReplayMatches([])
                  return
                }
                const data = await r.json()
                setReplayMatches(Array.isArray(data) ? data : [])
              } catch (e) {
                setReplayError(e instanceof Error ? e.message : String(e))
                setReplayMatches([])
              }
            }}
          >
            Load replay matches
          </button>
        </p>
        <p>
          <label>
            Match:{' '}
            <select
              value={replayMatchId}
              onChange={(e) => setReplayMatchId(e.target.value)}
              style={{ minWidth: 280 }}
            >
              <option value="">— Select or type below —</option>
              {replayMatches.map((m) => (
                <option key={m.match_id} value={String(m.match_id)}>
                  {m.match_id} — {m.team1} vs {m.team2} ({m.count})
                </option>
              ))}
            </select>
          </label>
        </p>
        <p>
          <label>
            Match ID (manual):{' '}
            <input
              type="number"
              value={replayMatchId}
              onChange={(e) => setReplayMatchId(e.target.value)}
              placeholder="optional filter"
              style={{ width: 100 }}
            />
          </label>
        </p>
        <p>
          <label>
            Speed:{' '}
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={replaySpeed}
              onChange={(e) => setReplaySpeed(Number(e.target.value) || 1)}
              style={{ width: 60 }}
            />
          </label>
          {' '}
          <label>
            <input type="checkbox" checked={replayLoop} onChange={(e) => setReplayLoop(e.target.checked)} />
            Loop
          </label>
        </p>
        <p>
          <button
            type="button"
            onClick={async () => {
              setReplayError(null)
              try {
                const r = await fetch(`${API_BASE}/api/v1/replay/load`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    path: replayPath,
                    match_id: replayMatchId ? Number(replayMatchId) : undefined,
                    speed: replaySpeed,
                    loop: replayLoop,
                  }),
                })
                if (!r.ok) {
                  const body = await r.json().catch(() => ({}))
                  setReplayError((body as { detail?: string })?.detail ?? r.statusText)
                  return
                }
                setWsReconnectTrigger((prev) => prev + 1)
              } catch (e) {
                setReplayError(e instanceof Error ? e.message : String(e))
              }
            }}
          >
            Load Replay
          </button>
          {' '}
          <button
            type="button"
            onClick={async () => {
              setReplayError(null)
              try {
                await fetch(`${API_BASE}/api/v1/replay/stop`, { method: 'POST' })
                setWsReconnectTrigger((prev) => prev + 1)
              } catch (e) {
                setReplayError(e instanceof Error ? e.message : String(e))
              }
            }}
          >
            Stop Replay
          </button>
        </p>
        {replayError && <p style={{ color: '#ef4444', fontSize: 14 }}>{replayError}</p>}
      </section>
      {/* MatchHUD: broadcast scoreboard (Thunderpick-like) + chart */}
      <div style={{ display: 'flex', flexDirection: 'column', marginTop: 16, height: '75vh', minHeight: 520, minWidth: 0 }}>
        <MatchHUD current={current} />
        <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
          <div
            style={{
              position: 'absolute',
              left: 8,
              top: 8,
              zIndex: 10,
              fontSize: 12,
              fontFamily: 'monospace',
              color: '#9ca3af',
              background: 'rgba(26, 26, 26, 0.9)',
              padding: '4px 8px',
              borderRadius: 4,
            }}
          >
            Crosshair t: {crosshairT !== null ? String(crosshairT) : '—'}
          </div>
          <div
            ref={chartRef}
            style={{
              width: '100%',
              height: '100%',
              minHeight: 300,
            }}
          />
        </div>
      </div>
    </div>
  )
}

/** Player state from feed (team_one/team_two.player_states) when available (raw fallback) */
type PlayerStateRow = {
  nickname?: string
  is_alive?: boolean
  health?: number
  balance?: number
  equipment_value?: number
  has_bomb?: boolean
  has_defuse_kit?: boolean
  has_helmet?: boolean
  has_kevlar?: boolean
}

/** PlayerRow DTO from backend Frame.players_a / players_b */
type PlayerDto = {
  name?: string | null
  alive?: boolean | null
  hp?: number | null
  armor?: number | null
  helmet?: boolean | null
  cash?: number | null
  loadout?: number | null
  weapons?: string[] | null
  has_bomb?: boolean | null
  has_kit?: boolean | null
}

/** Last frame: normalized (teams, scores, totals) + HUD players + optional raw team_one/team_two with player_states */
type LastFrame = {
  teams?: string[]
  scores?: number[]
  alive_counts?: number[]
  hp_totals?: number[]
  cash_totals?: number[] | null
  loadout_totals?: number[] | null
  wealth_totals?: number[] | null
  bomb_phase_time_remaining?: { round_phase?: string; is_bomb_planted?: boolean } | null
  round_time_remaining_s?: number | null
  map_index?: number
  series_score?: number[]
  map_name?: string
  a_side?: string | null
  players_a?: PlayerDto[]
  players_b?: PlayerDto[]
  team_one?: { name?: string; side?: string; player_states?: PlayerStateRow[] }
  team_two?: { name?: string; side?: string; player_states?: PlayerStateRow[] }
}

function MatchHUD({ current }: { current: { state?: { last_frame?: LastFrame }; derived?: { debug?: Record<string, unknown> } } | null }) {
  const frame = current?.state?.last_frame
  const debug = current?.derived?.debug as {
    bo3_health?: string
    bo3_buffer_age_s?: number | null
  } | undefined
  const bo3Health = debug?.bo3_health ?? '—'
  const bufferAgeS = debug?.bo3_buffer_age_s
  const bufferAgeStr = bufferAgeS != null ? `${Math.round(bufferAgeS)}s` : ''

  if (!frame) {
    return (
      <section style={{ flex: '0 0 auto', height: 120, padding: 12, border: '1px solid #374151', borderRadius: 4, background: '#1f2937', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#9ca3af', fontSize: 14 }}>No match frame — connect BO3 or Replay</span>
      </section>
    )
  }

  const teams = frame.teams ?? ['A', 'B']
  const scores = frame.scores ?? [0, 0]
  const mapName = frame.map_name || '—'
  const mapIndex = (frame.map_index ?? 0) + 1
  const seriesScore = frame.series_score ?? [0, 0]
  const phaseObj = frame.bomb_phase_time_remaining
  const roundPhase = (phaseObj && typeof phaseObj === 'object' && phaseObj.round_phase) ? String(phaseObj.round_phase) : '—'
  const clockS = frame.round_time_remaining_s
  const clockStr = clockS != null ? `${Math.floor(clockS / 60)}:${String(Math.floor(clockS % 60)).padStart(2, '0')}` : '—'
  const aSide = frame.a_side ? String(frame.a_side).toUpperCase() : null
  const teamASide = frame.team_one?.side ? String(frame.team_one.side).toUpperCase() : aSide
  const teamBSide = frame.team_two?.side ? String(frame.team_two.side).toUpperCase() : (aSide === 'CT' ? 'T' : aSide === 'T' ? 'CT' : null)

  const sideColor = (side: string | null) => {
    if (!side) return { bg: '#374151', color: '#e5e7eb' }
    if (side === 'CT') return { bg: '#1e3a5f', color: '#93c5fd' }
    if (side === 'TERRORIST' || side === 'T') return { bg: '#5c3d1e', color: '#fcd34d' }
    return { bg: '#374151', color: '#e5e7eb' }
  }

  const dtoPlayersA: PlayerDto[] = Array.isArray(frame.players_a) ? frame.players_a : []
  const dtoPlayersB: PlayerDto[] = Array.isArray(frame.players_b) ? frame.players_b : []
  const rawPlayersA: PlayerStateRow[] =
    frame.team_one?.player_states && Array.isArray(frame.team_one.player_states) ? frame.team_one.player_states : []
  const rawPlayersB: PlayerStateRow[] =
    frame.team_two?.player_states && Array.isArray(frame.team_two.player_states) ? frame.team_two.player_states : []

  type HudPlayerRow = {
    name: string
    alive: boolean
    hp: number
    cash: number
    loadout: number
    hasBomb: boolean
    hasKit: boolean
    hasHelmet: boolean
    hasKevlar: boolean
  }

  const fromDto = (rows: PlayerDto[]): HudPlayerRow[] =>
    rows.map((p) => {
      const alive = p.alive !== false
      const hp = typeof p.hp === 'number' ? p.hp : 0
      const cash = typeof p.cash === 'number' ? p.cash : 0
      const loadout = typeof p.loadout === 'number' ? p.loadout : 0
      const armor = typeof p.armor === 'number' ? p.armor : 0
      const name = (p.name && String(p.name)) || '—'
      return {
        name,
        alive,
        hp,
        cash,
        loadout,
        hasBomb: !!p.has_bomb,
        hasKit: !!p.has_kit,
        hasHelmet: !!p.helmet,
        hasKevlar: armor > 0,
      }
    })

  const fromRaw = (rows: PlayerStateRow[]): HudPlayerRow[] =>
    rows.map((p) => {
      const alive = p.is_alive !== false
      const hp = typeof p.health === 'number' ? p.health : 0
      const cash = typeof p.balance === 'number' ? p.balance : 0
      const loadout = typeof p.equipment_value === 'number' ? p.equipment_value : 0
      const name = (p.nickname && String(p.nickname)) || '—'
      return {
        name,
        alive,
        hp,
        cash,
        loadout,
        hasBomb: !!p.has_bomb,
        hasKit: !!p.has_defuse_kit,
        hasHelmet: !!p.has_helmet,
        hasKevlar: !!p.has_kevlar,
      }
    })

  const useDto = dtoPlayersA.length > 0 || dtoPlayersB.length > 0
  const playersA: HudPlayerRow[] = useDto ? fromDto(dtoPlayersA) : fromRaw(rawPlayersA)
  const playersB: HudPlayerRow[] = useDto ? fromDto(dtoPlayersB) : fromRaw(rawPlayersB)

  const hasPlayers = playersA.length > 0 || playersB.length > 0
  const pad = (arr: HudPlayerRow[], n: number) => {
    const out = [...arr]
    while (out.length < n) out.push({})
    return out.slice(0, n)
  }
  const rowsA = pad(playersA, 5)
  const rowsB = pad(playersB, 5)
  const hpA = frame.hp_totals?.[0] ?? 0
  const hpB = frame.hp_totals?.[1] ?? 0
  const cashA = frame.cash_totals?.[0] ?? 0
  const cashB = frame.cash_totals?.[1] ?? 0
  const loadoutA = frame.loadout_totals?.[0] ?? 0
  const loadoutB = frame.loadout_totals?.[1] ?? 0
  const aliveA = frame.alive_counts?.[0] ?? 0
  const aliveB = frame.alive_counts?.[1] ?? 0

  const PlayerRow = ({ p, index }: { p: HudPlayerRow; index: number }) => {
    const { alive, hp, cash, loadout, name, hasBomb, hasKit, hasHelmet, hasKevlar } = p
    return (
      <div
        key={index}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '4px 8px',
          fontSize: 12,
          opacity: alive ? 1 : 0.5,
          color: alive ? '#e5e7eb' : '#6b7280',
          background: alive ? 'transparent' : 'rgba(0,0,0,0.2)',
          borderRadius: 4,
        }}
      >
        <span style={{ flex: '0 0 12px', textAlign: 'center' }}>{hasBomb ? '💣' : ''}</span>
        <span style={{ flex: '0 0 12px', textAlign: 'center' }}>{hasKit ? '🛡' : ''}</span>
        <span style={{ flex: '0 0 12px', textAlign: 'center' }}>{(hasHelmet || hasKevlar) ? '⛑' : ''}</span>
        <span style={{ minWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={name}>{name}</span>
        <span style={{ width: 60 }}>
          <span style={{ display: 'inline-block', width: 36, height: 6, background: '#374151', borderRadius: 2, overflow: 'hidden' }}>
            <span style={{ display: 'inline-block', height: '100%', width: `${Math.max(0, Math.min(100, hp))}%`, background: alive ? '#22c55e' : '#6b7280' }} />
          </span>
          <span style={{ marginLeft: 4 }}>{hp}</span>
        </span>
        <span style={{ width: 44, color: '#9ca3af' }}>${cash}</span>
        <span style={{ width: 48, color: '#9ca3af' }}>${loadout}</span>
      </div>
    )
  }

  const PlaceholderRow = () => (
    <div style={{ padding: '4px 8px', fontSize: 12, color: '#6b7280' }}>—</div>
  )

  return (
    <section
      style={{
        flex: '0 0 auto',
        height: 280,
        maxHeight: 320,
        padding: 0,
        border: '1px solid #374151',
        borderRadius: 4,
        background: '#111827',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Top bar: series info + BO3 health */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: '#1f2937', borderBottom: '1px solid #374151', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontSize: 12, color: '#9ca3af' }}>
          <span>BO3</span>
          <span style={{ marginLeft: 8 }}>Map {mapIndex}</span>
          <span style={{ marginLeft: 8 }}>{mapName}</span>
          <span style={{ marginLeft: 8 }}>({seriesScore[0] ?? 0}–{seriesScore[1] ?? 0})</span>
        </div>
        <div style={{ fontSize: 12, color: '#9ca3af' }}>
          BO3 health: <strong style={{ color: '#e5e7eb' }}>{bo3Health}</strong>
          {bufferAgeStr && <span style={{ marginLeft: 8 }}>age {bufferAgeStr}</span>}
        </div>
      </div>
      {/* Middle: big map score + phase + clock */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24, padding: '12px 16px', borderBottom: '1px solid #374151' }}>
        <div style={{ fontSize: 28, fontWeight: 700, color: '#e5e7eb' }}>{teams[0] ?? 'A'}</div>
        <div style={{ fontSize: 32, fontWeight: 800, color: '#fbbf24' }}>{scores[0] ?? 0} – {scores[1] ?? 0}</div>
        <div style={{ fontSize: 28, fontWeight: 700, color: '#e5e7eb' }}>{teams[1] ?? 'B'}</div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 16, padding: '4px 16px', fontSize: 12, color: '#9ca3af' }}>
        <span>{roundPhase}</span>
        <span>Clock: {clockStr}</span>
      </div>
      {/* Two columns: Team A (left), Team B (right) */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: '1px solid #374151' }}>
          <div style={{ padding: '6px 8px', fontSize: 13, fontWeight: 600, ...sideColor(teamASide) }}>
            {teams[0] ?? 'Team A'}
          </div>
          {hasPlayers ? rowsA.map((p, i) => (p.nickname != null || p.health != null ? <PlayerRow key={i} p={p} index={i} /> : <PlaceholderRow key={i} />)) : (
            <>
              {[0, 1, 2, 3, 4].map((i) => <PlaceholderRow key={i} />)}
              <div style={{ padding: '4px 8px', fontSize: 11, color: '#6b7280', borderTop: '1px solid #374151' }}>
                Alive: {aliveA} · HP: {hpA} · $: {cashA} · Eq: {loadoutA}
              </div>
            </>
          )}
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '6px 8px', fontSize: 13, fontWeight: 600, ...sideColor(teamBSide) }}>
            {teams[1] ?? 'Team B'}
          </div>
          {hasPlayers ? rowsB.map((p, i) => (p.nickname != null || p.health != null ? <PlayerRow key={i} p={p} index={i} /> : <PlaceholderRow key={i} />)) : (
            <>
              {[0, 1, 2, 3, 4].map((i) => <PlaceholderRow key={i} />)}
              <div style={{ padding: '4px 8px', fontSize: 11, color: '#6b7280', borderTop: '1px solid #374151' }}>
                Alive: {aliveB} · HP: {hpB} · $: {cashB} · Eq: {loadoutB}
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  )
}

export default App

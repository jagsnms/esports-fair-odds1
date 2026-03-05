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
 * - event?: episode event (setup_trigger, episode_start, episode_end, episode_outcome)
 * - explain?: per-tick decomposition
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
  event?: unknown
  explain?: unknown
}

/** Marker for p_hat series (episode entry / resolution). */
type EpisodeMarker = {
  time: number
  position: 'aboveBar' | 'belowBar'
  shape: 'arrowUp' | 'arrowDown' | 'circle'
  color: string
  text: string
  id?: string
}

/** BO3 candidate from /api/v1/bo3/candidates (current + upcoming, not trusting BO3 live/current) */
type Bo3Match = {
  id: number
  team1_name: string
  team2_name: string
  bo_type: number
  tier?: unknown
  start_date?: unknown
  live_coverage?: boolean
  parsed_status?: string | null
}

/** Readiness probe result from POST /api/v1/bo3/readiness */
type Bo3Readiness = {
  match_id: number
  telemetry_ready: boolean
  status_code: number
  reason: string
  last_probe_ts: string
}

/** Session row from GET /api/v1/debug/telemetry/sessions */
type TelemetrySessionRow = {
  session_key: string
  source: string
  id: string
  last_update_ts: number | null
  age_s: number | null
  fetch_age_s?: number | null
  good_age_s?: number | null
  telemetry_ok?: boolean
  telemetry_reason?: string | null
  ctx?: {
    active_source?: string | null
    selector_decision?: { chosen_source?: string | null; reason?: string; considered?: unknown[] }
    last_accepted_key?: string | null
    last_env?: { key_display?: string } | null
    per_source_health?: Record<string, { ok_count?: number; err_count?: number; last_reason?: string | null }>
    last_frame?: { teams?: [string, string] | string[] }
  }
  grid_schedule?: { next_fetch_in_s?: number; last_rate_limit_reason?: string }
  last_error?: string | null
}

/** Full response from GET /api/v1/debug/telemetry/sessions */
type TelemetrySessionsResponse = {
  now_ts?: number
  sessions: TelemetrySessionRow[]
  bo3_auto_track_enabled?: boolean
  bo3_auto_match_ids?: number[]
  bo3_auto_last_refresh_age_s?: number | null
  bo3_readiness_cache_size?: number
  grid_auto_track_enabled?: boolean
  grid_auto_track_limit?: number
  grid_auto_series_ids?: string[]
  grid_auto_last_refresh_age_s?: number | null
}

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

/** Defensive: ensure a value is a valid Point (has t and p as numbers) so chart/state never crash. */
function isValidPoint(pt: unknown): pt is Point {
  return pt != null && typeof pt === 'object' && typeof (pt as Point).t === 'number' && typeof (pt as Point).p === 'number'
}

/** UTC time for lightweight-charts (epoch seconds as UTCTimestamp). */
function utcTimestamp(t: number): import('lightweight-charts').UTCTimestamp {
  return Math.floor(t) as import('lightweight-charts').UTCTimestamp
}

/** Build entry or resolution marker from event; returns marker to store and use in setMarkers. */
function eventToMarker(
  pt: Point,
  ev: { event_type?: string; episode_id?: string; trade_id?: string; end_reason?: string; direction?: string },
): EpisodeMarker | null {
  const time = utcTimestamp(pt.t)
  const id = ev.episode_id ?? ev.trade_id ?? `ep-${pt.t}`
  if (ev.event_type === 'episode_start' || ev.event_type === 'trade_open') {
    const direction = (ev.direction as string) ?? ''
    const isLongYes = direction === 'LONG_A'
    return {
      time,
      position: isLongYes ? 'belowBar' : 'aboveBar',
      shape: isLongYes ? 'arrowUp' : 'arrowDown',
      color: isLongYes ? '#22c55e' : '#ef4444',
      text: 'E',
      id: `entry-${id}`,
    }
  }
  if (ev.event_type === 'episode_end') {
    const reason = (ev.end_reason as string) ?? ''
    const text = reason ? `X ${reason}` : 'X'
    return {
      time,
      position: 'aboveBar',
      shape: 'circle',
      color: '#9ca3af',
      text,
      id: `res-${id}`,
    }
  }
  return null
}

function App() {
  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>('connecting')
  const [current, setCurrent] = useState<{ state?: unknown; derived?: { p_hat?: number } } | null>(null)
  const [hudFrame, setHudFrame] = useState<LastFrame | null>(null)
  const [snapshotHistory, setSnapshotHistory] = useState<Point[]>([])
  const [chartReady, setChartReady] = useState(false)
  const [isPaused] = useState(false)
  const [wsReconnectTrigger, setWsReconnectTrigger] = useState(0)

  const [liveMatches, setLiveMatches] = useState<Bo3Match[]>([])
  const [bo3ListLoaded, setBo3ListLoaded] = useState(false)
  const [bo3Readiness, setBo3Readiness] = useState<Record<number, Bo3Readiness>>({})
  const [bo3ShowAllCandidates, setBo3ShowAllCandidates] = useState(false)
  const [selectedMatchId, setSelectedMatchId] = useState<string>('')
  const [teamAIsTeamOne, setTeamAIsTeamOne] = useState(true)
  const [configError, setConfigError] = useState<string | null>(null)
  const [resetStatus, setResetStatus] = useState<string | null>(null)

  const [replaySources, setReplaySources] = useState<
    Array<{ label: string; path: string; kind: string; mtime: number; size: number }>
  >([])
  const [replayPath, setReplayPath] = useState('logs/bo3_pulls.jsonl')
  const [replayMatchId, setReplayMatchId] = useState('')
  const [replaySpeed, setReplaySpeed] = useState(1)
  const [replayLoop, setReplayLoop] = useState(true)
  const [replayError, setReplayError] = useState<string | null>(null)
  const [replayMatches, setReplayMatches] = useState<
    Array<{ match_id: number; team1: string; team2: string; count: number; path?: string }>
  >([])

  const [, setCrosshairT] = useState<string | number | null>(null)

  const [kalshiUrl, setKalshiUrl] = useState('')
  const [marketOptions, setMarketOptions] = useState<Array<{ key: string; label: string; ticker_yes: string }>>([])
  const [selectedMarketKey, setSelectedMarketKey] = useState('')
  const [marketError, setMarketError] = useState<string | null>(null)

  const [prematchSeriesInput, setPrematchSeriesInput] = useState('')

  const [rawSnapshotJson, setRawSnapshotJson] = useState<string | null>(null)
  const [rawSnapshotError, setRawSnapshotError] = useState<string | null>(null)
  const [rawSnapshotOpen, setRawSnapshotOpen] = useState(false)

  const [breaches, setBreaches] = useState<
    Array<{
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
    }>
  >([])

  const [debugOpen, setDebugOpen] = useState(false)

  const [telemetrySessions, setTelemetrySessions] = useState<TelemetrySessionsResponse | null>(null)
  const [selectedSession, setSelectedSession] = useState<{ source: string; id: string } | null>(null)
  const [telemetryFilterSource, setTelemetryFilterSource] = useState<'all' | 'BO3' | 'GRID'>('all')
  const [telemetryFilterStatus, setTelemetryFilterStatus] = useState<'all' | 'LIVE' | 'STALE' | 'DEAD'>('all')
  const [telemetrySearch, setTelemetrySearch] = useState('')
  const [telemetrySortColumn, setTelemetrySortColumn] = useState<string>('age_s')
  const [telemetrySortDir, setTelemetrySortDir] = useState<'asc' | 'desc'>('asc')

  const [gridCandidates, setGridCandidates] = useState<Array<{ series_id?: string; name?: string; tournament_name?: string; start_time?: string; live_data_feed_level?: string }>>([])

  // LEFT TOOLBAR: one drawer panel at a time
  const [activePanel, setActivePanel] = useState<'bo3' | 'prematch' | 'market' | 'replay' | 'telemetry' | 'telemetry_controls' | null>('bo3')

  // Midround V2 weight profile (temporary A/B toggle); persisted in backend via POST /config
  const [midroundV2WeightProfile, setMidroundV2WeightProfile] = useState<'current' | 'learned_v1' | 'learned_v2' | 'learned_fit'>('current')

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
  /** When true (default), chart shows full match across all segments; when false, current segment only. */
  const showFullMatchRef = useRef(true)
  /** Episode markers (entry E + resolution X) for p_hat series. Rebuilt from episode map and set via setMarkers. */
  const markersRef = useRef<EpisodeMarker[]>([])
  /** By episode_id: { entry?, resolution? } so we can rebuild full marker list and persist across segment/history refresh. */
  const episodeMarkersByIdRef = useRef<Map<string, { entry?: EpisodeMarker; resolution?: EpisodeMarker }>>(new Map())
  const liveMatchesRef = useRef<Bo3Match[]>([])
  liveMatchesRef.current = liveMatches
  /** Latest config (primary_session_*) so BO3 readiness poll can gate fanout when pinned (avoids 404 storms). */
  const configRef = useRef<{ primary_session_source?: string; primary_session_id?: string } | null>(null)
  configRef.current = (current?.state as { config?: { primary_session_source?: string; primary_session_id?: string } } | undefined)?.config ?? null

  // Sync midround V2 weight profile from server state when available (e.g. after fetch/WS)
  useEffect(() => {
    const config = (current?.state as { config?: { midround_v2_weight_profile?: string } } | undefined)?.config
    const profile = config?.midround_v2_weight_profile
    if (profile === 'current' || profile === 'learned_v1' || profile === 'learned_v2' || profile === 'learned_fit') setMidroundV2WeightProfile(profile)
  }, [current?.state])

  // BO3 candidates: refresh every 60s while panel is open (reduced from 10s; backend discovery handles the rest)
  useEffect(() => {
    if (activePanel !== 'bo3' || !bo3ListLoaded) return
    const refresh = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/bo3/candidates`)
        const data = await r.json()
        if (Array.isArray(data)) setLiveMatches(data)
      } catch {
        // keep previous list on error
      }
    }
    const id = setInterval(refresh, 60_000)
    return () => clearInterval(id)
  }, [activePanel, bo3ListLoaded])

  // BO3 readiness: poll every 60s; when primary is pinned to BO3, only probe pinned id (stops 404 fanout storms)
  useEffect(() => {
    if (activePanel !== 'bo3' || !bo3ListLoaded || liveMatches.length === 0) return
    const poll = async () => {
      const cfg = configRef.current
      const pinnedBo3 = cfg?.primary_session_source === 'BO3' && cfg?.primary_session_id
      const ids = pinnedBo3
        ? [Number(cfg!.primary_session_id)].filter(Number.isInteger)
        : liveMatchesRef.current.map((m) => m.id)
      if (ids.length === 0) return
      try {
        const r = await fetch(`${API_BASE}/api/v1/bo3/readiness`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ match_ids: ids }),
        })
        const data = (await r.json()) as Bo3Readiness[] | undefined
        if (Array.isArray(data)) {
          setBo3Readiness((prev) => {
            const next = { ...prev }
            for (const row of data) next[row.match_id] = row
            return next
          })
        }
      } catch {
        // keep previous readiness on error
      }
    }
    poll()
    const id = setInterval(poll, 60_000)
    return () => clearInterval(id)
  }, [activePanel, bo3ListLoaded, liveMatches.length])

  // Replay sources: fetch when opening replay panel; set default path from server
  useEffect(() => {
    if (activePanel !== 'replay') return
    let cancelled = false
    const load = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/replay/sources`)
        if (!r.ok || cancelled) return
        const data = (await r.json()) as { default_path?: string; sources?: Array<{ label: string; path: string; kind: string; mtime: number; size: number }> }
        if (cancelled) return
        setReplaySources(Array.isArray(data.sources) ? data.sources : [])
        if (data.default_path) setReplayPath((prev) => (prev === 'logs/bo3_pulls.jsonl' ? data.default_path! : prev))
      } catch {
        if (!cancelled) setReplaySources([])
      }
    }
    load()
    return () => { cancelled = true }
  }, [activePanel])

  // Refresh match list when replay panel is open: fetch matches from ALL sources so every replay file is included
  useEffect(() => {
    if (activePanel !== 'replay') return
    let cancelled = false
    const load = async () => {
      setReplayError(null)
      try {
        const r = await fetch(`${API_BASE}/api/v1/replay/matches?all_sources=1`)
        if (cancelled) return
        if (!r.ok) {
          const body = await r.json().catch(() => ({}))
          setReplayError((body as { detail?: string })?.detail ?? r.statusText)
          setReplayMatches([])
          return
        }
        const data = (await r.json()) as unknown
        if (cancelled) return
        if (Array.isArray(data)) {
          setReplayMatches(data as Array<{ match_id: number; team1: string; team2: string; count: number; path?: string }>)
        } else {
          const matches = (data as { matches?: unknown[] }).matches
          setReplayMatches(Array.isArray(matches) ? (matches as Array<{ match_id: number; team1: string; team2: string; count: number; path?: string }>) : [])
        }
      } catch (e) {
        if (!cancelled) {
          setReplayError(e instanceof Error ? e.message : String(e))
          setReplayMatches([])
        }
      }
    }
    load()
    return () => { cancelled = true }
  }, [activePanel])

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

  // Telemetry Sessions: poll every 5s when panel is open (render-only; does not trigger provider fetches)
  useEffect(() => {
    if (activePanel !== 'telemetry') return
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/v1/debug/telemetry/sessions`)
        if (!r.ok) return
        const data = (await r.json()) as TelemetrySessionsResponse
        setTelemetrySessions(data)
      } catch {
        setTelemetrySessions((prev) => prev)
      }
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
  }, [activePanel])

  // Sync selectedSession from config when backend has primary pinned (so UI stays in sync; polling does not overwrite selection)
  useEffect(() => {
    const cfg = (current?.state as { config?: { primary_session_source?: string; primary_session_id?: string } } | undefined)?.config
    const src = cfg?.primary_session_source
    const id = cfg?.primary_session_id
    if (src && id) setSelectedSession({ source: src, id })
  }, [current])

  /** Rebuild marker list from episode map and set on p_hat series. Call after adding/updating episode markers. */
  const applyEpisodeMarkers = useCallback(() => {
    const series = pSeriesRef.current
    if (!series) return
    const map = episodeMarkersByIdRef.current
    const all: EpisodeMarker[] = []
    map.forEach(({ entry, resolution }) => {
      if (entry) all.push(entry)
      if (resolution) all.push(resolution)
    })
    all.sort((a, b) => a.time - b.time)
    markersRef.current = all
    series.setMarkers(
      all.map((m) => ({
        time: m.time as import('lightweight-charts').UTCTimestamp,
        position: m.position,
        shape: m.shape,
        color: m.color,
        text: m.text,
        id: m.id,
      })),
    )
  }, [])

  /** Handle a point that has event: update episode marker map and refresh series markers. */
  const handleEventMarker = useCallback(
    (point: Point) => {
      const ev = point.event
      if (ev == null || typeof ev !== 'object') return
      const e = ev as { event_type?: string; episode_id?: string; trade_id?: string; end_reason?: string; direction?: string }
      const id = e.episode_id ?? e.trade_id
      if (id == null || typeof id !== 'string') return
      const marker = eventToMarker(point, e)
      if (!marker) return
      const map = episodeMarkersByIdRef.current
      let pair = map.get(id)
      if (!pair) {
        pair = {}
        map.set(id, pair)
      }
      if (e.event_type === 'episode_start' || e.event_type === 'trade_open') {
        pair.entry = marker
      } else if (e.event_type === 'episode_end') {
        pair.resolution = marker
      }
      applyEpisodeMarkers()
    },
    [applyEpisodeMarkers],
  )

  /** From a full history array, extract event points and build episode marker map; then apply to chart. */
  const buildEpisodeMarkersFromHistory = useCallback(
    (history: Point[]) => {
      episodeMarkersByIdRef.current.clear()
      for (const pt of history) {
        if (pt.event == null || typeof pt.event !== 'object') continue
        const ev = pt.event as { event_type?: string; episode_id?: string; trade_id?: string }
        const id = ev.episode_id ?? ev.trade_id
        if (id == null || typeof id !== 'string') continue
        const marker = eventToMarker(pt, ev)
        if (!marker) continue
        let pair = episodeMarkersByIdRef.current.get(id)
        if (!pair) {
          pair = {}
          episodeMarkersByIdRef.current.set(id, pair)
        }
        if (ev.event_type === 'episode_start' || ev.event_type === 'trade_open') {
          pair.entry = marker
        } else if (ev.event_type === 'episode_end') {
          pair.resolution = marker
        }
      }
      applyEpisodeMarkers()
    },
    [applyEpisodeMarkers],
  )

  const applyPointToChart = useCallback((point: Point) => {
    if (!isValidPoint(point)) return
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
    // Defensive: only use points with valid t and p to avoid chart crash or white screen
    const valid: Point[] = Array.isArray(history) ? history.filter(isValidPoint) : []
    if (valid.length === 0) {
      pSeriesRef.current.setData([])
      loSeriesRef.current.setData([])
      hiSeriesRef.current.setData([])
      railLoSeriesRef.current?.setData([])
      railHiSeriesRef.current?.setData([])
      marketSeriesRef.current?.setData([])
      episodeMarkersByIdRef.current.clear()
      pSeriesRef.current.setMarkers([])
      return
    }
    // lightweight-charts requires data strictly ascending by time; dedupe by keeping last point per time
    const sorted = [...valid].sort((a, b) => a.t - b.t)
    const deduped: Point[] = []
    for (let i = 0; i < sorted.length; i++) {
      const pt = sorted[i]
      if (deduped.length === 0 || pt.t > deduped[deduped.length - 1].t) deduped.push(pt)
      else if (pt.t === deduped[deduped.length - 1].t) deduped[deduped.length - 1] = pt
    }
    const utc = (t: number) => t as import('lightweight-charts').UTCTimestamp
    const pData = deduped.map((pt) => ({ time: utc(pt.t), value: pt.p }))
    const loData = deduped.map((pt) => ({
      time: utc(pt.t),
      value: pt.series_low ?? pt.lo,
    }))
    const hiData = deduped.map((pt) => ({
      time: utc(pt.t),
      value: pt.series_high ?? pt.hi,
    }))
    const railLoData = deduped.map((pt) => {
      const seriesLo = pt.series_low ?? pt.lo
      const mapLo = pt.map_low ?? pt.rail_low ?? seriesLo
      return { time: utc(pt.t), value: mapLo }
    })
    const railHiData = deduped.map((pt) => {
      const seriesHi = pt.series_high ?? pt.hi
      const mapHi = pt.map_high ?? pt.rail_high ?? seriesHi
      return { time: utc(pt.t), value: mapHi }
    })
    const marketData = deduped.filter((pt) => pt.m != null).map((pt) => ({ time: utc(pt.t), value: pt.m as number }))
    pSeriesRef.current.setData(pData)
    loSeriesRef.current.setData(loData)
    hiSeriesRef.current.setData(hiData)
    railLoSeriesRef.current?.setData(railLoData)
    railHiSeriesRef.current?.setData(railHiData)
    marketSeriesRef.current?.setData(marketData)
    buildEpisodeMarkersFromHistory(deduped)
    chartInstanceRef.current?.timeScale().fitContent()
  }, [buildEpisodeMarkersFromHistory])

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

  // Resize chart when drawer opens/closes (or chart becomes ready)
  useEffect(() => {
    if (!chartReady || !chartRef.current || !chartInstanceRef.current) return
    chartInstanceRef.current.applyOptions({
      width: chartRef.current.clientWidth,
      height: chartRef.current.clientHeight || 300,
    })
    chartInstanceRef.current.timeScale().fitContent()
  }, [activePanel, chartReady])

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
      lineWidth: 1,
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

    const onCrosshair = (param: { time?: unknown }) => {
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
    }
    chart.subscribeCrosshairMove(onCrosshair)

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
      try {
        chart.unsubscribeCrosshairMove(onCrosshair)
      } catch {
        // ignore
      }
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

  // IMPORTANT: /api/v1/state/current does NOT include history; fetch history endpoint too.
  const refreshSegmentFromBackend = useCallback(
    async (newSeg: number) => {
      try {
        const oldSeg = currentSegRef.current
        // Temporary debug for seg transitions
        // eslint-disable-next-line no-console
        console.debug('seg-change: refreshing from backend', { oldSeg, newSeg })

        const [curResp, histResp] = await Promise.all([
          fetch(`${API_BASE}/api/v1/state/current`),
          fetch(`${API_BASE}/api/v1/state/history?limit=2000`),
        ])

        if (!curResp.ok || !histResp.ok) {
          // eslint-disable-next-line no-console
          console.debug('seg-change: refresh failed', {
            currentStatus: curResp.status,
            historyStatus: histResp.status,
          })
          return
        }

        const [cur, historyRaw] = await Promise.all([curResp.json(), histResp.json()])
        const history = Array.isArray(historyRaw) ? (historyRaw as Point[]) : []
        const filtered = showFullMatchRef.current ? history : filterHistoryToSeg(history, newSeg)

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
          const seg = (msg.current as { state?: { segment_id?: number; last_frame?: LastFrame } })?.state?.segment_id ?? 0
          currentSegRef.current = seg
          const rawHist = Array.isArray(msg.history) ? msg.history : []
          const hist = rawHist.filter((pt: unknown): pt is Point => isValidPoint(pt))
          const visibleHistory = showFullMatchRef.current ? hist : filterHistoryToSeg(hist, seg)
          setSnapshotHistory(visibleHistory)
          if (pSeriesRef.current && visibleHistory.length > 0) setDataFromHistory(visibleHistory)
          const lf = (msg.current as { state?: { last_frame?: LastFrame } } | undefined)?.state?.last_frame ?? null
          setHudFrame(lf != null && typeof lf === 'object' ? lf : null)
        } else if (msg.type === 'frame' && msg.frame) {
          setHudFrame(typeof msg.frame === 'object' && msg.frame != null ? (msg.frame as LastFrame) : null)
        } else if (msg.type === 'point' && msg.point) {
          const pt = msg.point as Point
          if (!isValidPoint(pt)) return
          if (pt.event != null) handleEventMarker(pt)
          setCurrent((prev) => ({
            ...(prev != null && typeof prev === 'object' ? prev : {}),
            state: prev != null && typeof prev === 'object' ? (prev as { state?: unknown }).state : undefined,
            derived: { ...(prev != null && typeof prev === 'object' && (prev as { derived?: object }).derived != null ? (prev as { derived: object }).derived : {}), p_hat: pt.p },
          }))
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
  }, [wsReconnectTrigger, applyPointToChart, refreshSegmentFromBackend, setDataFromHistory, handleEventMarker])

  // Status label for BO3 list: telemetry proof from readiness probe
  const bo3MatchStatusLabel = (m: Bo3Match): string => {
    const r = bo3Readiness[m.id]
    if (r?.telemetry_ready) return 'Telemetry ✅'
    if (r) return `No telemetry (${r.reason || r.status_code})`
    return '…'
  }

  // Filter: default show only telemetry_ready; toggle shows all candidates
  const bo3DisplayMatches = bo3ShowAllCandidates
    ? liveMatches
    : liveMatches.filter((m) => bo3Readiness[m.id]?.telemetry_ready === true)

  // Drawer content (we reuse your existing sections unchanged, just moved)
  const Bo3Panel = (
    <section style={{ padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
      <h3 style={{ marginTop: 0 }}>BO3</h3>
      <p style={{ marginTop: 0 }}>
        <button
          type="button"
          onClick={async () => {
            try {
              const r = await fetch(`${API_BASE}/api/v1/bo3/candidates`)
              const data = await r.json()
              setLiveMatches(Array.isArray(data) ? data : [])
              setBo3ListLoaded(true)
            } catch {
              // keep previous list on error
            }
          }}
        >
          Load BO3 candidates
        </button>
        {bo3ListLoaded && (
          <span style={{ marginLeft: 8, fontSize: 11, color: '#9ca3af' }}>
            Candidates + readiness every 60s (pinned: only pinned id probed)
          </span>
        )}
      </p>
      {liveMatches.length > 0 && (
        <p style={{ marginTop: 4, marginBottom: 4 }}>
          <label>
            <input
              type="checkbox"
              checked={bo3ShowAllCandidates}
              onChange={(e) => setBo3ShowAllCandidates(e.target.checked)}
            />
            {' '}
            Show all candidates
          </label>
          <span style={{ marginLeft: 8, fontSize: 11, color: '#9ca3af' }}>
            {bo3ShowAllCandidates ? 'Showing all' : 'Telemetry ✅ only'}
          </span>
        </p>
      )}
      <p>
        <label>
          Match:{' '}
          <select value={selectedMatchId} onChange={(e) => setSelectedMatchId(e.target.value)} style={{ minWidth: 240 }}>
            <option value="">—</option>
            {bo3DisplayMatches.map((m) => (
              <option key={m.id} value={String(m.id)}>
                {m.team1_name} vs {m.team2_name} (bo{m.bo_type}) · {bo3MatchStatusLabel(m)}
              </option>
            ))}
          </select>
        </label>
      </p>
      {liveMatches.length > 0 && (
        <p style={{ fontSize: 11, color: '#6b7280', marginTop: -4, marginBottom: 8 }}>
          Telemetry ✅ = snapshot ready; status from probe (not BO3 live/current)
        </p>
      )}
      <p>
        <label>
          Team A is:{' '}
          <select value={teamAIsTeamOne ? 'team1' : 'team2'} onChange={(e) => setTeamAIsTeamOne(e.target.value === 'team1')}>
            <option value="team1">Team 1</option>
            <option value="team2">Team 2</option>
          </select>
        </label>
      </p>
      <p>
        <button
          type="button"
          disabled={!selectedMatchId || !Number.isFinite(Number(selectedMatchId)) || !Number.isInteger(Number(selectedMatchId))}
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
                const msg = (body as { detail?: string })?.detail ?? (body as { message?: string })?.message ?? r.statusText
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
      {configError && <p style={{ color: '#ef4444', fontSize: 13, margin: 0 }}>{configError}</p>}

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
          const extraBits = [agePart, failsPart].filter(Boolean)
          const extra = extraBits.length ? ` (${extraBits.join(', ')})` : ''
          return (
            <p style={{ fontSize: 12, color: '#9ca3af', marginBottom: 0 }}>
              BO3 health: <strong style={{ color: '#e5e7eb' }}>{health}</strong>
              {reason}
              {extra}
            </p>
          )
        }
        if (status == null) return null
        return (
          <p style={{ fontSize: 12, color: '#9ca3af', marginBottom: 0 }}>
            BO3 status: <strong style={{ color: '#e5e7eb' }}>{status}</strong>
            {err != null && err !== '' && <> ({err})</>}
          </p>
        )
      })()}
    </section>
  )

  const PrematchPanel = (
    <section style={{ padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
      <h3 style={{ marginTop: 0 }}>Prematch</h3>
      <p style={{ marginTop: 0 }}>
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
        </label>{' '}
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
        </button>{' '}
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
          Unlock
        </button>{' '}
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
        const config = (current?.state as {
          config?: { prematch_series?: number | null; prematch_map?: number | null; prematch_locked?: boolean }
        })?.config
        const ps = config?.prematch_series
        const pm = config?.prematch_map
        const locked = config?.prematch_locked ?? false
        if (ps == null && pm == null) {
          return (
            <p style={{ fontSize: 12, color: '#9ca3af', margin: 0 }}>
              No prematch set. Use 0.01–0.99 and Set prematch (lock).
            </p>
          )
        }
        return (
          <p style={{ fontSize: 12, color: '#9ca3af', margin: 0 }}>
            prematch_series: <strong style={{ color: '#e5e7eb' }}>{ps != null ? ps.toFixed(4) : '—'}</strong>
            {' · '}
            prematch_map: <strong style={{ color: '#e5e7eb' }}>{pm != null ? pm.toFixed(4) : '—'}</strong>
            {' · '}
            locked: <strong style={{ color: '#e5e7eb' }}>{locked ? 'yes' : 'no'}</strong>
          </p>
        )
      })()}
    </section>
  )

  const MarketPanel = (
    <section style={{ padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
      <h3 style={{ marginTop: 0 }}>Market (Kalshi)</h3>
      <p style={{ marginTop: 0 }}>
        <label>
          Kalshi URL:{' '}
          <input
            type="text"
            value={kalshiUrl}
            onChange={(e) => setKalshiUrl(e.target.value)}
            placeholder="https://kalshi.com/..."
            style={{ width: '100%' }}
          />
        </label>{' '}
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
              const opts = Array.isArray((data as any).options) ? ((data as any).options as any[]) : []
              setMarketOptions(opts as any)
              if (opts.length > 0 && (data as any).suggested) setSelectedMarketKey((data as any).suggested as string)
              else if (opts.length > 0) setSelectedMarketKey(String((opts[0] as any).key))
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
        <p style={{ marginTop: 0 }}>
          <label>
            Team / side:{' '}
            <select value={selectedMarketKey} onChange={(e) => setSelectedMarketKey(e.target.value)} style={{ minWidth: 200 }}>
              {marketOptions.map((opt) => (
                <option key={opt.key} value={opt.key}>
                  {opt.label || opt.key}
                </option>
              ))}
            </select>
          </label>{' '}
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

      {marketError && (
        <p style={{ color: '#ef4444', fontSize: 13, margin: 0 }}>
          {marketError}
        </p>
      )}
    </section>
  )

  const ReplayPanel = (
    <section style={{ padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
      <h3 style={{ marginTop: 0 }}>Replay (JSONL)</h3>
      <p style={{ marginTop: 0 }}>
        <label>
          Source:{' '}
          <select
            value={replayPath}
            onChange={(e) => setReplayPath(e.target.value)}
            style={{ minWidth: 220, maxWidth: '100%' }}
          >
            {replaySources.length === 0 ? (
              <option value={replayPath}>{replayPath || '—'}</option>
            ) : (
              replaySources.map((s) => (
                <option key={s.path} value={s.path}>
                  {s.label}
                </option>
              ))
            )}
          </select>
        </label>
        {replaySources.length === 0 && (
          <span style={{ fontSize: 12, color: '#9ca3af', marginLeft: 8 }}>No logs/ files; using path above.</span>
        )}
      </p>
      <p style={{ marginTop: 0 }}>
        <label>
          Match:{' '}
          <select
            value={replayMatchId}
            onChange={(e) => {
              const id = e.target.value
              setReplayMatchId(id)
              const m = replayMatches.find((x) => String(x.match_id) === id)
              if (m?.path) setReplayPath(m.path)
            }}
            style={{ minWidth: 200 }}
          >
            <option value="">—</option>
            {replayMatches.map((m) => (
              <option key={m.match_id} value={String(m.match_id)}>
                {m.match_id} — {m.team1} vs {m.team2} ({m.count} ticks)
              </option>
            ))}
          </select>
        </label>
        {replayMatchId && (() => {
          const m = replayMatches.find((x) => String(x.match_id) === replayMatchId)
          return m ? (
            <span style={{ marginLeft: 8, color: '#9ca3af', fontSize: 13 }}>
              {m.team1} vs {m.team2} · {m.count} ticks
            </span>
          ) : null
        })()}
      </p>
      <p style={{ marginTop: 0 }}>
        <label>
          Speed:{' '}
          <input type="number" min={0.1} step={0.5} value={replaySpeed} onChange={(e) => setReplaySpeed(Number(e.target.value) || 1)} style={{ width: 60 }} />
        </label>{' '}
        <label>
          <input type="checkbox" checked={replayLoop} onChange={(e) => setReplayLoop(e.target.checked)} /> Loop
        </label>
      </p>
      <p style={{ marginTop: 0 }}>
        <button
          type="button"
            onClick={async () => {
            setReplayError(null)
            const selectedMatch = replayMatchId ? replayMatches.find((x) => String(x.match_id) === replayMatchId) : null
            const pathToLoad = selectedMatch?.path ?? replayPath
            const pathSent = pathToLoad || replayPath
            try {
              const r = await fetch(`${API_BASE}/api/v1/replay/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  path: pathSent,
                  match_id: replayMatchId ? Number(replayMatchId) : undefined,
                  speed: replaySpeed,
                  loop: replayLoop,
                  midround_v2_weight_profile: midroundV2WeightProfile,
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
        </button>{' '}
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
          Stop
        </button>
      </p>
      {replayError && (
        <p style={{ color: '#ef4444', fontSize: 13, margin: 0 }}>
          {replayError}
        </p>
      )}
    </section>
  )

  const DEAD_S = 90
  const STALE_S = 60
  const sessionStatusBadge = (row: TelemetrySessionRow): 'LIVE' | 'STALE' | 'DEAD' | 'TELEM LOST' => {
    const fetchAge = row.fetch_age_s ?? row.age_s ?? 9999
    if (row.last_error) return 'DEAD'
    if (fetchAge >= DEAD_S) return 'DEAD'
    if (row.telemetry_ok === false) return 'TELEM LOST'
    const goodAge = row.good_age_s ?? row.age_s ?? 9999
    if (goodAge >= STALE_S) return 'STALE'
    return 'LIVE'
  }

  const sessionsFiltered = ((): TelemetrySessionRow[] => {
    const raw = telemetrySessions?.sessions ?? []
    let list = raw
    if (telemetryFilterSource !== 'all') {
      list = list.filter((s) => s.source === telemetryFilterSource)
    }
    if (telemetryFilterStatus !== 'all') {
      list = list.filter((s) => sessionStatusBadge(s) === telemetryFilterStatus)
    }
    if (telemetrySearch.trim()) {
      const q = telemetrySearch.trim().toLowerCase()
      list = list.filter((s) => {
        const key = (s.session_key ?? '').toLowerCase()
        const id = (s.id ?? '').toLowerCase()
        const ctxStr = JSON.stringify(s.ctx ?? {}).toLowerCase()
        return key.includes(q) || id.includes(q) || ctxStr.includes(q)
      })
    }
    const col = telemetrySortColumn
    const dir = telemetrySortDir === 'asc' ? 1 : -1
    list = [...list].sort((a, b) => {
      const aVal = col === 'age_s' ? (a.age_s ?? 9999) : col === 'last_update_ts' ? (a.last_update_ts ?? 0) : col === 'source' ? (a.source ?? '') : col === 'id' ? (a.id ?? '') : (a.session_key ?? '')
      const bVal = col === 'age_s' ? (b.age_s ?? 9999) : col === 'last_update_ts' ? (b.last_update_ts ?? 0) : col === 'source' ? (b.source ?? '') : col === 'id' ? (b.id ?? '') : (b.session_key ?? '')
      if (typeof aVal === 'number' && typeof bVal === 'number') return (aVal - bVal) * dir
      return String(aVal).localeCompare(String(bVal)) * dir
    })
    return list
  })()

  const handleSort = (column: string) => {
    setTelemetrySortColumn(column)
    setTelemetrySortDir((prev) => (telemetrySortColumn === column && prev === 'asc' ? 'desc' : 'asc'))
  }

  const pinSession = useCallback(async (source: string, id: string) => {
    setSelectedSession({ source, id })
    setConfigError(null)
    try {
      const r = await fetch(`${API_BASE}/api/v1/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ primary_session_source: source, primary_session_id: id }),
      })
      if (!r.ok) {
        const body = await r.json().catch(() => ({}))
        setConfigError(String((body as { detail?: string })?.detail ?? (body as { message?: string })?.message ?? r.statusText))
        return
      }
      setWsReconnectTrigger((prev) => prev + 1)
    } catch (e) {
      setConfigError(e instanceof Error ? e.message : String(e))
    }
  }, [])
  const unpinSession = useCallback(async () => {
    setSelectedSession(null)
    setConfigError(null)
    try {
      const r = await fetch(`${API_BASE}/api/v1/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ primary_session_source: null, primary_session_id: null }),
      })
      if (!r.ok) {
        const body = await r.json().catch(() => ({}))
        setConfigError(String((body as { detail?: string })?.detail ?? (body as { message?: string })?.message ?? r.statusText))
        return
      }
      setWsReconnectTrigger((prev) => prev + 1)
    } catch (e) {
      setConfigError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const TelemetrySessionsPanel = (
    <section style={{ padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
      <h3 style={{ marginTop: 0 }}>Telemetry Sessions</h3>
      <p style={{ marginTop: 0, fontSize: 12, color: '#9ca3af' }}>
        GET /debug/telemetry/sessions · poll 5s
      </p>
      {(() => {
        const d = telemetrySessions
        const hasBo3 = (d?.bo3_auto_match_ids?.length ?? 0) > 0 || d?.bo3_auto_track_enabled
        const hasGrid = (d?.grid_auto_series_ids?.length ?? 0) > 0 || d?.grid_auto_track_enabled
        if (!hasBo3 && !hasGrid) return null
        return (
          <div style={{ marginBottom: 12, padding: 8, background: '#1f2937', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#e5e7eb' }}>Auto-track</strong>
            {d?.bo3_auto_track_enabled && (
              <div style={{ marginTop: 4 }}>
                BO3: {Array.isArray(d.bo3_auto_match_ids) ? d.bo3_auto_match_ids.join(', ') : '—'}
                {d.bo3_auto_last_refresh_age_s != null && ` · refresh ${d.bo3_auto_last_refresh_age_s}s ago`}
                {d.bo3_readiness_cache_size != null && ` · cache ${d.bo3_readiness_cache_size}`}
              </div>
            )}
            {d?.grid_auto_track_enabled && (
              <div style={{ marginTop: 4 }}>
                GRID: {Array.isArray(d.grid_auto_series_ids) ? d.grid_auto_series_ids.join(', ') : '—'}
                {d.grid_auto_last_refresh_age_s != null && ` · refresh ${d.grid_auto_last_refresh_age_s}s ago`}
                {d.grid_auto_track_limit != null && ` · limit ${d.grid_auto_track_limit}`}
              </div>
            )}
          </div>
        )
      })()}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 8, alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 12, color: '#9ca3af' }}>Source</span>
          <select value={telemetryFilterSource} onChange={(e) => setTelemetryFilterSource(e.target.value as 'all' | 'BO3' | 'GRID')} style={{ padding: '2px 6px', fontSize: 12 }}>
            <option value="all">All</option>
            <option value="BO3">BO3</option>
            <option value="GRID">GRID</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 12, color: '#9ca3af' }}>Status</span>
          <select value={telemetryFilterStatus} onChange={(e) => setTelemetryFilterStatus(e.target.value as 'all' | 'LIVE' | 'STALE' | 'DEAD' | 'TELEM LOST')} style={{ padding: '2px 6px', fontSize: 12 }}>
            <option value="all">All</option>
            <option value="LIVE">LIVE</option>
            <option value="STALE">STALE</option>
            <option value="TELEM LOST">TELEM LOST</option>
            <option value="DEAD">DEAD</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 12, color: '#9ca3af' }}>Search</span>
          <input type="text" value={telemetrySearch} onChange={(e) => setTelemetrySearch(e.target.value)} placeholder="id / key / ctx…" style={{ width: 140, padding: '2px 6px', fontSize: 12 }} />
        </label>
        <span style={{ marginLeft: 8, fontSize: 12, color: '#9ca3af' }}>Click a row to pin that match (runner will not rotate).</span>
        <button
          type="button"
          disabled={!selectedSession}
          onClick={() => selectedSession && pinSession(selectedSession.source, selectedSession.id)}
          style={{ padding: '4px 10px', fontSize: 12, borderRadius: 6, cursor: selectedSession ? 'pointer' : 'not-allowed', opacity: selectedSession ? 1 : 0.6 }}
        >
          Run this match
        </button>
        <button
          type="button"
          onClick={() => unpinSession()}
          style={{ padding: '4px 10px', fontSize: 12, borderRadius: 6, cursor: 'pointer' }}
        >
          Stop match
        </button>
      </div>
      <div style={{ overflowX: 'auto', fontSize: 11 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', color: '#e5e7eb' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #374151' }}>
              <th style={{ width: 28, padding: '6px 4px', textAlign: 'center' }} title="Select to run">Use</th>
              {(['matchup', 'status', 'source', 'age_s', 'last_update_ts', 'active_source', 'chosen_source', 'key_display', 'health', 'next_fetch_in_s'] as const).map((col) => {
                const sortCol = col === 'status' ? 'age_s' : ['matchup', 'source', 'age_s', 'last_update_ts'].includes(col) ? (col === 'matchup' ? 'session_key' : col) : null
                return (
                  <th key={col} style={{ textAlign: 'left', padding: '6px 4px', whiteSpace: 'nowrap', cursor: sortCol ? 'pointer' : 'default' }} onClick={sortCol ? () => handleSort(sortCol) : undefined}>
                    {col === 'matchup'
                      ? 'match'
                      : col === 'key_display'
                        ? 'last_env.key'
                        : col === 'health'
                          ? 'per_source_health'
                          : col === 'next_fetch_in_s'
                            ? 'grid next_fetch_s'
                            : col}
                    {sortCol && telemetrySortColumn === sortCol ? (telemetrySortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {sessionsFiltered.length === 0 && (
              <tr><td colSpan={12} style={{ padding: 12, color: '#9ca3af' }}>No sessions (or none match filters).</td></tr>
            )}
            {sessionsFiltered.map((row) => {
              const status = sessionStatusBadge(row)
              const badgeColor = status === 'LIVE' ? '#22c55e' : status === 'STALE' ? '#f59e0b' : '#ef4444'
              const ctx = row.ctx ?? {}
              const sel = ctx.selector_decision
              const chosen = sel && typeof sel === 'object' && 'chosen_source' in sel ? (sel as { chosen_source?: string }).chosen_source : null
              const keyDisplay = ctx.last_env && typeof ctx.last_env === 'object' && 'key_display' in ctx.last_env ? (ctx.last_env as { key_display?: string }).key_display : null
              const health = ctx.per_source_health
              const healthStr = health && typeof health === 'object' ? Object.entries(health).map(([k, v]) => `${k}: ok=${(v as any)?.ok_count ?? 0} err=${(v as any)?.err_count ?? 0}${(v as any)?.last_reason ? ` ${(v as any).last_reason}` : ''}`).join('; ') : '—'
              const cfg = (current?.state as { config?: { primary_session_source?: string; primary_session_id?: string } } | undefined)?.config
              const isRunning = cfg?.primary_session_source === row.source && cfg?.primary_session_id === row.id
              const isSelected = selectedSession?.source === row.source && selectedSession?.id === row.id
              const lastFrame = (ctx as any)?.last_frame as { teams?: [string, string] } | undefined
              const teamsTuple = Array.isArray(lastFrame?.teams) ? lastFrame?.teams : undefined
              const matchLabel =
                teamsTuple && teamsTuple.length === 2
                  ? `${teamsTuple[0] || 'Team A'} vs ${teamsTuple[1] || 'Team B'}`
                  : String(row.session_key ?? '').slice(0, 20) || row.id || row.source || 'Session'
              return (
                <tr
                  key={`${row.source}:${row.id}`}
                  style={{ borderBottom: '1px solid #374151', cursor: 'pointer', background: isSelected ? 'rgba(59, 130, 246, 0.15)' : isRunning ? 'rgba(34, 197, 94, 0.08)' : undefined }}
                  onClick={() => pinSession(row.source, row.id)}
                >
                  <td style={{ padding: '4px 4px', textAlign: 'center', verticalAlign: 'middle' }} onClick={(e) => e.stopPropagation()}>
                    <button
                      type="button"
                      aria-label={isSelected ? 'Pinned session' : 'Pin this session'}
                      title={isRunning ? 'Pinned (click to keep; Stop match to unpin)' : isSelected ? 'Pinned' : 'Click to pin this match'}
                      style={{
                        width: 18,
                        height: 18,
                        borderRadius: '50%',
                        border: `2px solid ${isSelected ? '#3b82f6' : '#6b7280'}`,
                        background: isSelected ? '#3b82f6' : 'transparent',
                        cursor: 'pointer',
                        padding: 0,
                      }}
                      onClick={() => (isSelected ? unpinSession() : pinSession(row.source, row.id))}
                    />
                  </td>
                  <td style={{ padding: '4px 4px' }} title={matchLabel}>
                    {matchLabel}
                    {isRunning ? <span style={{ marginLeft: 4, fontSize: 10, color: '#22c55e', fontWeight: 600 }} title="Pinned primary">● Pinned</span> : ''}
                  </td>
                  <td style={{ padding: '4px 4px' }}><span style={{ background: badgeColor, color: '#fff', padding: '1px 6px', borderRadius: 4, fontSize: 10 }} title={statusTitle}>{status}</span></td>
                  <td style={{ padding: '4px 4px' }}>{row.source ?? '—'}</td>
                  <td style={{ padding: '4px 4px' }}>{row.age_s != null ? row.age_s : '—'}</td>
                  <td style={{ padding: '4px 4px' }}>{row.last_update_ts != null ? new Date(row.last_update_ts * 1000).toISOString().slice(11, 19) : '—'}</td>
                  <td style={{ padding: '4px 4px' }}>{ctx.active_source ?? '—'}</td>
                  <td style={{ padding: '4px 4px' }}>{chosen ?? '—'}</td>
                  <td style={{ padding: '4px 4px', maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis' }} title={keyDisplay ?? ''}>{keyDisplay ?? '—'}</td>
                  <td style={{ padding: '4px 4px', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }} title={healthStr}>{healthStr.slice(0, 40)}{healthStr.length > 40 ? '…' : ''}</td>
                  <td style={{ padding: '4px 4px' }}>{row.grid_schedule?.next_fetch_in_s != null ? row.grid_schedule.next_fetch_in_s : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )

  const cfg = (current?.state as { config?: Record<string, unknown> } | undefined)?.config ?? {} as Record<string, unknown>
  const postConfigPartial = async (partial: Record<string, unknown>) => {
    setConfigError(null)
    try {
      const r = await fetch(`${API_BASE}/api/v1/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(partial),
      })
      if (!r.ok) {
        const body = await r.json().catch(() => ({}))
        setConfigError(String((body as { detail?: string })?.detail ?? (body as { message?: string })?.message ?? r.statusText))
        return
      }
      setWsReconnectTrigger((prev) => prev + 1)
    } catch (e) {
      setConfigError(e instanceof Error ? e.message : String(e))
    }
  }

  const clampLimit = (v: number) => Math.max(0, Math.min(50, Math.round(v)))
  const clampRefreshS = (v: number) => Math.max(10, Number(v))
  const clampProbeBudget = (v: number) => Math.max(5, Math.min(200, Math.round(v)))

  const TelemetryControlsPanel = (
    <section style={{ padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
      <h3 style={{ marginTop: 0 }}>Telemetry Controls</h3>
      <p style={{ marginTop: 0, fontSize: 12, color: '#9ca3af' }}>Config from current.state.config · POST /api/v1/config</p>

      {/* BO3 */}
      <div style={{ marginTop: 12, paddingBottom: 12, borderBottom: '1px solid #374151' }}>
        <strong style={{ color: '#9ca3af', fontSize: 12 }}>BO3</strong>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
          <input
            type="checkbox"
            checked={!!cfg.bo3_auto_track}
            onChange={(e) => postConfigPartial({ bo3_auto_track: e.target.checked })}
          />
          <span style={{ fontSize: 12 }}>bo3_auto_track</span>
        </label>
        <label style={{ display: 'block', marginTop: 6, fontSize: 12 }}>
          bo3_auto_track_limit (0–50)
          <input
            type="number"
            min={0}
            max={50}
            value={(() => { const v = Number(cfg.bo3_auto_track_limit ?? 5); return Number.isFinite(v) ? v : 5; })()}
            onChange={(e) => postConfigPartial({ bo3_auto_track_limit: clampLimit(Number(e.target.value) || 0) })}
            style={{ marginLeft: 8, width: 56, padding: '2px 4px' }}
          />
        </label>
        <label style={{ display: 'block', marginTop: 6, fontSize: 12 }}>
          bo3_auto_track_refresh_s (min 10)
          <input
            type="number"
            min={10}
            step={1}
            value={(() => { const v = Number(cfg.bo3_auto_track_refresh_s ?? 30); return Number.isFinite(v) ? v : 30; })()}
            onChange={(e) => postConfigPartial({ bo3_auto_track_refresh_s: clampRefreshS(Number(e.target.value) || 10) })}
            style={{ marginLeft: 8, width: 56, padding: '2px 4px' }}
          />
        </label>
        <label style={{ display: 'block', marginTop: 6, fontSize: 12 }}>
          bo3_auto_track_probe_budget (5–200)
          <input
            type="number"
            min={5}
            max={200}
            value={(() => { const v = Number(cfg.bo3_auto_track_probe_budget ?? 40); return Number.isFinite(v) ? v : 40; })()}
            onChange={(e) => postConfigPartial({ bo3_auto_track_probe_budget: clampProbeBudget(Number(e.target.value) || 40) })}
            style={{ marginLeft: 8, width: 56, padding: '2px 4px' }}
          />
        </label>
        <div style={{ marginTop: 8, fontSize: 12 }}>
          <span style={{ color: '#9ca3af' }}>bo3_match_ids (manual)</span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
            {(Array.isArray(cfg.bo3_match_ids) ? cfg.bo3_match_ids : []).map((id: unknown) => (
              <span
                key={String(id)}
                style={{ background: '#374151', padding: '2px 8px', borderRadius: 4, fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}
              >
                {String(id)}
                <button type="button" onClick={() => postConfigPartial({ bo3_match_ids: (cfg.bo3_match_ids as unknown[]).filter((x) => x !== id) })} style={{ background: 'transparent', border: 'none', color: '#9ca3af', cursor: 'pointer' }}>×</button>
              </span>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
            <input
              type="number"
              placeholder="Match ID"
              id="bo3-pin-input"
              style={{ width: 80, padding: '2px 4px', fontSize: 12 }}
              onKeyDown={(e) => {
                if (e.key !== 'Enter') return
                const input = document.getElementById('bo3-pin-input') as HTMLInputElement
                const v = parseInt(input?.value ?? '', 10)
                if (!Number.isInteger(v) || v < 0) return
                const existing = Array.isArray(cfg.bo3_match_ids) ? cfg.bo3_match_ids as number[] : []
                const next = [...existing.filter((x) => x !== v), v]
                postConfigPartial({ bo3_match_ids: next })
                input.value = ''
              }}
            />
            <button type="button" onClick={() => {
              const input = document.getElementById('bo3-pin-input') as HTMLInputElement
              const v = parseInt(input?.value ?? '', 10)
              if (!Number.isInteger(v) || v < 0) return
              const existing = Array.isArray(cfg.bo3_match_ids) ? (cfg.bo3_match_ids as number[]) : []
              const next = [...existing.filter((x) => x !== v), v]
              postConfigPartial({ bo3_match_ids: next })
              input.value = ''
            }} style={{ padding: '2px 8px', fontSize: 12 }}>Add BO3 ID</button>
          </div>
        </div>
      </div>

      {/* Market delay: only market quote (bid/ask/mid) is delayed; chart/telemetry are real-time */}
      <div style={{ marginTop: 12, paddingBottom: 12, borderBottom: '1px solid #374151' }}>
        <strong style={{ color: '#9ca3af', fontSize: 12 }}>Market delay</strong>
        <p style={{ marginTop: 2, marginBottom: 6, fontSize: 11, color: '#6b7280' }}>
          Only the market quote attached to each point is delayed; chart and telemetry are real-time.
        </p>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
          <span>Delay (seconds):</span>
          <select
            value={(() => {
              const sec = Math.max(0, Math.min(300, Number(cfg.market_delay_sec) ?? 120))
              const opts = [0, 30, 60, 120, 180]
              return opts.includes(sec) ? sec : opts.reduce((a, b) => (Math.abs(b - sec) <= Math.abs(a - sec) ? b : a))
            })()}
            onChange={(e) => {
              const sec = Math.max(0, Math.min(300, parseInt(e.target.value, 10) || 0))
              postConfigPartial({ market_delay_sec: sec })
            }}
            style={{ padding: '2px 6px', fontSize: 12, minWidth: 72 }}
          >
            <option value={0}>Off (live)</option>
            <option value={30}>30 s</option>
            <option value={60}>60 s</option>
            <option value={120}>120 s</option>
            <option value={180}>180 s</option>
          </select>
        </label>
      </div>

      {/* GRID */}
      <div style={{ marginTop: 12, paddingBottom: 12, borderBottom: '1px solid #374151' }}>
        <strong style={{ color: '#9ca3af', fontSize: 12 }}>GRID</strong>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
          <input
            type="checkbox"
            checked={!!cfg.grid_auto_track}
            onChange={(e) => postConfigPartial({ grid_auto_track: e.target.checked })}
          />
          <span style={{ fontSize: 12 }}>grid_auto_track</span>
        </label>
        <label style={{ display: 'block', marginTop: 6, fontSize: 12 }}>
          grid_auto_track_limit (0–50)
          <input
            type="number"
            min={0}
            max={50}
            value={(() => { const v = Number(cfg.grid_auto_track_limit ?? 5); return Number.isFinite(v) ? v : 5; })()}
            onChange={(e) => postConfigPartial({ grid_auto_track_limit: clampLimit(Number(e.target.value) || 0) })}
            style={{ marginLeft: 8, width: 56, padding: '2px 4px' }}
          />
        </label>
        <label style={{ display: 'block', marginTop: 6, fontSize: 12 }}>
          grid_auto_track_refresh_s (min 10)
          <input
            type="number"
            min={10}
            step={1}
            value={(() => { const v = Number(cfg.grid_auto_track_refresh_s ?? 60); return Number.isFinite(v) ? v : 60; })()}
            onChange={(e) => postConfigPartial({ grid_auto_track_refresh_s: clampRefreshS(Number(e.target.value) || 10) })}
            style={{ marginLeft: 8, width: 56, padding: '2px 4px' }}
          />
        </label>
        <div style={{ marginTop: 8, fontSize: 12 }}>
          <span style={{ color: '#9ca3af' }}>grid_series_ids (manual)</span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
            {(Array.isArray(cfg.grid_series_ids) ? cfg.grid_series_ids : []).map((id: unknown) => (
              <span
                key={String(id)}
                style={{ background: '#374151', padding: '2px 8px', borderRadius: 4, fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}
              >
                {String(id)}
                <button type="button" onClick={() => postConfigPartial({ grid_series_ids: (cfg.grid_series_ids as unknown[]).filter((x) => x !== id) })} style={{ background: 'transparent', border: 'none', color: '#9ca3af', cursor: 'pointer' }}>×</button>
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Buttons */}
      <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <button
          type="button"
          onClick={() => postConfigPartial({ bo3_auto_track: false, grid_auto_track: false, bo3_match_ids: [], grid_series_ids: [] })}
          style={{ padding: '6px 12px', fontSize: 12, background: '#7f1d1d', color: '#fecaca', border: '1px solid #991b1b', borderRadius: 6, cursor: 'pointer' }}
        >
          Stop all telemetry
        </button>
        <button
          type="button"
          onClick={async () => {
            setConfigError(null)
            try {
              const r = await fetch(`${API_BASE}/api/v1/debug/telemetry/clear_sessions`, { method: 'POST' })
              if (!r.ok) {
                const body = await r.json().catch(() => ({}))
                setConfigError(String((body as { detail?: string })?.detail ?? r.statusText))
                return
              }
              setWsReconnectTrigger((prev) => prev + 1)
            } catch (e) {
              setConfigError(e instanceof Error ? e.message : String(e))
            }
          }}
          style={{ padding: '6px 12px', fontSize: 12, background: '#374151', color: '#e5e7eb', border: '1px solid #4b5563', borderRadius: 6, cursor: 'pointer' }}
        >
          Clear sessions (runtime)
        </button>
        <button
          type="button"
          onClick={async () => {
            setSnapshotHistory([])
            setDataFromHistory([])
            setCurrent(null)
            setHudFrame(null)
            setConfigError(null)
            try {
              const r = await fetch(`${API_BASE}/api/v1/state/clear`, { method: 'POST' })
              if (!r.ok) {
                const body = await r.json().catch(() => ({}))
                setConfigError(String((body as { detail?: string })?.detail ?? r.statusText))
                return
              }
              setWsReconnectTrigger((prev) => prev + 1)
            } catch (e) {
              setConfigError(e instanceof Error ? e.message : String(e))
            }
          }}
          style={{ padding: '6px 12px', fontSize: 12, background: '#374151', color: '#e5e7eb', border: '1px solid #4b5563', borderRadius: 6, cursor: 'pointer' }}
        >
          Clear chart
        </button>
        <button
          type="button"
          onClick={async () => {
            setResetStatus('Resetting...')
            setConfigError(null)
            try {
              const r = await fetch(`${API_BASE}/api/v1/debug/reset`, { method: 'POST' })
              if (!r.ok) {
                const body = await r.json().catch(() => ({}))
                const msg = String((body as { detail?: string })?.detail ?? r.statusText)
                setResetStatus(`Failed: ${msg}`)
                console.error('Reset failed:', msg)
                return
              }
              localStorage.clear()
              sessionStorage.clear()
              location.reload()
            } catch (e) {
              const msg = e instanceof Error ? e.message : String(e)
              setResetStatus(`Failed: ${msg}`)
              console.error('Reset failed:', msg)
            }
          }}
          style={{ padding: '8px 12px', fontSize: 12, background: '#78350f', color: '#fef3c7', border: '1px solid #92400e', borderRadius: 6, cursor: 'pointer', marginTop: 4 }}
        >
          Reset App
        </button>
        {resetStatus && (
          <span style={{ fontSize: 11, color: resetStatus.startsWith('Failed') ? '#ef4444' : '#9ca3af', marginTop: 4 }}>
            {resetStatus}
          </span>
        )}
      </div>

      {/* GRID candidate picker */}
      <div style={{ marginTop: 12, paddingBottom: 12, borderBottom: '1px solid #374151' }}>
        <strong style={{ color: '#9ca3af', fontSize: 12 }}>GRID candidates</strong>
        <button
          type="button"
          onClick={async () => {
            try {
              const r = await fetch(`${API_BASE}/api/v1/grid/candidates?limit=25`)
              const data = await r.json()
              setGridCandidates(Array.isArray(data) ? data : [])
            } catch {
              setGridCandidates([])
            }
          }}
          style={{ marginTop: 6, padding: '4px 8px', fontSize: 12 }}
        >
          Load GRID candidates
        </button>
        {gridCandidates.length > 0 && (
          <div style={{ marginTop: 8, maxHeight: 180, overflowY: 'auto', fontSize: 11 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #374151' }}>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}>series_id</th>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}>name</th>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}>tournament</th>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}>start_time</th>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}>level</th>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}></th>
                </tr>
              </thead>
              <tbody>
                {gridCandidates.map((c) => {
                  const sid = c.series_id ?? ''
                  return (
                    <tr key={sid} style={{ borderBottom: '1px solid #1f2937' }}>
                      <td style={{ padding: '2px 4px' }}>{sid}</td>
                      <td style={{ padding: '2px 4px' }}>{(c as { name?: string }).name ?? '—'}</td>
                      <td style={{ padding: '2px 4px' }}>{(c as { tournament_name?: string }).tournament_name ?? '—'}</td>
                      <td style={{ padding: '2px 4px' }}>{(c as { start_time?: string }).start_time ?? '—'}</td>
                      <td style={{ padding: '2px 4px' }}>{(c as { live_data_feed_level?: string }).live_data_feed_level ?? '—'}</td>
                      <td style={{ padding: '2px 4px' }}>
                        <button
                          type="button"
                          onClick={() => {
                            const existing = Array.isArray(cfg.grid_series_ids) ? (cfg.grid_series_ids as string[]) : []
                            const next = [...existing.filter((x) => x !== sid), sid]
                            postConfigPartial({ grid_series_ids: next })
                          }}
                          style={{ padding: '1px 6px', fontSize: 10 }}
                        >
                          Pin
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {configError && <p style={{ color: '#ef4444', fontSize: 12, marginTop: 8 }}>{configError}</p>}
    </section>
  )

  const DrawerContent =
    activePanel === 'bo3' ? Bo3Panel :
    activePanel === 'prematch' ? PrematchPanel :
    activePanel === 'market' ? MarketPanel :
    activePanel === 'replay' ? ReplayPanel :
    activePanel === 'telemetry' ? TelemetrySessionsPanel :
    activePanel === 'telemetry_controls' ? TelemetryControlsPanel :
    null

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div
        style={{
          padding: 16,
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          overflow: 'hidden',
        }}
      >
        {/* Main row: toolbar + drawer + HUD+chart */}
        <div style={{ display: 'flex', flex: 1, minHeight: 0, marginTop: 12, gap: 12 }}>
          {/* Left toolbar */}
          <div
            style={{
              width: 60,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
              alignItems: 'stretch',
              flexShrink: 0,
            }}
          >
            {(
              [
                { id: 'bo3' as const, label: 'BO3' },
                { id: 'prematch' as const, label: 'Pre' },
                { id: 'market' as const, label: 'Mkt' },
                { id: 'replay' as const, label: 'Rep' },
                { id: 'telemetry' as const, label: 'Sess' },
                { id: 'telemetry_controls' as const, label: 'Ctl' },
              ] as const
            ).map((tab) => {
              const isActive = activePanel === tab.id
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActivePanel((prev) => (prev === tab.id ? null : tab.id))}
                  style={{
                    padding: '6px 6px',
                    fontSize: 11,
                    borderRadius: 6,
                    border: '1px solid ' + (isActive ? '#4b5563' : '#374151'),
                    background: isActive ? '#111827' : '#020617',
                    color: '#e5e7eb',
                    cursor: 'pointer',
                    textAlign: 'center',
                  }}
                >
                  {tab.label}
                </button>
              )
            })}
          </div>

          {/* Drawer (optional) */}
          {activePanel && (
            <div
              style={{
                width: 320,
                minWidth: 280,
                maxWidth: 360,
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
                overflowY: 'auto',
                border: '1px solid #1f2937',
                borderRadius: 8,
                padding: 10,
                fontSize: 13,
                color: '#e5e7eb',
                background: '#020617',
                flexShrink: 0,
              }}
            >
              {/* Temporary: Midround V2 weight profile A/B toggle */}
              <div style={{ padding: '8px 0', borderBottom: '1px solid #374151' }}>
                <label style={{ display: 'block', marginBottom: 4, fontSize: 12, color: '#9ca3af' }}>
                  Midround V2 weight profile
                </label>
                <select
                  value={midroundV2WeightProfile}
                  onChange={async (e) => {
                    const value = e.target.value as 'current' | 'learned_v1' | 'learned_v2' | 'learned_fit'
                    setMidroundV2WeightProfile(value)
                    setConfigError(null)
                    try {
                      const r = await fetch(`${API_BASE}/api/v1/config`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ midround_v2_weight_profile: value }),
                      })
                      if (!r.ok) {
                        const body = await r.json().catch(() => ({}))
                        setConfigError((body as { detail?: string })?.detail ?? r.statusText)
                        return
                      }
                      setWsReconnectTrigger((prev) => prev + 1)
                    } catch (err) {
                      setConfigError(err instanceof Error ? err.message : String(err))
                    }
                  }}
                  style={{ minWidth: 140, padding: '4px 8px', fontSize: 12 }}
                  title="Temporary A/B toggle. learned_v1/v2 rebalance terms (loadout smaller, hp/alive/bomb higher)."
                >
                  <option value="current">current</option>
                  <option value="learned_v1">learned_v1</option>
                  <option value="learned_v2">learned_v2</option>
                  <option value="learned_fit">learned_fit</option>
                </select>
                <p style={{ margin: '4px 0 0', fontSize: 11, color: '#6b7280' }}>
                  Profile: <strong style={{ color: '#e5e7eb' }}>{midroundV2WeightProfile}</strong>
                </p>
                <p style={{ margin: '2px 0 0', fontSize: 10, color: '#4b5563' }}>
                  Temporary; learned_v1 rebalances terms for A/B testing.
                </p>
              </div>
              {DrawerContent}
            </div>
          )}

          {/* Main: HUD + chart */}
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
              <MatchHUD
                frame={hudFrame ?? ((current?.state as { last_frame?: LastFrame } | undefined)?.last_frame ?? null)}
                debug={(current?.derived as { debug?: { bo3_health?: string; bo3_buffer_age_s?: number | null } } | undefined)?.debug}
              />
              <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
                {/* Minimal overlay: status dot + p_hat only */}
                <div
                  style={{
                    position: 'absolute',
                    left: 8,
                    top: 8,
                    zIndex: 10,
                    fontSize: 11,
                    fontFamily: 'monospace',
                    color: '#9ca3af',
                    background: 'rgba(15, 23, 42, 0.95)',
                    padding: '4px 6px',
                    borderRadius: 4,
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    gap: 4,
                  }}
                >
                  <span
                    style={{
                      display: 'inline-block',
                      width: 8,
                      height: 8,
                      borderRadius: '999px',
                      marginRight: 4,
                      backgroundColor:
                        wsStatus === 'open'
                          ? '#22c55e'
                          : wsStatus === 'error'
                          ? '#ef4444'
                          : wsStatus === 'closed'
                          ? '#f97316'
                          : '#6b7280',
                    }}
                  />
                  <span>{wsStatus}</span>
                  {current?.derived?.p_hat != null && (
                    <span>
                      {' · '}p_hat={current.derived.p_hat.toFixed(4)}
                    </span>
                  )}
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
        </div>
      </div>

      <DebugDrawer open={debugOpen} onToggle={() => setDebugOpen((o) => !o)}>
        <section style={{ marginBottom: 12, padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
          <h3 style={{ marginTop: 0 }}>Breaches</h3>
          <p style={{ marginTop: 0 }}>
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
            </button>{' '}
            <span style={{ fontSize: 12, color: '#9ca3af' }}>Last 20 shown</span>
          </p>
          {breaches.length === 0 && (
            <p style={{ fontSize: 13, color: '#9ca3af' }}>No breach events. Click Refresh or wait for auto-poll.</p>
          )}
          {breaches
            .slice(-20)
            .reverse()
            .map((evt, i) => (
              <div
                key={i}
                style={{ fontSize: 12, marginBottom: 6, padding: 6, background: '#1f2937', borderRadius: 6 }}
              >
                <span style={{ color: '#9ca3af' }}>
                  {new Date((evt.ts_epoch ?? 0) * 1000).toISOString().replace('T', ' ').slice(0, 19)}
                </span>{' '}
                <strong>{evt.breach_type}</strong>
                {evt.breach_mag != null && <> mag={evt.breach_mag.toFixed(4)}</>}{' '}
                score {evt.scores?.[0] ?? 0}-{evt.scores?.[1] ?? 0}
                {evt.market_mid != null && (
                  <>
                    {' '}
                    · market_mid={evt.market_mid.toFixed(4)} vs [{evt.map_low?.toFixed(4) ?? '?'},{' '}
                    {evt.map_high?.toFixed(4) ?? '?'}]
                  </>
                )}
              </div>
            ))}
        </section>

        <section style={{ marginBottom: 12, padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
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
            const mid =
              d.midround_v2 != null && typeof d.midround_v2 === 'object' ? (d.midround_v2 as Record<string, unknown>) : d
            const mv = (key: string) => {
              const val = (mid as any)[key]
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
              <pre
                style={{
                  fontSize: 11,
                  fontFamily: 'ui-monospace, monospace',
                  margin: 0,
                  color: '#9ca3af',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                }}
              >
                {lines.join('\n')}
              </pre>
            )
          })()}
        </section>

        <section style={{ marginBottom: 0, padding: 12, border: '1px solid #374151', borderRadius: 6 }}>
          <h3 style={{ marginTop: 0 }}>Debug: Raw BO3 Snapshot</h3>
          <p style={{ marginTop: 0 }}>
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
            {rawSnapshotError && (
              <span style={{ marginLeft: 8, color: '#ef4444', fontSize: 12 }}>
                {rawSnapshotError}
              </span>
            )}
          </p>
          {rawSnapshotJson != null && (
            <div>
              <button type="button" onClick={() => setRawSnapshotOpen((o) => !o)} style={{ fontSize: 12, marginBottom: 4 }}>
                {rawSnapshotOpen ? 'Collapse' : 'Expand'}
              </button>
              {rawSnapshotOpen && (
                <pre
                  style={{
                    fontSize: 10,
                    fontFamily: 'ui-monospace, monospace',
                    margin: 0,
                    color: '#9ca3af',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all',
                    maxHeight: 400,
                    overflow: 'auto',
                    border: '1px solid #374151',
                    padding: 8,
                    borderRadius: 6,
                  }}
                >
                  {rawSnapshotJson}
                </pre>
              )}
            </div>
          )}
        </section>
      </DebugDrawer>
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

function MatchHUD({
  frame,
  debug,
}: {
  frame: LastFrame | null
  debug?: { bo3_health?: string; bo3_buffer_age_s?: number | null }
}) {
  const bo3Health = debug?.bo3_health ?? '—'
  const bufferAgeS = debug?.bo3_buffer_age_s
  const bufferAgeStr = bufferAgeS != null ? `${Math.round(bufferAgeS)}s` : ''

  if (!frame) {
    return (
      <section
        style={{
          flex: '0 0 auto',
          height: 120,
          padding: 12,
          border: '1px solid #374151',
          borderRadius: 4,
          background: '#1f2937',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
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
  const roundPhase = phaseObj && typeof phaseObj === 'object' && phaseObj.round_phase ? String(phaseObj.round_phase) : '—'
  const clockS = frame.round_time_remaining_s
  const clockStr = clockS != null ? `${Math.floor(clockS / 60)}:${String(Math.floor(clockS % 60)).padStart(2, '0')}` : '—'
  const aSide = frame.a_side ? String(frame.a_side).toUpperCase() : null
  const teamASide = frame.team_one?.side ? String(frame.team_one.side).toUpperCase() : aSide
  const teamBSide = frame.team_two?.side
    ? String(frame.team_two.side).toUpperCase()
    : aSide === 'CT'
    ? 'T'
    : aSide === 'T'
    ? 'CT'
    : null

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
  const rowsA: Array<HudPlayerRow | null> = [...playersA]
  const rowsB: Array<HudPlayerRow | null> = [...playersB]
  while (rowsA.length < 5) rowsA.push(null)
  while (rowsB.length < 5) rowsB.push(null)
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
          borderRadius: 6,
        }}
      >
        <span style={{ flex: '0 0 12px', textAlign: 'center' }}>{hasBomb ? '💣' : ''}</span>
        <span style={{ flex: '0 0 12px', textAlign: 'center' }}>{hasKit ? '🛡' : ''}</span>
        <span style={{ flex: '0 0 12px', textAlign: 'center' }}>{hasHelmet || hasKevlar ? '⛑' : ''}</span>
        <span style={{ minWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={name}>
          {name}
        </span>
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

  const PlaceholderRow = () => <div style={{ padding: '4px 8px', fontSize: 12, color: '#6b7280' }}>—</div>

  return (
    <section
      style={{
        flex: '0 0 auto',
        height: 280,
        maxHeight: 320,
        padding: 0,
        border: '1px solid #374151',
        borderRadius: 6,
        background: '#111827',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 12px',
          background: '#1f2937',
          borderBottom: '1px solid #374151',
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        <div style={{ fontSize: 12, color: '#9ca3af' }}>
          <span>BO3</span>
          <span style={{ marginLeft: 8 }}>Map {mapIndex}</span>
          <span style={{ marginLeft: 8 }}>{mapName}</span>
          <span style={{ marginLeft: 8 }}>
            ({seriesScore[0] ?? 0}–{seriesScore[1] ?? 0})
          </span>
        </div>
        <div style={{ fontSize: 12, color: '#9ca3af' }}>
          BO3 health: <strong style={{ color: '#e5e7eb' }}>{bo3Health}</strong>
          {bufferAgeStr && <span style={{ marginLeft: 8 }}>age {bufferAgeStr}</span>}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24, padding: '12px 16px', borderBottom: '1px solid #374151' }}>
        <div style={{ fontSize: 28, fontWeight: 700, color: '#e5e7eb' }}>{teams[0] ?? 'A'}</div>
        <div style={{ fontSize: 32, fontWeight: 800, color: '#fbbf24' }}>
          {scores[0] ?? 0} – {scores[1] ?? 0}
        </div>
        <div style={{ fontSize: 28, fontWeight: 700, color: '#e5e7eb' }}>{teams[1] ?? 'B'}</div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 16, padding: '4px 16px', fontSize: 12, color: '#9ca3af' }}>
        <span>{roundPhase}</span>
        <span>Clock: {clockStr}</span>
      </div>

      <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: '1px solid #374151' }}>
          <div style={{ padding: '6px 8px', fontSize: 13, fontWeight: 600, ...sideColor(teamASide) }}>{teams[0] ?? 'Team A'}</div>
          {hasPlayers ? (
            rowsA.map((p, i) => (p ? <PlayerRow key={i} p={p} index={i} /> : <PlaceholderRow key={i} />))
          ) : (
            <>
              {[0, 1, 2, 3, 4].map((i) => (
                <PlaceholderRow key={i} />
              ))}
              <div style={{ padding: '4px 8px', fontSize: 11, color: '#6b7280', borderTop: '1px solid #374151' }}>
                Alive: {aliveA} · HP: {hpA} · $: {cashA} · Eq: {loadoutA}
              </div>
            </>
          )}
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '6px 8px', fontSize: 13, fontWeight: 600, ...sideColor(teamBSide) }}>{teams[1] ?? 'Team B'}</div>
          {hasPlayers ? (
            rowsB.map((p, i) => (p ? <PlayerRow key={i} p={p} index={i} /> : <PlaceholderRow key={i} />))
          ) : (
            <>
              {[0, 1, 2, 3, 4].map((i) => (
                <PlaceholderRow key={i} />
              ))}
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

type DebugDrawerProps = {
  open: boolean
  onToggle: () => void
  children: React.ReactNode
}

function DebugDrawer({ open, onToggle, children }: DebugDrawerProps) {
  const height = open ? 320 : 0
  return (
    <div style={{ flexShrink: 0, borderTop: '1px solid #374151', background: '#020617' }}>
      <button
        type="button"
        onClick={onToggle}
        style={{
          width: '100%',
          padding: '4px 8px',
          fontSize: 12,
          background: '#111827',
          color: '#e5e7eb',
          border: 'none',
          borderTop: '1px solid #374151',
          cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        Debug {open ? '▼' : '▲'}
      </button>
      <div
        style={{
          height,
          overflow: 'hidden',
          transition: 'height 0.2s ease',
        }}
      >
        <div
          style={{
            height: '100%',
            overflowY: 'auto',
            padding: 12,
          }}
        >
          {children}
        </div>
      </div>
    </div>
  )
}
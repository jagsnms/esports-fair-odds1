import { createChart, ColorType, type IChartApi, type ISeriesApi } from 'lightweight-charts'
import { useCallback, useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/api/v1/stream'

/** Wire format: t (unix s), p (p_hat), lo, hi, m (market_mid or null) */
type Point = { t: number; p: number; lo: number; hi: number; m: number | null }

function App() {
  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>('connecting')
  const [current, setCurrent] = useState<{ state?: unknown; derived?: { p_hat?: number } } | null>(null)
  const [snapshotHistory, setSnapshotHistory] = useState<Point[]>([])
  const [chartReady, setChartReady] = useState(false)
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<IChartApi | null>(null)
  const pSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const loSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const hiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const historyLoadedRef = useRef(false)

  const appendPoint = useCallback((point: Point) => {
    const time = point.t as import('lightweight-charts').UTCTimestamp
    pSeriesRef.current?.update({ time, value: point.p })
    loSeriesRef.current?.update({ time, value: point.lo })
    hiSeriesRef.current?.update({ time, value: point.hi })
  }, [])

  const loadHistory = useCallback((history: Point[]) => {
    if (!pSeriesRef.current || !loSeriesRef.current || !hiSeriesRef.current || history.length === 0) return
    const pData = history.map((pt) => ({ time: pt.t as import('lightweight-charts').UTCTimestamp, value: pt.p }))
    const loData = history.map((pt) => ({ time: pt.t as import('lightweight-charts').UTCTimestamp, value: pt.lo }))
    const hiData = history.map((pt) => ({ time: pt.t as import('lightweight-charts').UTCTimestamp, value: pt.hi }))
    pSeriesRef.current.setData(pData)
    loSeriesRef.current.setData(loData)
    hiSeriesRef.current.setData(hiData)
    chartInstanceRef.current?.timeScale().fitContent()
    historyLoadedRef.current = true
  }, [])

  // Apply snapshot history when chart is ready or when snapshot arrives later
  useEffect(() => {
    if (snapshotHistory.length > 0 && chartReady && pSeriesRef.current) loadHistory(snapshotHistory)
  }, [snapshotHistory, chartReady, loadHistory])

  // Chart init: one series for p_hat, two for bounds
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
      historyLoadedRef.current = false
      setChartReady(false)
    }
  }, [])

  // WebSocket: connect, handle snapshot and point
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
          const hist = Array.isArray(msg.history) ? (msg.history as Point[]) : []
          setSnapshotHistory(hist)
        } else if (msg.type === 'point' && msg.point) {
          const pt = msg.point as Point
          if (historyLoadedRef.current) appendPoint(pt)
          else setSnapshotHistory((prev) => (prev.length ? prev : [pt]))
        }
      } catch {
        // ignore parse errors
      }
    }
    return () => ws.close()
  }, [loadHistory, appendPoint])

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
      </p>
      {current?.derived?.p_hat != null && (
        <p style={{ fontSize: 14, color: '#9ca3af' }}>
          Current p_hat: <strong>{current.derived.p_hat.toFixed(4)}</strong>
        </p>
      )}
      <div ref={chartRef} style={{ marginTop: 16 }} />
    </div>
  )
}

export default App

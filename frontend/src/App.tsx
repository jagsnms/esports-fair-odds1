import { createChart, ColorType, type IChartApi, type ISeriesApi } from 'lightweight-charts'
import { useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/api/v1/stream'

// Dummy data for placeholder line series
const DUMMY_DATA = [
  { time: '2024-01-01', value: 0.45 },
  { time: '2024-01-02', value: 0.52 },
  { time: '2024-01-03', value: 0.48 },
  { time: '2024-01-04', value: 0.55 },
  { time: '2024-01-05', value: 0.58 },
  { time: '2024-01-06', value: 0.52 },
  { time: '2024-01-07', value: 0.61 },
]

function App() {
  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>('connecting')
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<IChartApi | null>(null)

  // WebSocket connection to backend stream
  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    ws.onopen = () => setWsStatus('open')
    ws.onclose = () => setWsStatus('closed')
    ws.onerror = () => setWsStatus('error')
    return () => ws.close()
  }, [])

  // Lightweight-charts: one line series with dummy data
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
    const series: ISeriesApi<'Line'> = chart.addLineSeries({ color: '#3b82f6' })
    series.setData(DUMMY_DATA)
    chart.timeScale().fitContent()

    const handleResize = () => {
      if (chartRef.current && chartInstanceRef.current)
        chartInstanceRef.current.applyOptions({ width: chartRef.current!.clientWidth })
    }
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartInstanceRef.current = null
    }
  }, [])

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
      <div ref={chartRef} style={{ marginTop: 16 }} />
    </div>
  )
}

export default App

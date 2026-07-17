import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { Loader2, AlertCircle } from 'lucide-react'

export default function ChartRenderer({ chartData, title, height = '450px' }) {
  const [plotData, setPlotData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    if (!chartData) {
      setLoading(true)
      return
    }

    try {
      // If the chartData is already a dict, use it. If it is a string, parse it.
      const parsed = typeof chartData === 'string' ? JSON.parse(chartData) : chartData
      
      // Update background and layout settings to ensure perfect glass dark theme blending
      if (parsed.layout) {
        parsed.layout.paper_bgcolor = 'rgba(0, 0, 0, 0)'
        parsed.layout.plot_bgcolor = 'rgba(0, 0, 0, 0)'
        parsed.layout.autosize = true
        
        // Ensure responsive fonts
        parsed.layout.font = {
          color: '#F8FAFC',
          family: "'Plus Jakarta Sans', sans-serif"
        }
        
        if (parsed.layout.legend) {
          parsed.layout.legend.bgcolor = 'rgba(0,0,0,0)'
        }
      }
      
      setPlotData(parsed)
      setError(false)
    } catch (e) {
      console.error('Error parsing Plotly data:', e)
      setError(true)
    } finally {
      setLoading(false)
    }
  }, [chartData])

  if (loading) {
    return (
      <div style={{ display: 'flex', height, justifyContent: 'center', alignItems: 'center', background: 'rgba(30, 41, 59, 0.15)', borderRadius: '16px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
          <Loader2 className="animate-spin" size={24} color="var(--primary)" />
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Generating chart...</span>
        </div>
      </div>
    )
  }

  if (error || !plotData) {
    return (
      <div style={{ display: 'flex', height, justifyContent: 'center', alignItems: 'center', background: 'rgba(239, 68, 68, 0.05)', borderRadius: '16px', border: '1px solid rgba(239, 68, 68, 0.15)', color: 'var(--danger)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertCircle size={20} />
          <span style={{ fontWeight: 600 }}>Error loading chart</span>
        </div>
      </div>
    )
  }

  return (
    <div style={{ width: '100%', height, overflow: 'hidden' }}>
      <Plot
        data={plotData.data}
        layout={{
          ...plotData.layout,
          margin: { l: 50, r: 20, t: 50, b: 50 },
        }}
        config={{
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  )
}

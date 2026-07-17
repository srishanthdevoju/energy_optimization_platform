import { useState, useEffect } from 'react'
import ChartRenderer from '../components/ChartRenderer'
import { AlertOctagon, Filter, Calendar, Zap, RefreshCw, Loader2 } from 'lucide-react'
import { apiFetch } from '../utils/api'

export default function Anomalies({ filters }) {
  const [selectedHh, setSelectedHh] = useState('All Households')
  const [sensitivity, setSensitivity] = useState(5)

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    const params = new URLSearchParams({
      selected_hh: selectedHh,
      sensitivity: sensitivity.toString()
    })

    apiFetch(`/api/anomalies?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to run anomaly detector.')
        return res.json()
      })
      .then((resData) => {
        setData(resData)
        setLoading(false)
      })
      .catch((err) => {
        console.error(err)
        setError(err.message)
        setLoading(false)
      })
  }, [selectedHh, sensitivity])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h2 style={{ fontSize: '2rem', color: '#fff', marginBottom: '0.25rem' }}>🚨 Anomaly Detection</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Identify unusual consumption anomalies (appliance faults, leaks, or surges) using Isolation Forest.</p>
      </div>

      {/* Selectors panel */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Filter size={18} color="var(--primary)" /> Detection Parameters
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', alignItems: 'center' }}>
          <div>
            <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Select Scope</label>
            <select
              value={selectedHh}
              onChange={(e) => setSelectedHh(e.target.value)}
              style={{ width: '100%', background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '8px', color: '#fff', padding: '0.65rem 0.8rem', fontSize: '0.9rem' }}
            >
              <option value="All Households">All Households (System)</option>
              {filters.households.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
          </div>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
              <span>Sensitivity (Contamination)</span>
              <span style={{ color: 'var(--danger)' }}>{sensitivity}%</span>
            </div>
            <input
              type="range"
              min="1"
              max="15"
              value={sensitivity}
              onChange={(e) => setSensitivity(parseInt(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--danger)', cursor: 'pointer' }}
            />
          </div>
        </div>
      </div>

      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '4rem 0' }}>
          <Loader2 className="animate-spin" size={32} color="var(--primary)" />
          <span style={{ marginLeft: '0.75rem', color: 'var(--text-secondary)' }}>Scanning energy history...</span>
        </div>
      )}

      {error && (
        <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', borderColor: 'var(--danger)', background: 'rgba(239, 68, 68, 0.02)' }}>
          <p style={{ color: 'var(--danger)', fontWeight: 600, fontSize: '1.1rem', marginBottom: '0.5rem' }}>⚠️ Anomaly Scan Failure</p>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{error}</p>
        </div>
      )}

      {!loading && !error && data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
          {/* KPI summaries */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1.25rem' }}>
            {[
              { label: 'Total Records', value: data.metrics.total_records.toLocaleString(), color: 'var(--info)' },
              { label: 'Anomalies Found', value: data.metrics.anomalies_found.toLocaleString(), color: 'var(--danger)' },
              { label: 'Anomaly Rate', value: `${data.metrics.anomaly_rate}%`, color: 'var(--accent)' },
              { label: 'Avg Anomaly Load', value: `${data.metrics.avg_anomaly_energy.toFixed(2)} kWh`, color: 'var(--secondary)' },
            ].map((kpi, i) => (
              <div key={i} className="glass-card" style={{
                padding: '1.25rem',
                textAlign: 'center',
                border: `1px solid rgba(255, 255, 255, 0.05)`,
                borderBottom: `3px solid ${kpi.color}`
              }}>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.25rem', textTransform: 'uppercase' }}>{kpi.label}</p>
                <p style={{ color: kpi.color, fontSize: '1.75rem', fontWeight: 700 }}>{kpi.value}</p>
              </div>
            ))}
          </div>

          {/* Timeline plot */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertOctagon size={18} color="var(--danger)" /> Anomaly Outliers Timeline
            </h4>
            <ChartRenderer chartData={data.charts.timeline} height="400px" />
          </div>

          {/* Score distribution & contrast card */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem' }}>📉 Anomaly Score Distribution</h4>
              <ChartRenderer chartData={data.charts.distribution} height="300px" />
            </div>

            <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff' }}>⚡ Normal vs. Anomalous Consumption</h4>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', flex: 1, alignItems: 'center' }}>
                <div style={{ padding: '1.5rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--card-border)', borderRadius: '12px', textAlign: 'center' }}>
                  <span style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.35rem' }}>Normal Avg Load</span>
                  <span style={{ color: '#fff', fontSize: '1.8rem', fontWeight: 700 }}>{data.metrics.avg_normal_energy.toFixed(2)} kWh</span>
                </div>
                
                <div style={{ padding: '1.5rem', background: 'rgba(239, 68, 68, 0.03)', border: '1px solid rgba(239, 68, 68, 0.15)', borderRadius: '12px', textAlign: 'center' }}>
                  <span style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.35rem' }}>Anomaly Avg Load</span>
                  <span style={{ color: 'var(--danger)', fontSize: '1.8rem', fontWeight: 700 }}>
                    {data.metrics.avg_anomaly_energy.toFixed(2)} kWh
                  </span>
                  {data.metrics.anomalies_found > 0 && (
                    <span style={{ display: 'block', color: 'var(--danger)', fontSize: '0.75rem', fontWeight: 600, marginTop: '0.25rem' }}>
                      {((data.metrics.avg_anomaly_energy - data.metrics.avg_normal_energy) / data.metrics.avg_normal_energy * 100).toFixed(1)}% shift
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Anomaly Details Table */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem' }}>📋 Anomaly Logs (Top 100 Outliers)</h4>
            
            <div style={{ overflowX: 'auto', maxHeight: '400px' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left', position: 'sticky', top: 0, background: '#121626', zIndex: 1 }}>
                    <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Date</th>
                    <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Household ID</th>
                    <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Anomaly Score</th>
                    <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Daily Consumption</th>
                    <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Mean Load</th>
                    <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Max Load</th>
                  </tr>
                </thead>
                <tbody>
                  {data.details && data.details.length > 0 ? (
                    data.details.map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>{row.day}</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{row.LCLid}</td>
                        <td style={{ padding: '0.75rem 0.5rem', color: 'var(--danger)', fontWeight: 600 }}>{row.anomaly_score.toFixed(4)}</td>
                        <td style={{ padding: '0.75rem 0.5rem', color: '#fff' }}>{row.energy_sum.toFixed(2)} kWh</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{row.energy_mean.toFixed(2)} kWh</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{row.energy_max.toFixed(2)} kWh</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="6" style={{ textAlign: 'center', padding: '1.5rem', color: 'var(--text-muted)' }}>No anomalies detected with current sensitivity setting.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

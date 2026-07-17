import { useState, useEffect } from 'react'
import ChartRenderer from '../components/ChartRenderer'
import { Users, Search, BarChart3, TrendingUp, HelpCircle, Loader2 } from 'lucide-react'
import { apiFetch } from '../utils/api'


export default function Clusters({ filters }) {
  const [xAxis, setXAxis] = useState('mean_energy')
  const [yAxis, setYAxis] = useState('std_energy')
  const [searchHh, setSearchHh] = useState('')
  const [lookupResult, setLookupResult] = useState(null)
  const [lookupLoading, setLookupLoading] = useState(false)
  const [lookupError, setLookupError] = useState(null)

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    const params = new URLSearchParams({
      x_axis: xAxis,
      y_axis: yAxis,
    })

    apiFetch(`/api/clustering?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load clustering details.')
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
  }, [xAxis, yAxis])

  const handleLookup = (e) => {
    e.preventDefault()
    if (!searchHh) return
    setLookupLoading(true)
    setLookupError(null)

    apiFetch(`/api/clustering?search_household=${searchHh}`)
      .then((res) => res.json())
      .then((resData) => {
        if (resData.lookup) {
          setLookupResult(resData.lookup)
        } else {
          setLookupError('Household ID not found in current clustering index.')
          setLookupResult(null)
        }
        setLookupLoading(false)
      })
      .catch((err) => {
        console.error(err)
        setLookupError('Failed to fetch lookup details.')
        setLookupLoading(false)
      })
  }

  // Set default axes if columns are loaded
  useEffect(() => {
    if (data && data.profile_columns) {
      if (!data.profile_columns.includes(xAxis) && data.profile_columns.length > 0) {
        setXAxis(data.profile_columns[0])
      }
      if (!data.profile_columns.includes(yAxis) && data.profile_columns.length > 1) {
        setYAxis(data.profile_columns[1])
      }
    }
  }, [data])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h2 style={{ fontSize: '2rem', color: '#fff', marginBottom: '0.25rem' }}>🎯 Household Consumption Patterns</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Identify and profile distinct consumer behaviors using unsupervised K-Means clustering.</p>
      </div>

      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '4rem 0' }}>
          <Loader2 className="animate-spin" size={32} color="var(--primary)" />
          <span style={{ marginLeft: '0.75rem', color: 'var(--text-secondary)' }}>Loading consumer segments...</span>
        </div>
      )}

      {error && (
        <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', borderColor: 'var(--danger)', background: 'rgba(239, 68, 68, 0.02)' }}>
          <p style={{ color: 'var(--danger)', fontWeight: 600, fontSize: '1.1rem', marginBottom: '0.5rem' }}>⚠️ Clustering Error</p>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{error}</p>
        </div>
      )}

      {!loading && !error && data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
          {/* Cluster Summary Cards */}
          <div>
            <h3 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Users size={20} color="var(--primary)" /> Cluster Segments Overview
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem' }}>
              {data.summary.map((cluster, i) => {
                const colors = ['var(--primary)', 'var(--secondary)', 'var(--accent)', 'var(--info)']
                const color = colors[i % colors.length]
                return (
                  <div key={i} className="glass-card" style={{
                    padding: '1.25rem',
                    border: '1px solid var(--card-border)',
                    borderTop: `4px solid ${color}`,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem'
                  }}>
                    <h4 style={{ color, fontSize: '1.1rem', fontWeight: 700 }}>{cluster.cluster_label}</h4>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 500 }}>{cluster.count} households</p>
                    <p style={{ color: '#fff', fontSize: '1.5rem', fontWeight: 700, marginTop: '0.5rem' }}>
                      {cluster.mean_energy.toFixed(2)}
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}> kWh/day</span>
                    </p>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Scatter Plot with Selectors */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart3 size={18} color="var(--primary)" /> Cluster Feature Mapping
              </h4>
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <div>
                  <label style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', fontWeight: 600, marginRight: '0.5rem' }}>X-Axis</label>
                  <select
                    value={xAxis}
                    onChange={(e) => setXAxis(e.target.value)}
                    style={{ background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '6px', color: '#fff', padding: '0.35rem 0.6rem', fontSize: '0.85rem' }}
                  >
                    {data.profile_columns.map(col => <option key={col} value={col}>{col.replace('_', ' ')}</option>)}
                  </select>
                </div>
                <div>
                  <label style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', fontWeight: 600, marginRight: '0.5rem' }}>Y-Axis</label>
                  <select
                    value={yAxis}
                    onChange={(e) => setYAxis(e.target.value)}
                    style={{ background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '6px', color: '#fff', padding: '0.35rem 0.6rem', fontSize: '0.85rem' }}
                  >
                    {data.profile_columns.map(col => <option key={col} value={col}>{col.replace('_', ' ')}</option>)}
                  </select>
                </div>
              </div>
            </div>
            <ChartRenderer chartData={data.charts.scatter} height="400px" />
          </div>

          {/* Profiles and Stats Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem' }}>📡 Cluster Normalized Profiles</h4>
              <ChartRenderer chartData={data.charts.radar} height="350px" />
            </div>

            <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem' }}>📋 Profile Feature Means</h4>
              <div style={{ flex: 1, overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left' }}>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Cluster</th>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Mean Energy</th>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Max Energy</th>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Min Energy</th>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Weekend Ratio</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.summary.map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: '#fff' }}>{row.cluster_label}</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{row.mean_energy.toFixed(2)} kWh</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{row.max_energy.toFixed(2)} kWh</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{row.min_energy.toFixed(2)} kWh</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{(row.weekend_ratio * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Elbow Optimization and Lookup */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem' }}>📈 Optimal K Clusters Elbow Analysis</h4>
              <ChartRenderer chartData={data.charts.elbow} height="300px" />
            </div>

            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Search size={18} color="var(--primary)" /> Household Lookup Tool
              </h4>
              <form onSubmit={handleLookup} style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
                <select
                  value={searchHh}
                  onChange={(e) => setSearchHh(e.target.value)}
                  style={{ flex: 1, background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '6px', color: '#fff', padding: '0.5rem', fontSize: '0.85rem' }}
                >
                  <option value="">Search household (LCLid)...</option>
                  {filters.households.map(id => <option key={id} value={id}>{id}</option>)}
                </select>
                <button type="submit" className="btn-primary" style={{ padding: '0.5rem 1.25rem', fontSize: '0.85rem' }}>Search</button>
              </form>

              {lookupLoading && (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '1rem' }}>
                  <Loader2 className="animate-spin" size={24} color="var(--primary)" />
                </div>
              )}

              {lookupError && (
                <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.15)', borderRadius: '8px', color: 'var(--danger)', fontSize: '0.9rem' }}>
                  {lookupError}
                </div>
              )}

              {lookupResult && !lookupLoading && (
                <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--card-border)', borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>{lookupResult.LCLid}</span>
                    <span className="badge badge-primary" style={{ fontSize: '0.85rem' }}>{lookupResult.cluster_label}</span>
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', fontSize: '0.9rem' }}>
                    {[
                      { name: 'Daily Average', val: `${lookupResult.stats.mean_energy.toFixed(2)} kWh` },
                      { name: 'Peak Output', val: `${lookupResult.stats.max_energy.toFixed(2)} kWh` },
                      { name: 'Minimum Output', val: `${lookupResult.stats.min_energy.toFixed(2)} kWh` },
                      { name: 'Weekend Shift', val: `${(lookupResult.stats.weekend_ratio * 100).toFixed(1)}%` },
                    ].map((stat, i) => (
                      <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 600 }}>{stat.name}</span>
                        <span style={{ color: '#fff', fontWeight: 700 }}>{stat.val}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

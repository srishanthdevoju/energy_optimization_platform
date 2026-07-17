import { useState, useEffect } from 'react'
import ChartRenderer from '../components/ChartRenderer'
import { Brain, Calendar, Home, CheckCircle2, TrendingUp, Loader2 } from 'lucide-react'
import { apiFetch } from '../utils/api'

export default function Forecast({ filters }) {
  const [modelType, setModelType] = useState('xgboost')
  const [selectedHh, setSelectedHh] = useState('All')
  const [testDays, setTestDays] = useState(60)

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    const params = new URLSearchParams({
      model_type: modelType,
      selected_hh: selectedHh,
      test_days: testDays.toString()
    })

    apiFetch(`/api/forecasting?${params.toString()}`)
      .then((res) => {
        if (!res.ok) {
          return res.json().then(e => { throw new Error(e.detail || 'Forecasting failed') })
        }
        return res.json()
      })
      .then((resData) => {
        setData(resData)
        setError(null)
        setLoading(false)
      })
      .catch((err) => {
        console.error(err)
        setError(err.message)
        setLoading(false)
      })
  }, [modelType, selectedHh, testDays])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h2 style={{ fontSize: '2rem', color: '#fff', marginBottom: '0.25rem' }}>🔮 Energy Consumption Forecasting</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Predict prospective power grid demand using optimized ensemble models trained on historical data.</p>
      </div>

      {/* Query selectors */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Brain size={18} color="var(--primary)" /> Forecaster Directives
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', alignItems: 'center' }}>
          <div>
            <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Model Type</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              style={{ width: '100%', background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '8px', color: '#fff', padding: '0.65rem 0.8rem', fontSize: '0.9rem' }}
            >
              {data && data.available_models ? (
                data.available_models.map(m => <option key={m} value={m}>{m.toUpperCase()}</option>)
              ) : (
                <>
                  <option value="xgboost">XGBoost</option>
                  <option value="lightgbm">LightGBM</option>
                  <option value="random_forest">Random Forest</option>
                </>
              )}
            </select>
          </div>
          <div>
            <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Household Context</label>
            <select
              value={selectedHh}
              onChange={(e) => setSelectedHh(e.target.value)}
              style={{ width: '100%', background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '8px', color: '#fff', padding: '0.65rem 0.8rem', fontSize: '0.9rem' }}
            >
              <option value="All">All (Aggregated)</option>
              {filters.households.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
          </div>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
              <span>Test Period</span>
              <span style={{ color: 'var(--primary)' }}>{testDays} Days</span>
            </div>
            <input
              type="range"
              min="14"
              max="90"
              value={testDays}
              onChange={(e) => setTestDays(parseInt(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--primary)', cursor: 'pointer' }}
            />
          </div>
        </div>
      </div>

      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '4rem 0' }}>
          <Loader2 className="animate-spin" size={32} color="var(--primary)" />
          <span style={{ marginLeft: '0.75rem', color: 'var(--text-secondary)' }}>Invoking predictive model...</span>
        </div>
      )}

      {error && (
        <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', borderColor: 'var(--danger)', background: 'rgba(239, 68, 68, 0.02)' }}>
          <p style={{ color: 'var(--danger)', fontWeight: 600, fontSize: '1.1rem', marginBottom: '0.5rem' }}>⚠️ Forecasting Error</p>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{error}</p>
        </div>
      )}

      {!loading && !error && data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
          {/* Performance KPIs */}
          <div>
            <h3 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '1rem' }}>📈 Forecast Performance</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1.25rem' }}>
              {[
                { label: 'MAE', value: data.metrics.MAE.toFixed(4), unit: ' kWh', color: 'var(--primary)' },
                { label: 'RMSE', value: data.metrics.RMSE.toFixed(4), unit: ' kWh', color: 'var(--secondary)' },
                { label: 'MAPE', value: `${data.metrics.MAPE.toFixed(2)}%`, unit: '', color: 'var(--accent)' },
                { label: 'R² Score', value: data.metrics.R2.toFixed(4), unit: '', color: 'var(--success)' },
              ].map((kpi, i) => (
                <div key={i} className="glass-card" style={{
                  padding: '1.25rem',
                  textAlign: 'center',
                  border: `1px solid rgba(255, 255, 255, 0.05)`,
                  borderBottom: `3px solid ${kpi.color}`
                }}>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.25rem', textTransform: 'uppercase' }}>{kpi.label}</p>
                  <p style={{ color: kpi.color, fontSize: '1.8rem', fontWeight: 700 }}>
                    {kpi.value}
                    <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500 }}>{kpi.unit}</span>
                  </p>
                </div>
              ))}
            </div>
            {data.metrics.MAPE > 25 && (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.825rem', marginTop: '0.75rem', lineHeight: 1.5 }}>
                ℹ️ <strong>Note on MAPE:</strong> MAPE is highly skewed by low-consumption days (&lt; 1.0 kWh), where minute absolute deviations generate extreme percentage errors. The <strong>Trimmed MAPE</strong> (filtering actuals &lt; 1.0 kWh) evaluates to a more representative <strong>{data.metrics.trimmed_mape.toFixed(2)}%</strong>.
              </p>
            )}
          </div>

          {/* Core forecast plots */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <TrendingUp size={18} color="var(--primary)" /> Actual vs. Predicted Demand
            </h4>
            <ChartRenderer chartData={data.charts.forecast} height="400px" />
          </div>

          {/* Feature Importance & Model Comparison */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem' }}>🏆 Feature Importance Breakdown</h4>
              {data.charts.importance ? (
                <ChartRenderer chartData={data.charts.importance} height="350px" />
              ) : (
                <div style={{ height: '350px', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)' }}>
                  Feature importance data is unavailable for this model.
                </div>
              )}
            </div>

            <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem' }}>🏆 Leaderboard (Baseline Comparison)</h4>
              <div style={{ flex: 1, overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left' }}>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>Model</th>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>MAE</th>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>RMSE</th>
                      <th style={{ padding: '0.75rem 0.5rem', color: '#fff', fontWeight: 600 }}>R² Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.model_comparison && data.model_comparison.length > 0 ? (
                      data.model_comparison.map((row, idx) => (
                        <tr key={idx} style={{ 
                          borderBottom: '1px solid rgba(255,255,255,0.04)',
                          background: row.Model.toLowerCase() === modelType.toLowerCase() ? 'rgba(0, 212, 170, 0.03)' : 'transparent'
                        }}>
                          <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: row.Model.toLowerCase() === modelType.toLowerCase() ? 'var(--primary)' : '#fff' }}>
                            {row.Model} {row.Model.toLowerCase() === modelType.toLowerCase() && '👈'}
                          </td>
                          <td style={{ padding: '0.75rem 0.5rem' }}>{row.MAE.toFixed(4)}</td>
                          <td style={{ padding: '0.75rem 0.5rem' }}>{row.RMSE.toFixed(4)}</td>
                          <td style={{ padding: '0.75rem 0.5rem' }}>{row.R2.toFixed(4)}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="4" style={{ textAlign: 'center', padding: '1rem' }}>No comparison models loaded.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

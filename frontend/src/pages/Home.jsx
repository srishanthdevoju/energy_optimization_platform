import { useEffect, useState } from 'react'
import { Zap, Users, Calendar, BarChart2, Shield } from 'lucide-react'
import { apiFetch } from '../utils/api'

export default function Home() {
  const [kpis, setKpis] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    apiFetch('/api/kpis')
      .then((res) => res.json())
      .then((data) => {
        setKpis(data)
        setLoading(false)
      })
      .catch((err) => {
        console.error(err)
        setLoading(false)
      })
  }, [])


  const kpiItems = kpis
    ? [
        { label: 'Households', value: kpis.n_households.toLocaleString(), icon: Users, gradient: 'linear-gradient(135deg, #0f766e, #14b8a6)' },
        { label: 'Data Period', value: kpis.date_range, icon: Calendar, gradient: 'linear-gradient(135deg, #7c3aed, #a78bfa)' },
        { label: 'Avg Daily Energy', value: `${kpis.avg_daily_energy.toFixed(2)} kWh`, icon: Zap, gradient: 'linear-gradient(135deg, #d97706, #fbbf24)' },
        { label: 'Total Records', value: kpis.total_records.toLocaleString(), icon: BarChart2, gradient: 'linear-gradient(135deg, #2563eb, #60a5fa)' },
      ]
    : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      {/* Hero Header */}
      <div style={{ textAlign: 'center', padding: '1.5rem 0' }}>
        <h1 style={{
          background: 'linear-gradient(135deg, var(--primary), var(--secondary))',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          fontSize: '3rem',
          fontWeight: 800,
          marginBottom: '0.5rem'
        }}>⚡ AI-Powered Energy Analytics</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.15rem', fontWeight: 500 }}>
          Smart Meter Demand Forecasting · Pattern Clustering · Anomaly Detection
        </p>
      </div>

      <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)' }}></div>

      {/* KPI Cards Grid */}
      <div className="kpi-grid">
        {loading
          ? Array(4).fill(0).map((_, i) => (
              <div key={i} className="glass-card" style={{ height: '130px', animationDelay: `${i * 0.1}s` }}></div>
            ))
          : kpiItems.map((item, idx) => {
              const Icon = item.icon
              return (
                <div key={idx} className="glass-card" style={{
                  padding: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1.25rem',
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    background: item.gradient,
                    padding: '0.85rem',
                    borderRadius: '12px',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <Icon size={24} />
                  </div>
                  <div>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, textTransform: 'uppercase', trackingLetter: '0.05em' }}>{item.label}</p>
                    <p style={{ color: '#fff', fontSize: '1.65rem', fontWeight: 700, margin: '0.15rem 0 0 0' }}>{item.value}</p>
                  </div>
                  {/* Decorative background glow */}
                  <div style={{
                    position: 'absolute',
                    top: '-20px',
                    right: '-20px',
                    width: '80px',
                    height: '80px',
                    background: 'rgba(255,255,255,0.02)',
                    borderRadius: '50%',
                    pointerEvents: 'none'
                  }}></div>
                </div>
              )
            })
        }
      </div>

      {/* Split Content: Overview vs. Models */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '2rem' }}>
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '1.35rem', color: '#fff', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Zap size={20} color="var(--primary)" /> Project Overview
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.975rem', lineHeight: 1.7, marginBottom: '1rem' }}>
            This system analyzes <strong>London Smart Meter</strong> electricity consumption datasets to forecast demand,
            uncover residential usage patterns, and detect anomalous behavior.
          </p>
          <ul style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.8, paddingLeft: '1.2rem' }}>
            <li><strong>Demand Forecasting:</strong> Multi-model ensembles predicting upcoming usage.</li>
            <li><strong>Clustering Analysis:</strong> Identifies segments (e.g. high/low energy savers).</li>
            <li><strong>Anomaly Detection:</strong> Pinpoints abnormal appliance loads or power surges.</li>
            <li><strong>Interactive Explorer:</strong> Filter energy logs by demographics or tariff type.</li>
          </ul>
        </div>

        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '1.35rem', color: '#fff', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Shield size={20} color="var(--secondary)" /> AI & Machine Learning Stack
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {[
              { name: 'XGBoost', desc: 'Extreme gradient-boosted trees for high-precision time series forecasting.' },
              { name: 'LightGBM', desc: 'Leaf-wise gradient boosting optimized for rapid training and scalability.' },
              { name: 'Random Forest', desc: 'Sturdy bootstrap aggregation baseline forecaster.' },
              { name: 'K-Means Clustering', desc: 'Unsupervised grouping mapping typical consumption profiles.' },
              { name: 'Isolation Forest', desc: 'Tree-based anomaly scoring targeting device failures or spikes.' }
            ].map((model, idx) => (
              <div key={idx} style={{ display: 'flex', gap: '0.75rem', fontSize: '0.925rem' }}>
                <span style={{ color: 'var(--primary)', fontWeight: 700, minWidth: '120px' }}>{model.name}</span>
                <span style={{ color: 'var(--text-secondary)' }}>{model.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* System Architecture */}
      <div className="glass-panel" style={{ padding: '2rem' }}>
        <h3 style={{ fontSize: '1.35rem', color: '#fff', marginBottom: '1.5rem' }}>🏗️ System Architecture Flowchart</h3>
        <div style={{ 
          background: 'rgba(15, 23, 42, 0.3)', 
          border: '1px solid rgba(255,255,255,0.04)', 
          borderRadius: '12px', 
          padding: '1.5rem',
          overflowX: 'auto',
          fontFamily: 'monospace',
          color: 'var(--text-secondary)',
          lineHeight: 1.5,
          fontSize: '0.85rem',
          whiteSpace: 'pre'
        }}>
{` ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
 │   Raw CSVs   │────▶│  Preprocessing   │────▶│  Feature Eng.   │
 │  (Datasets)  │     │  Clean & Merge   │     │  Temporal, Lag  │
 └──────────────┘     └──────────────────┘     │  Rolling, Weather│
                                               └────────┬────────┘
                                                        │
                      ┌─────────────────────────────────┼────────────────────────┐
                      │                                 │                        │
            ┌─────────▼─────────┐             ┌─────────▼─────────┐    ┌─────────▼────────┐
            │   Forecasting     │             │     Clustering    │    │Anomaly Detection │
            │  XGB / LGBM / RF  │             │     K-Means       │    │ Isolation Forest │
            └─────────┬─────────┘             └─────────┬─────────┘    └─────────┬────────┘
                      │                                 │                        │
                      └─────────────────────────────────┼────────────────────────┘
                                                        │
                                              ┌─────────▼─────────┐
                                              │   FastAPI Server  │
                                              │  (REST API / RAG) │
                                              └─────────┬─────────┘
                                                        │
                                              ┌─────────▼─────────┐
                                              │  React Dashboard  │
                                              │  (Interactive UI) │
                                              └───────────────────┘`}
        </div>
      </div>
    </div>
  )
}

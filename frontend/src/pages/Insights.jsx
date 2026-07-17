import { useState, useEffect } from 'react'
import ChartRenderer from '../components/ChartRenderer'
import { Calendar, DollarSign, Award, AlertCircle, ArrowUpRight, TrendingUp, Loader2 } from 'lucide-react'
import { apiFetch } from '../utils/api'

export default function Insights() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    apiFetch('/api/insights')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch AI insights')
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
  }, [])


  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '4rem 0' }}>
        <Loader2 className="animate-spin" size={32} color="var(--primary)" />
        <span style={{ marginLeft: '0.75rem', color: 'var(--text-secondary)' }}>Synthesizing AI insights...</span>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', borderColor: 'var(--danger)', background: 'rgba(239, 68, 68, 0.02)' }}>
        <p style={{ color: 'var(--danger)', fontWeight: 600, fontSize: '1.1rem', marginBottom: '0.5rem' }}>⚠️ Insights Generation Failed</p>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{error || 'No data returned.'}</p>
      </div>
    )
  }

  const { peak_insights, forecast_insights, cluster_insights, recommendations, charts } = data

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h2 style={{ fontSize: '2rem', color: '#fff', marginBottom: '0.25rem' }}>💡 AI-Generated Energy Insights</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Dynamic recommendations and savings projections powered by machine learning model outputs.</p>
      </div>

      <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)' }}></div>

      {/* Peak Analysis Widgets */}
      <div>
        <h3 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Calendar size={20} color="var(--primary)" /> Peak Consumption Analysis
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.5rem' }}>
          <div className="glass-card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--danger)' }}>
            <span style={{ color: 'var(--danger)', fontWeight: 700, fontSize: '0.9rem', display: 'block', marginBottom: '0.5rem' }}>🔥 Peak Usage Day</span>
            <span style={{ color: '#fff', fontSize: '1.8rem', fontWeight: 800, display: 'block' }}>{peak_insights.peak_day}</span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Highest average daily load</span>
          </div>

          <div className="glass-card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent)' }}>
            <span style={{ color: 'var(--accent)', fontWeight: 700, fontSize: '0.9rem', display: 'block', marginBottom: '0.5rem' }}>❄️ Peak Season</span>
            <span style={{ color: '#fff', fontSize: '1.8rem', fontWeight: 800, display: 'block' }}>{peak_insights.peak_season}</span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Lowest: {peak_insights.lowest_season}</span>
          </div>

          <div className="glass-card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--info)' }}>
            <span style={{ color: 'var(--info)', fontWeight: 700, fontSize: '0.9rem', display: 'block', marginBottom: '0.5rem' }}>📅 Weekend Effect</span>
            <span style={{ color: '#fff', fontSize: '1.8rem', fontWeight: 800, display: 'block' }}>{peak_insights.weekend_pct_diff > 0 ? `+${peak_insights.weekend_pct_diff.toFixed(1)}%` : `${peak_insights.weekend_pct_diff.toFixed(1)}%`}</span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>vs. weekday consumption</span>
          </div>
        </div>

        <div style={{ marginTop: '1.5rem' }} className="glass-panel" style={{ padding: '1.5rem' }}>
          <ChartRenderer chartData={charts.day_of_week} height="350px" />
        </div>
      </div>

      {/* Projections and Savings */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '2rem' }}>
        {/* Forecast cost projection */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.25rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <TrendingUp size={20} color="var(--primary)" /> Estimated Monthly Projections
          </h3>
          
          {forecast_insights.avg_daily_predicted ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', flex: 1, justifyContent: 'center' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', textAlign: 'center' }}>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '1rem 0.5rem', borderRadius: '10px', border: '1px solid var(--card-border)' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 600, display: 'block' }}>DAILY AVG</span>
                  <span style={{ color: '#fff', fontSize: '1.3rem', fontWeight: 700 }}>{forecast_insights.avg_daily_predicted.toFixed(2)} kWh</span>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '1rem 0.5rem', borderRadius: '10px', border: '1px solid var(--card-border)' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 600, display: 'block' }}>MONTHLY EST</span>
                  <span style={{ color: '#fff', fontSize: '1.3rem', fontWeight: 700 }}>{forecast_insights.estimated_monthly_kwh.toFixed(1)} kWh</span>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '1rem 0.5rem', borderRadius: '10px', border: '1px solid var(--card-border)', borderBottom: '2px solid var(--primary)' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 600, display: 'block' }}>MONTHLY COST</span>
                  <span style={{ color: 'var(--primary)', fontSize: '1.3rem', fontWeight: 700 }}>£{forecast_insights.estimated_monthly_cost_gbp.toFixed(2)}</span>
                </div>
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.5, background: 'rgba(0, 212, 170, 0.03)', border: '1px dashed rgba(0, 212, 170, 0.15)', padding: '0.75rem 1rem', borderRadius: '8px' }}>
                ℹ️ Projections generated using the <strong>{forecast_insights.model_type.toUpperCase()}</strong> model based on the last 30 active daily load trends (assumes UK avg cost rate of £0.34/kWh).
              </p>
              {charts.projection && <ChartRenderer chartData={charts.projection} height="220px" />}
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textAlign: 'center', padding: '2rem' }}>Forecaster insights unavailable. Re-run training scripts.</p>
          )}
        </div>

        {/* Cluster-based savings */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.25rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <DollarSign size={20} color="var(--success)" /> Energy Saving Potentials
          </h3>

          {cluster_insights.pct_saving ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', flex: 1, justifyContent: 'center' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                <div style={{ background: 'rgba(16, 185, 129, 0.04)', padding: '1.25rem', borderRadius: '12px', border: '1px solid rgba(16, 185, 129, 0.15)' }}>
                  <span style={{ color: 'var(--success)', fontWeight: 700, fontSize: '0.85rem', display: 'block', marginBottom: '0.35rem' }}>💡 Most Efficient Segment</span>
                  <span style={{ color: '#fff', fontSize: '1.35rem', fontWeight: 800, display: 'block' }}>{cluster_insights.low_cluster_label}</span>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Avg: {cluster_insights.low_cluster_avg.toFixed(2)} kWh/day</span>
                </div>
                
                <div style={{ background: 'rgba(239, 68, 68, 0.04)', padding: '1.25rem', borderRadius: '12px', border: '1px solid rgba(239, 68, 68, 0.15)' }}>
                  <span style={{ color: 'var(--danger)', fontWeight: 700, fontSize: '0.85rem', display: 'block', marginBottom: '0.35rem' }}>🔥 Highest Consumption Segment</span>
                  <span style={{ color: '#fff', fontSize: '1.35rem', fontWeight: 800, display: 'block' }}>{cluster_insights.high_cluster_label}</span>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Avg: {cluster_insights.high_cluster_avg.toFixed(2)} kWh/day</span>
                </div>
              </div>

              <div style={{ background: 'rgba(16, 185, 129, 0.03)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <span style={{ color: 'var(--success)', fontSize: '1.1rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Award size={18} /> Optimization Target
                </span>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>
                  If high-consumption households adopted the consumption patterns of the most efficient peer group, they could scale down daily load by <strong style={{ color: '#fff' }}>{cluster_insights.potential_saving_kwh.toFixed(2)} kWh/day ({cluster_insights.pct_saving.toFixed(1)}%)</strong>.
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.75rem', marginTop: '0.25rem', fontSize: '0.85rem' }}>
                  <span>Annual Savings: <strong style={{ color: '#fff' }}>{cluster_insights.annual_saving_kwh.toLocaleString()} kWh</strong></span>
                  <span>Financial Yield: <strong style={{ color: 'var(--success)' }}>£{cluster_insights.annual_saving_gbp.toLocaleString()}/yr</strong></span>
                </div>
              </div>
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textAlign: 'center', padding: '2rem' }}>Clustering metrics unavailable. Re-run training scripts.</p>
          )}
        </div>
      </div>

      {/* AI Recommendations List */}
      <div className="glass-panel" style={{ padding: '2rem' }}>
        <h3 style={{ fontSize: '1.25rem', color: '#fff', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          💡 Actionable Savings Recommendations
        </h3>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {recommendations.map((rec, i) => (
            <div key={i} style={{
              background: 'rgba(255,255,255,0.02)',
              borderLeft: '4px solid var(--primary)',
              borderRadius: '0 12px 12px 0',
              padding: '1.25rem 1.5rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.35rem'
            }}>
              <span style={{ color: '#fff', fontWeight: 700, fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '1.2rem' }}>{rec.icon}</span> {rec.title}
              </span>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem', lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: rec.detail }}></p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

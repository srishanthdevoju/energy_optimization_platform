import { useState, useEffect } from 'react'
import ChartRenderer from '../components/ChartRenderer'
import { Filter, BarChart, Sun, HelpCircle, Layers, Loader2 } from 'lucide-react'
import { apiFetch } from '../utils/api'

export default function Eda({ filters }) {
  const [acorn, setAcorn] = useState('All')
  const [tariff, setTariff] = useState('All')
  const [startDate, setStartDate] = useState('2011-11-23')
  const [endDate, setEndDate] = useState('2014-02-28')
  const [weatherVar, setWeatherVar] = useState('temperatureMin')
  const [compareHouseholds, setCompareHouseholds] = useState([])
  const [householdInput, setHouseholdInput] = useState('')
  
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    const compareStr = compareHouseholds.join(',')
    const params = new URLSearchParams({
      start_date: startDate,
      end_date: endDate,
      acorn,
      tariff,
      weather_var: weatherVar,
      ...(compareStr && { compare_households: compareStr })
    })

    apiFetch(`/api/eda?${params.toString()}`)
      .then((res) => res.json())
      .then((resData) => {
        setData(resData)
        setLoading(false)
      })
      .catch((err) => {
        console.error(err)
        setLoading(false)
      })
  }, [acorn, tariff, startDate, endDate, weatherVar, compareHouseholds])

  const handleAddHousehold = (e) => {
    e.preventDefault()
    if (householdInput && !compareHouseholds.includes(householdInput) && compareHouseholds.length < 5) {
      setCompareHouseholds([...compareHouseholds, householdInput])
      setHouseholdInput('')
    }
  }

  const handleRemoveHousehold = (id) => {
    setCompareHouseholds(compareHouseholds.filter((item) => item !== id))
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h2 style={{ fontSize: '2rem', color: '#fff', marginBottom: '0.25rem' }}>🔍 Exploratory Data Analysis</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Analyze historical electricity consumption patterns across households, demographics, and weather conditions.</p>
      </div>

      {/* Filter panel */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Filter size={18} color="var(--primary)" /> Interactive Query Filters
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem' }}>
          <div>
            <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Start Date</label>
            <input 
              type="date" 
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{ width: '100%', background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '8px', color: '#fff', padding: '0.6rem 0.8rem', fontSize: '0.9rem' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>End Date</label>
            <input 
              type="date" 
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{ width: '100%', background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '8px', color: '#fff', padding: '0.6rem 0.8rem', fontSize: '0.9rem' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>ACORN Demographic</label>
            <select 
              value={acorn}
              onChange={(e) => setAcorn(e.target.value)}
              style={{ width: '100%', background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '8px', color: '#fff', padding: '0.65rem 0.8rem', fontSize: '0.9rem' }}
            >
              <option value="All">All Groups</option>
              {filters.acorn_groups.map(grp => <option key={grp} value={grp}>{grp}</option>)}
            </select>
          </div>
          <div>
            <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Tariff Class</label>
            <select 
              value={tariff}
              onChange={(e) => setTariff(e.target.value)}
              style={{ width: '100%', background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '8px', color: '#fff', padding: '0.65rem 0.8rem', fontSize: '0.9rem' }}
            >
              <option value="All">All Tariffs</option>
              {filters.tariff_types.map(t => <option key={t} value={t}>{t === 'Std' ? 'Standard (Std)' : 'Time-of-Use (ToU)'}</option>)}
            </select>
          </div>
        </div>

        {data && (
          <div style={{ marginTop: '1.25rem', background: 'rgba(0,212,170,0.04)', border: '1px dashed rgba(0,212,170,0.15)', borderRadius: '8px', padding: '0.75rem 1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              Query Scope: <strong style={{ color: 'var(--primary)' }}>{data.record_count.toLocaleString()}</strong> rows matching parameters
            </span>
            <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              Active Sample Size: <strong style={{ color: 'var(--primary)' }}>{data.household_count}</strong> households
            </span>
          </div>
        )}
      </div>

      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '4rem 0' }}>
          <Loader2 className="animate-spin" size={32} color="var(--primary)" />
          <span style={{ marginLeft: '0.75rem', color: 'var(--text-secondary)' }}>Re-indexing EDA query...</span>
        </div>
      )}

      {!loading && data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Main line chart */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart size={18} color="var(--primary)" /> Daily Consumption Trends
            </h4>
            <ChartRenderer chartData={data.charts.daily} height="400px" />
          </div>

          {/* Monthly trends & Seasonal distribution side-by-side */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem' }}>📊 Monthly Energy Totals</h4>
              <ChartRenderer chartData={data.charts.monthly} height="350px" />
            </div>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem' }}>🌦️ Seasonal Distribution Spread</h4>
              <ChartRenderer chartData={data.charts.seasonal} height="350px" />
            </div>
          </div>

          {/* Weather vs Energy widget */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sun size={18} color="var(--accent)" /> Weather Temperature Correlation
              </h4>
              <select
                value={weatherVar}
                onChange={(e) => setWeatherVar(e.target.value)}
                style={{ background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '6px', color: '#fff', padding: '0.4rem 0.75rem', fontSize: '0.85rem' }}
              >
                <option value="temperatureMin">Min Temperature (°C)</option>
                <option value="temperatureMax">Max Temperature (°C)</option>
                <option value="windSpeed">Wind Speed (mph)</option>
                <option value="humidity">Humidity Ratio</option>
                <option value="cloudCover">Cloud Cover %</option>
              </select>
            </div>
            <ChartRenderer chartData={data.charts.weather} height="380px" />
          </div>

          {/* ACORN vs Day Type comparison */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem' }}>🏘️ ACORN Socioeconomic Analysis</h4>
              <ChartRenderer chartData={data.charts.acorn} height="350px" />
            </div>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem' }}>📅 Weekend vs Weekday Consumption</h4>
              <ChartRenderer chartData={data.charts.day_type} height="350px" />
            </div>
          </div>

          {/* Correlation and Household Comparison */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem' }}>🔗 Feature Cross-Correlation Heatmap</h4>
              <ChartRenderer chartData={data.charts.correlation} height="400px" />
            </div>

            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Layers size={18} color="var(--primary)" /> Peer Household Comparison
              </h4>
              
              <form onSubmit={handleAddHousehold} style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                <select
                  value={householdInput}
                  onChange={(e) => setHouseholdInput(e.target.value)}
                  style={{ flex: 1, background: 'rgba(15,23,42,0.4)', border: '1px solid var(--card-border)', borderRadius: '6px', color: '#fff', padding: '0.5rem', fontSize: '0.85rem' }}
                >
                  <option value="">Select household (LCLid)...</option>
                  {filters.households.map(id => <option key={id} value={id}>{id}</option>)}
                </select>
                <button type="submit" className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>Compare</button>
              </form>

              {compareHouseholds.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.25rem' }}>
                  {compareHouseholds.map(id => (
                    <span key={id} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', background: 'var(--secondary-glow)', color: 'var(--secondary)', border: '1px solid rgba(124,58,237,0.2)', borderRadius: '8px', padding: '0.35rem 0.65rem', fontSize: '0.8rem', fontWeight: 600 }}>
                      {id}
                      <button type="button" onClick={() => handleRemoveHousehold(id)} style={{ border: 'none', background: 'none', cursor: 'pointer', color: 'var(--secondary)', fontSize: '0.9rem', display: 'flex', alignItems: 'center' }}>&times;</button>
                    </span>
                  ))}
                </div>
              )}

              {compareHouseholds.length > 0 ? (
                <ChartRenderer chartData={data.charts.comparison} height="300px" />
              ) : (
                <div style={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', background: 'rgba(255,255,255,0.01)', border: '1px dashed var(--card-border)', borderRadius: '12px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Choose households from the dropdown above to render peer analysis.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

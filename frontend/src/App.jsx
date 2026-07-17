import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import { 
  Home as HomeIcon, 
  Search, 
  TrendingUp, 
  Users, 
  AlertTriangle, 
  Lightbulb, 
  MessageSquare, 
  Zap, 
  Loader2 
} from 'lucide-react'

import Home from './pages/Home'
import Eda from './pages/Eda'
import Forecast from './pages/Forecast'
import Clusters from './pages/Clusters'
import Anomalies from './pages/Anomalies'
import Insights from './pages/Insights'
import Chat from './pages/Chat'

import { apiFetch } from './utils/api'

function Navigation() {
  const location = useLocation()
  const navItems = [
    { path: '/', label: 'Home', icon: HomeIcon },
    { path: '/eda', label: 'Exploratory Analysis', icon: Search },
    { path: '/forecast', label: 'Forecasting', icon: TrendingUp },
    { path: '/clusters', label: 'Pattern Analysis', icon: Users },
    { path: '/anomalies', label: 'Anomaly Detection', icon: AlertTriangle },
    { path: '/insights', label: 'AI Insights', icon: Lightbulb },
    { path: '/chat', label: 'AI Chat', icon: MessageSquare },
  ]

  return (
    <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '1rem 0.75rem', flex: 1 }}>
      {navItems.map((item) => {
        const Icon = item.icon
        const isActive = location.pathname === item.path
        return (
          <Link
            key={item.path}
            to={item.path}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.85rem',
              padding: '0.8rem 1rem',
              color: isActive ? 'var(--primary)' : 'var(--text-secondary)',
              textDecoration: 'none',
              borderRadius: '12px',
              fontWeight: 600,
              fontSize: '0.95rem',
              background: isActive ? 'var(--primary-glow)' : 'transparent',
              border: isActive ? '1px solid rgba(0, 212, 170, 0.15)' : '1px solid transparent',
              transition: 'all 0.2s ease',
            }}
            className="nav-link"
          >
            <Icon size={18} />
            {item.label}
          </Link>
        )
      })}
    </nav>
  )
}

function App() {
  const [filters, setFilters] = useState({ acorn_groups: [], tariff_types: [], households: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Fetch options for filters on startup
    apiFetch('/api/filters')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch options')
        return res.json()
      })
      .then((data) => {
        setFilters(data)
        setLoading(false)
      })
      .catch((err) => {
        console.error(err)
        setError('Connection to backend failed. Please ensure the FastAPI server is running.')
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', justifyContent: 'center', alignItems: 'center', gap: '1rem', background: 'var(--bg-dark)' }}>
        <Loader2 className="animate-spin" size={48} color="var(--primary)" />
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', fontWeight: 500 }}>Initializing Smart Analytics...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', justifyContent: 'center', alignItems: 'center', gap: '1.5rem', background: 'var(--bg-dark)', padding: '2rem', textAlign: 'center' }}>
        <AlertTriangle size={64} color="var(--danger)" />
        <h2 style={{ fontSize: '1.8rem', fontWeight: 700 }}>Connection Error</h2>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '500px', lineHeight: 1.6 }}>{error}</p>
        <button className="btn-primary" onClick={() => window.location.reload()}>Retry Connection</button>
      </div>
    )
  }

  return (
    <BrowserRouter>
      <div className="dashboard-layout">
        <aside className="sidebar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1.75rem 1.25rem 1rem 1.25rem' }}>
            <div style={{ background: 'var(--primary-glow)', padding: '0.5rem', borderRadius: '10px', border: '1px solid rgba(0, 212, 170, 0.2)' }}>
              <Zap size={22} color="var(--primary)" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.25rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                Energy <span style={{ color: 'var(--primary)' }}>AI</span>
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 500 }}>Smart Meter Analytics</p>
            </div>
          </div>
          <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)', margin: '0.5rem 1.25rem' }}></div>
          
          <Navigation />

          <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)', margin: '0.5rem 1.25rem' }}></div>
          <div style={{ padding: '1.25rem', textAlign: 'center' }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', lineHeight: 1.4 }}>
              London Smart Meter Dataset<br />
              500 Sampled Households
            </p>
          </div>
        </aside>

        <main className="main-content animate-fade-in">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/eda" element={<Eda filters={filters} />} />
            <Route path="/forecast" element={<Forecast filters={filters} />} />
            <Route path="/clusters" element={<Clusters filters={filters} />} />
            <Route path="/anomalies" element={<Anomalies filters={filters} />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/chat" element={<Chat />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App

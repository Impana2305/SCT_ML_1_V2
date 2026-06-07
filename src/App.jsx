import React, { useState, useMemo } from 'react';
import modelParams from './model_params.json';
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  BarChart, Bar, ReferenceLine, Cell
} from 'recharts';
import { Home, BarChart2, Shield, Sparkles, Cpu, Layers } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('estimator');
  const [area, setArea] = useState(2200);
  const [bedrooms, setBedrooms] = useState(3);
  const [bathrooms, setBathrooms] = useState(2);

  // --- 1. Client-Side Machine Learning Valuation ---
  const valuation = useMemo(() => {
    const areaPerBedroom = area / (bedrooms + 0.1);
    const inputVec = [area, bedrooms, bathrooms, areaPerBedroom];
    
    // Scale inputs: (X - mean) / scale
    const scaledVec = inputVec.map((val, idx) => {
      return (val - modelParams.mean[idx]) / modelParams.scale[idx];
    });
    
    // Regression inference: Intercept + Sum(Coef * ScaledVar)
    let pred = modelParams.intercept;
    for (let i = 0; i < scaledVec.length; i++) {
      pred += scaledVec[i] * modelParams.coef[i];
    }
    
    return Math.max(0, pred);
  }, [area, bedrooms, bathrooms]);

  const areaPerBedroom = area / (bedrooms + 0.1);

  // --- 2. Calculate Dataset Summary Metrics ---
  const stats = useMemo(() => {
    const dataset = modelParams.dataset;
    const count = dataset.length;
    const prices = dataset.map(d => d.price);
    const areas = dataset.map(d => d.area);
    
    const avgPrice = prices.reduce((sum, val) => sum + val, 0) / count;
    const avgArea = areas.reduce((sum, val) => sum + val, 0) / count;
    
    return {
      count,
      avgPrice,
      avgArea,
      maxPrice: Math.max(...prices),
      minPrice: Math.min(...prices),
      minArea: Math.min(...areas),
      maxArea: Math.max(...areas)
    };
  }, []);

  // --- 3. Compute Price Histogram Bins dynamically ---
  const priceHistogramData = useMemo(() => {
    const prices = modelParams.dataset.map(d => d.price);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const binCount = 18;
    const binWidth = (max - min) / binCount;
    
    const bins = Array.from({ length: binCount }, (_, i) => {
      const start = min + i * binWidth;
      const binLabel = start >= 10000000 
        ? `₹${(start / 10000000).toFixed(1)}Cr` 
        : `₹${(start / 100000).toFixed(0)}L`;
      return {
        binStart: start,
        binEnd: start + binWidth,
        name: binLabel,
        count: 0
      };
    });
    
    prices.forEach(p => {
      const index = Math.min(binCount - 1, Math.floor((p - min) / binWidth));
      bins[index].count++;
    });
    
    return bins;
  }, []);

  // --- 4. Format Feature Importance / Coefficient Weights ---
  const weightsData = useMemo(() => {
    const featureLabels = ['Area', 'Bedrooms', 'Bathrooms', 'Area per Bedroom'];
    return modelParams.coef.map((val, idx) => ({
      name: featureLabels[idx],
      weight: val
    })).sort((a, b) => a.weight - b.weight);
  }, []);

  // --- Custom Tooltip formatter for Price ---
  const formatCurrency = (val) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(val);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="app-title">🏠 House Price Prediction</h1>
        <p className="app-subtitle">A high-performance application calculating house price valuations and analyzing local dataset statistics.</p>
      </header>

      {/* Main Dashboard Layout */}
      <div className="dashboard-grid">
        {/* Sidebar Inputs */}
        <aside className="sidebar-panel">
          <h3 className="sidebar-title">
            <Layers size={20} />
            Property Specs
          </h3>
          
          <div className="input-group">
            <label className="input-label">
              Living Area (sq. ft.)
              <span className="value-badge">{area.toLocaleString()} sq ft</span>
            </label>
            <input 
              type="range" 
              min={stats.minArea} 
              max={stats.maxArea} 
              step={50}
              value={area}
              onChange={(e) => setArea(Number(e.target.value))} 
            />
            <input 
              type="number"
              min={stats.minArea}
              max={stats.maxArea}
              value={area}
              onChange={(e) => setArea(Number(e.target.value))}
              style={{ marginTop: '8px' }}
            />
          </div>

          <div className="input-group">
            <label className="input-label">
              Bedrooms
              <span className="value-badge">{bedrooms} Rooms</span>
            </label>
            <input 
              type="range" 
              min={1} 
              max={5} 
              step={1}
              value={bedrooms}
              onChange={(e) => setBedrooms(Number(e.target.value))} 
            />
          </div>

          <div className="input-group">
            <label className="input-label">
              Bathrooms
              <span className="value-badge">{bathrooms} Baths</span>
            </label>
            <input 
              type="range" 
              min={1} 
              max={4} 
              step={1}
              value={bathrooms}
              onChange={(e) => setBathrooms(Number(e.target.value))} 
            />
          </div>

          <div style={{ marginTop: '20px', padding: '16px', background: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0', fontSize: '0.85rem' }}>
            <strong>Spatial Ratio:</strong> {areaPerBedroom.toFixed(1)} sq. ft. per bedroom
          </div>
        </aside>

        {/* Dashboard Panels */}
        <main className="main-content">
          {/* Navigation Tabs */}
          <div className="tabs-header">
            <button 
              className={`tab-button ${activeTab === 'estimator' ? 'active' : ''}`}
              onClick={() => setActiveTab('estimator')}
            >
              <Home size={18} />
              Valuation Estimator
            </button>
            <button 
              className={`tab-button ${activeTab === 'explorer' ? 'active' : ''}`}
              onClick={() => setActiveTab('explorer')}
            >
              <BarChart2 size={18} />
              Market Explorer
            </button>
            <button 
              className={`tab-button ${activeTab === 'diagnostics' ? 'active' : ''}`}
              onClick={() => setActiveTab('diagnostics')}
            >
              <Cpu size={18} />
              Diagnostics & Analytics
            </button>
          </div>

          {/* Main Visual Display Area */}
          <div className="main-display">
            
            {/* TAB 1: Valuation Estimator */}
            {activeTab === 'estimator' && (
              <div className="grid-2col">
                <div className="col-val">
                  <h3 className="section-header">Estimated Value</h3>
                  
                  {/* Valuation Card */}
                  <div className="glass-card">
                    <div className="valuation-label">Estimated Valuation</div>
                    <div className="valuation-price">{formatCurrency(valuation)}</div>
                    <p className="valuation-desc">
                      Valuation computed dynamically using multivariate linear weights. Live calculation removes python server lag entirely.
                    </p>
                  </div>

                  <h3 className="section-header">Architectural Insights</h3>
                  <div style={{ padding: '16px', borderRadius: '12px', background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                    {areaPerBedroom < 400 ? (
                      <div>
                        <span className="badge badge-warning">Compact Layout</span>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                          ⚠️ The spatial distribution is relatively compact ({areaPerBedroom.toFixed(1)} sq ft per room). Individual room sizes might feel restricted.
                        </p>
                      </div>
                    ) : areaPerBedroom > 900 ? (
                      <div>
                        <span className="badge badge-success">Premium Spacious</span>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                          ✨ Features a highly premium room distribution ({areaPerBedroom.toFixed(1)} sq ft per room). High luxury value appeal.
                        </p>
                      </div>
                    ) : (
                      <div>
                        <span className="badge badge-info">Balanced Design</span>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                          ✅ Balanced spatial distribution ({areaPerBedroom.toFixed(1)} sq ft per room). Optimal layout comfort and market standard.
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="col-val">
                  <h3 className="section-header">Market Placement</h3>
                  <div className="chart-container">
                    <div className="chart-title">Sales Price vs Area (Sq Ft)</div>
                    <div style={{ width: '100%', height: 350 }}>
                      <ResponsiveContainer>
                        <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis 
                            type="number" 
                            dataKey="area" 
                            name="Area" 
                            unit=" sq.ft" 
                            domain={[stats.minArea - 500, stats.maxArea + 500]}
                            tick={{ fontSize: 11 }}
                          />
                          <YAxis 
                            type="number" 
                            dataKey="price" 
                            name="Price" 
                            domain={[0, stats.maxPrice + 1000000]}
                            tickFormatter={(v) => {
                              if (v >= 10000000) return `₹${(v / 10000000).toFixed(1)}Cr`;
                              if (v >= 100000) return `₹${(v / 100000).toFixed(0)}L`;
                              return `₹${v}`;
                            }}
                            tick={{ fontSize: 11 }}
                          />
                          <Tooltip 
                            cursor={{ strokeDasharray: '3 3' }}
                            formatter={(value, name) => [name === 'Price' ? formatCurrency(value) : value, name]}
                          />
                          <Legend wrapperStyle={{ fontSize: 12 }} />
                          <Scatter name="Market Records" data={modelParams.dataset} fill="#94a3b8" opacity={0.65} />
                          <Scatter 
                            name="Your Valuation" 
                            data={[{ area, price: valuation }]} 
                            fill="#f59e0b" 
                            shape="diamond"
                            style={{ filter: 'drop-shadow(0 4px 6px rgba(0,0,0,0.15))' }}
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="chart-caption">
                      The yellow diamond represents where your currently configured property sits relative to historic data.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: Market Explorer */}
            {activeTab === 'explorer' && (
              <div>
                <h3 className="section-header">Market Trends & Statistics</h3>
                <div className="stats-row">
                  <div className="stat-widget">
                    <div className="stat-label">Properties Analyzed</div>
                    <div className="stat-value">{stats.count}</div>
                  </div>
                  <div className="stat-widget">
                    <div className="stat-label">Avg. Home Price</div>
                    <div className="stat-value">{formatCurrency(stats.avgPrice)}</div>
                  </div>
                  <div className="stat-widget">
                    <div className="stat-label">Avg. Living Space</div>
                    <div className="stat-value">{stats.avgArea.toFixed(0)} sq ft</div>
                  </div>
                  <div className="stat-widget">
                    <div className="stat-label">Price Ceiling</div>
                    <div className="stat-value">{formatCurrency(stats.maxPrice)}</div>
                  </div>
                </div>

                <div className="chart-container">
                  <div className="chart-title">House Price Market Valuation Distribution</div>
                  <div style={{ width: '100%', height: 350 }}>
                    <ResponsiveContainer>
                      <BarChart data={priceHistogramData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                        <YAxis tick={{ fontSize: 11 }} label={{ value: 'Number of Homes', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: 12, fill: 'var(--text-muted)' } }} />
                        <Tooltip formatter={(value) => [`${value} Properties`, 'Count']} />
                        <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                          {priceHistogramData.map((entry, index) => {
                            const isUserBin = valuation >= entry.binStart && valuation < entry.binEnd;
                            return <Cell key={`cell-${index}`} fill={isUserBin ? '#10b981' : '#3b82f6'} />;
                          })}
                        </Bar>
                        {/* Highlight the user's valuation position */}
                        <ReferenceLine 
                          x={priceHistogramData.find(b => valuation >= b.binStart && valuation < b.binEnd)?.name}
                          stroke="#10b981" 
                          strokeWidth={2.5}
                          strokeDasharray="4 4"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-caption">
                    The distribution of historical properties. The green bar and line indicate which price bracket your estimated property falls into.
                  </div>
                </div>
              </div>
            )}

            {/* TAB 3: Model Diagnostics */}
            {activeTab === 'diagnostics' && (
              <div className="grid-2col">
                <div className="col-val">
                  <h3 className="section-header">Accuracy Diagnostics</h3>
                  <div className="stats-row" style={{ gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
                    <div className="stat-widget" style={{ textAlign: 'left', padding: '16px' }}>
                      <div className="stat-label" style={{ fontSize: '0.75rem' }}>R-squared Score (R²)</div>
                      <div className="stat-value" style={{ fontSize: '1.6rem', color: 'var(--color-primary)' }}>{modelParams.r2.toFixed(4)}</div>
                      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                        Explains {(modelParams.r2 * 100).toFixed(1)}% of variance in market prices.
                      </p>
                    </div>
                    <div className="stat-widget" style={{ textAlign: 'left', padding: '16px' }}>
                      <div className="stat-label" style={{ fontSize: '0.75rem' }}>Mean Absolute Error (MAE)</div>
                      <div className="stat-value" style={{ fontSize: '1.6rem', color: 'var(--color-primary)' }}>{formatCurrency(modelParams.mae)}</div>
                      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                        Avg dollar price deviation in predictions.
                      </p>
                    </div>
                  </div>

                  <div className="chart-container">
                    <div className="chart-title">Model Coefficients (Feature Weights)</div>
                    <div style={{ width: '100%', height: 280 }}>
                      <ResponsiveContainer>
                        <BarChart data={weightsData} layout="vertical" margin={{ top: 10, right: 20, left: 20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis 
                            type="number" 
                            tickFormatter={(v) => {
                              const sign = v < 0 ? '-' : '';
                              const absV = Math.abs(v);
                              if (absV >= 100000) return `${sign}₹${(absV / 100000).toFixed(1)}L`;
                              return `${sign}₹${(absV / 1000).toFixed(0)}k`;
                            }} 
                            tick={{ fontSize: 10 }} 
                          />
                          <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} />
                          <Tooltip formatter={(value) => [formatCurrency(value), 'Coeff Weight']} />
                          <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
                            {weightsData.map((entry, index) => {
                              const isPositive = entry.weight > 0;
                              return <Cell key={`cell-${index}`} fill={isPositive ? '#10b981' : '#ef4444'} />;
                            })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="chart-caption">
                      Standardized linear coefficients. Positive values (green) drive prices up; negative values (red) pull prices down.
                    </div>
                  </div>
                </div>

                <div className="col-val">
                  <h3 className="section-header">Prediction Fit Quality</h3>
                  <div className="chart-container">
                    <div className="chart-title">Actual vs. Predicted Price Fit</div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '12px', textAlign: 'center' }}>
                      Validation of predicted values compared against real test values.
                    </p>
                    <div style={{ width: '100%', height: 350 }}>
                      <ResponsiveContainer>
                        <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis 
                            type="number" 
                            dataKey="actual" 
                            name="Actual Price" 
                            domain={[0, 14000000]}
                            tickFormatter={(v) => {
                              if (v >= 10000000) return `₹${(v / 10000000).toFixed(1)}Cr`;
                              if (v >= 100000) return `₹${(v / 100000).toFixed(0)}L`;
                              return `₹${v}`;
                            }}
                            tick={{ fontSize: 11 }}
                          />
                          <YAxis 
                            type="number" 
                            dataKey="predicted" 
                            name="Predicted Price" 
                            domain={[0, 14000000]}
                            tickFormatter={(v) => {
                              if (v >= 10000000) return `₹${(v / 10000000).toFixed(1)}Cr`;
                              if (v >= 100000) return `₹${(v / 100000).toFixed(0)}L`;
                              return `₹${v}`;
                            }}
                            tick={{ fontSize: 11 }}
                          />
                          <Tooltip formatter={(value) => [formatCurrency(value), 'Price']} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          {/* We display a sample of actual vs predicted pairs for diagnostics */}
                          <Scatter name="Validation Fits" data={
                            modelParams.dataset.slice(0, 100).map((d) => {
                              // Standardize
                              const areaPerBedroom = d.area / (d.bedrooms + 0.1);
                              const xVec = [d.area, d.bedrooms, d.bathrooms, areaPerBedroom];
                              const scaledVec = xVec.map((val, idx) => (val - modelParams.mean[idx]) / modelParams.scale[idx]);
                              let pred = modelParams.intercept;
                              for (let i = 0; i < scaledVec.length; i++) {
                                pred += scaledVec[i] * modelParams.coef[i];
                              }
                              return {
                                actual: d.price,
                                predicted: Math.max(0, pred)
                              };
                            })
                          } fill="#3b82f6" opacity={0.6} />
                          {/* Ideal Line y=x */}
                          <ReferenceLine segment={[{ x: 1000000, y: 1000000 }, { x: 13000000, y: 13000000 }]} stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" label={{ value: "Perfect Prediction (y=x)", position: "top", fill: "#ef4444", fontSize: 11 }} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="chart-caption">
                      Points clustered tightly along the red dashed line indicate highly accurate baseline estimations.
                    </div>
                  </div>
                </div>
              </div>
            )}

          </div>

        </main>
      </div>
    </div>
  );
}

export default App;

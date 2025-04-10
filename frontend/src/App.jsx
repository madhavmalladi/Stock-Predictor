import { useState } from 'react'
import './App.css'

function App() {
  const [ticker, setTicker] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const popularStocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA']

  const predictStock = async (stockTicker) => {
    try {
      setLoading(true)
      setError('')
      setPrediction(null)
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker: stockTicker }),
      })
      
      // Check if response is ok
      if (!response.ok) {
        // Handle HTTP errors
        const errorText = await response.text()
        console.error(`HTTP Error: ${response.status} - ${errorText}`)
        throw new Error(`Server returned ${response.status}: ${errorText || response.statusText}`)
      }

      const text = await response.text()
      if (!text) {
        throw new Error('Empty response from server')
      }

      let data
      try {
        data = JSON.parse(text)
      } catch (e) {
        console.error('Failed to parse JSON:', text)
        throw new Error(`Invalid JSON response: ${text.substring(0, 100)}...`)
      }
      
      if (data.status === 'success') {
        setPrediction(data)
      } else {
        setError(data.error || 'Failed to predict stock price')
      }
    } catch (err) {
      setError(`Error: ${err.message || 'Could not connect to server'}`)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (ticker.trim()) {
      predictStock(ticker.toUpperCase())
    }
  }

  const handleQuickSelect = (stock) => {
    setTicker(stock)
    predictStock(stock)
  }

  const getChangeClass = () => {
    if (!prediction) return ''
    return prediction.change_percentage > 0 ? 'positive' : 'negative'
  }

  return (
    <div className="app-container">
      <header>
        <h1>Stock Price Predictor</h1>
        <p>Predict tomorrow's stock prices with machine learning</p>
      </header>

      <div className="search-section">
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="Enter stock symbol (e.g., AAPL)"
              required
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </div>
        </form>

        <div className="quick-stocks">
          <p>Popular stocks:</p>
          <div className="stock-buttons">
            {popularStocks.map((stock) => (
              <button
                key={stock}
                onClick={() => handleQuickSelect(stock)}
                disabled={loading}
              >
                {stock}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      {prediction && (
        <div className="prediction-result">
          <h2>{prediction.ticker}</h2>
          <div className="price-container">
            <div className="price-box">
              <span className="price-label">Current Price</span>
              <span className="price-value">${prediction.current_price.toFixed(2)}</span>
            </div>
            <div className="arrow">→</div>
            <div className="price-box">
              <span className="price-label">Predicted Price</span>
              <span className="price-value">${prediction.predicted_price.toFixed(2)}</span>
            </div>
          </div>
          <div className={`change-percentage ${getChangeClass()}`}>
            {prediction.change_percentage > 0 ? '↑' : '↓'} {Math.abs(prediction.change_percentage).toFixed(2)}%
          </div>
          <div className="prediction-note">
            <p>Based on historical patterns and machine learning</p>
          </div>
        </div>
      )}

      <footer>
        <p>Used the YFinance API</p>
      </footer>
    </div>
  )
}

export default App

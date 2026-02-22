import json
import time
import sys
import yfinance as yf
from groq import Groq
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- CONFIGURATION ---
# ⚠️ REPLACE THESE WITH YOUR NEW KEYS
GROQ_API_KEY = "gsk_pJpraQSYBs9kV2thMqu8WGdyb3FYT0Rnp6EAp0luWDOYM1Z85RmUNO"
ALPACA_KEY = "PKTK6ERS6FHXZ2LLKHDLWG3HXK"
ALPACA_SECRET = "6BXQHikrH7T2LF9qTT9pspLHoeL8xCQ2H5iDkV6GusCC"

# --- 🛑 HUMAN SAFETY SWITCH 🛑 ---
REQUIRE_HUMAN_CONFIRMATION = True 

# List of stocks to analyze
WATCHLIST = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "COIN", "PLTR"]

# Initialize Clients
try:
    client = Groq(api_key=GROQ_API_KEY)
    trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
except Exception as e:
    print(f"❌ CRITICAL SETUP ERROR: {e}")
    sys.exit()

def get_market_data_reliable(tickers):
    """
    Reliable Method: Fetches stocks one by one.
    Includes 'News-Proof' logic to prevent crashing on missing titles.
    """
    print(f"⚡ Fetching live data for {len(tickers)} stocks (Sequential Mode)...")
    
    summary_list = []
    
    for symbol in tickers:
        try:
            print(f"   -> Scanning {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # 1. Get Price History
            hist = ticker.history(period="5d")
            
            if len(hist) < 2:
                print(f"      ⚠️ No price data for {symbol}, skipping.")
                continue
            
            # Extract Data
            price = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = ((price - prev) / prev) * 100
            
            # 2. Get News (Safe Mode)
            try:
                news = ticker.news
                if news and len(news) > 0:
                    # Try to find 'title', but fallback if Yahoo changes keys
                    news_text = news[0].get('title', news[0].get('headline', 'News link available'))
                else:
                    news_text = "No recent news"
            except Exception:
                news_text = "News Unavailable (API Error)"

            # Format for the Brain
            summary_list.append(
                f"STOCK:{symbol} | PRICE:${price:.2f} | CHANGE:{change:.2f}% | NEWS:{news_text}"
            )
            
            # Sleep briefly to be nice to Yahoo
            time.sleep(0.5)
            
        except Exception as e:
            print(f"      ❌ Critical Error on {symbol}: {e}")
            continue
            
    return "\n".join(summary_list)

def ask_brain(market_text):
    print("\n🧠 Analyzing Market Data with Groq...")
    
    prompt = f"""
    ROLE: Elite Hedge Fund Algorithm.
    TASK: Analyze the market data below. Pick the SINGLE best stock to BUY based on positive news and momentum.
    
    MARKET DATA:
    {market_text}
    
    CONSTRAINTS:
    1. If no stock looks good, select "HOLD".
    2. Output MUST be valid JSON.
    
    JSON RESPONSE FORMAT:
    {{
      "action": "BUY" or "HOLD",
      "ticker": "SYMBOL",
      "confidence": 0-100,
      "reason": "Short reason (max 10 words)"
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a JSON-only trading bot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ Brain Error: {e}")
        return {"action": "HOLD", "reason": "Brain Malfunction"}

def execute_trade(signal):
    action = signal.get("action", "HOLD")
    ticker = signal.get("ticker")
    reason = signal.get("reason")
    confidence = signal.get("confidence", 0)
    
    print("-" * 40)
    print(f"🤖 BOT SIGNAL: {action} {ticker}")
    print(f"📝 REASON:     {reason}")
    print(f"📊 CONFIDENCE: {confidence}%")
    print("-" * 40)
    
    if action == "BUY" and ticker:
        # --- HUMAN CONFIRMATION CHECK ---
        if REQUIRE_HUMAN_CONFIRMATION:
            print(f"\n✋ WAIT! Human confirmation required to BUY {ticker}.")
            user_input = input(f"   Type 'YES' to buy {ticker} on Paper Account: ").strip().upper()
            
            if user_input != "YES":
                print("❌ Trade Cancelled by User.")
                return

        # Execute Order
        try:
            print(f"🚀 Sending Order to Alpaca...")
            order = MarketOrderRequest(
                symbol=ticker,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order_data=order)
            print(f"✅ ORDER FILLED: Bought 1 share of {ticker}!")
        except Exception as e:
            print(f"❌ Execution Failed: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 🚀 STARTING AGENT (SAFE MODE) ---")
    
    # 1. Fetch (Sequential & Reliable)
    data_text = get_market_data_reliable(WATCHLIST)
    
    if data_text:
        # 2. Think
        decision = ask_brain(data_text)
        
        # 3. Act
        execute_trade(decision)
    else:
        print("❌ Data fetch failed (Check internet connection).")
    
    print("\n--- 🏁 CYCLE COMPLETE ---")
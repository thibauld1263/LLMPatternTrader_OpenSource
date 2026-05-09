import logging
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class MT5OrderExecutor:
    """
    Handles executing trades directly on the verified MT5 Connection.
    Accepts standardized Risk Manager (LLM) parameters.
    """
    
    def __init__(self):
        # We assume mt5.initialize() has already been called safely by MT5DataFetcher
        pass

    def execute_market_order(self, symbol: str, is_long: bool, volume: float, stop_loss: float, take_profit: float, magic_number: int = 123456, comment: str = "LLM_Pattern_Trader") -> dict:
        """
        Sends a live Market Order based on LLM approval.
        Volume is precisely pre-calculated before the LLM step.
        """
        if not mt5.symbol_select(symbol, True):
            return {"status": "error", "message": f"Failed to select symbol {symbol}."}

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"status": "error", "message": f"Symbol {symbol} not found."}

        account = mt5.account_info()
        balance = account.balance if account else 0.0

        point = mt5.symbol_info(symbol).point
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        
        price = mt5.symbol_info_tick(symbol).ask if is_long else mt5.symbol_info_tick(symbol).bid
        
        # Guard against zero or inverted stop loss hallucination
        sl_distance = abs(price - float(stop_loss))
        if sl_distance <= 0 or (is_long and stop_loss >= price) or (not is_long and stop_loss <= price):
            return {"status": "error", "message": f"LLM Hallucinated invalid SL distance: {stop_loss} vs Entry {price}"}
            
        # Ensure LLM-provided volume respects broker limits
        min_lot = symbol_info.volume_min
        lot_step = symbol_info.volume_step
        final_volume = round(float(volume) / lot_step) * lot_step
        final_volume = max(min_lot, min(final_volume, symbol_info.volume_max))

        order_type = mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(final_volume),
            "type": order_type,
            "price": price,
            "sl": float(stop_loss),
            "tp": float(take_profit),
            "deviation": 20, # points
            "magic": magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, # Commonly accepted on most brokers
        }

        # Send to Terminal
        result = mt5.order_send(request)
        
        if result is None:
            err = mt5.last_error()
            logger.error(f"MT5 OrderSend failed: {err}")
            return {"status": "error", "message": f"Internal MT5 Error: {err}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"MT5 Order Rejected (RetCode {result.retcode}): {result.comment}")
            return {"status": "error", "message": f"Broker Rejected: {result.comment}"}

        logger.info(f"MT5 Order Executed Successfully. Ticket #{result.order}")
        return {
            "status": "success",
            "ticket": result.order,
            "volume": result.volume,
            "price": result.price
        }

    def close_all_positions(self, symbol: str) -> int:
        """Emergency close for a specific symbol."""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return 0
            
        closed = 0
        for pos in positions:
            tick = mt5.symbol_info_tick(symbol)
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            price = tick.bid if is_buy else tick.ask
            type_close = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
            
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": type_close,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "Force Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
                
        return closed

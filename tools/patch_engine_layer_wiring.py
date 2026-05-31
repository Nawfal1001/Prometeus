from pathlib import Path

path = Path('core/engine.py')
text = path.read_text()
old = '''        whale_result = {"layer_score": self.whale.get_layer_score()}
        sent_result = {"layer_score": self.sentiment.get_layer_score()}
        liq_result = self.liquidation.update(current_price, symbol)
'''
new = '''        whale_result = self.whale.update(df=df, symbol=symbol)
        sent_result = {"layer_score": self.sentiment.get_layer_score()}
        liq_result = self.liquidation.update(current_price, symbol, df=df)
'''
if old not in text:
    raise SystemExit('target block not found; engine.py may already be patched or changed')
path.write_text(text.replace(old, new, 1))
print('patched engine layer wiring')

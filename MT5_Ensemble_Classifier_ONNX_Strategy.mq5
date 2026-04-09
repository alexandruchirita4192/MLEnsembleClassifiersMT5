#property strict
#property version   "1.10"
#property description "EA MT5: Weighted Ensemble ONNX (MLP + LightGBM + HGB)"
#property description "Weights can completely shift the signal to a single model"

#include <Trade/Trade.mqh>

#resource "mlp.onnx" as uchar ExtModelMLP[]
#resource "lightgbm.onnx" as uchar ExtModelLGBM[]
#resource "hgb.onnx" as uchar ExtModelHGB[]

input double InpLots                  = 0.10;
input double InpEntryProbThreshold    = 0.60;
input double InpMinProbGap            = 0.15;
input bool   InpUseAtrStops           = true;
input double InpStopAtrMultiple       = 1.50;
input double InpTakeAtrMultiple       = 2.00;
input int    InpMaxBarsInTrade        = 8;
input bool   InpCloseOnOppositeSignal = true;
input bool   InpAllowLong             = true;
input bool   InpAllowShort            = true;

input double InpMlpWeight             = 1.00;
input double InpLgbmWeight            = 0.00;
input double InpHgbWeight             = 0.00;

input bool   InpUseTrendFilter        = true;
input ENUM_TIMEFRAMES InpTrendTF      = PERIOD_H1;
input int    InpTrendMAPeriod         = 100;
input bool   InpTrendRequireSlope     = true;

input bool   InpUseTrendDistanceFilter = false;
input double InpTrendMinDistancePct    = 0.0010;

input bool   InpUseAtrVolFilter        = true;
input int    InpAtrVolLookback         = 50;
input double InpAtrMinPercentile       = 0.25;
input double InpAtrMaxPercentile       = 0.85;

input bool   InpUseKillSwitch                 = false;
input int    InpKillSwitchLookbackTrades      = 8;
input double InpKillSwitchMinWinRate          = 0.40;
input double InpKillSwitchMinProfitFactor     = 0.95;
input int    InpKillSwitchConsecutiveLosses   = 4;
input int    InpKillSwitchPauseBars           = 96;
input bool   InpKillSwitchFlatOnActivate      = true;

input long   InpMagic                 = 26042026;
input bool   InpLog                   = false;
input bool   InpDebugLog              = false;

const int FEATURE_COUNT = 10;
const int CLASS_COUNT   = 3;
const long EXT_INPUT_SHAPE[]  = {1, FEATURE_COUNT};
const long EXT_LABEL_SHAPE[]  = {1};
const long EXT_PROBA_SHAPE[]  = {1, CLASS_COUNT};

CTrade trade;
long g_mlp_handle  = INVALID_HANDLE;
long g_lgbm_handle = INVALID_HANDLE;
long g_hgb_handle  = INVALID_HANDLE;
int  g_trend_ma_handle = INVALID_HANDLE;

datetime g_last_bar_time = 0;
int g_bars_in_trade = 0;
bool g_kill_switch_active = false;
int  g_kill_switch_pause_remaining = 0;
int  g_consecutive_losses = 0;
double g_recent_closed_profits[];
int g_last_history_deals_total = 0;

double g_w_mlp = 0.0;
double g_w_lgbm = 0.0;
double g_w_hgb = 0.0;

enum SignalDirection { SIGNAL_SELL=-1, SIGNAL_FLAT=0, SIGNAL_BUY=1 };

bool NormalizeWeights()
{
   double a = MathMax(0.0, InpMlpWeight);
   double b = MathMax(0.0, InpLgbmWeight);
   double c = MathMax(0.0, InpHgbWeight);
   double s = a + b + c;
   if(s <= 0.0)
      return false;

   g_w_mlp = a / s;
   g_w_lgbm = b / s;
   g_w_hgb = c / s;
   return true;
}

bool IsNewBar()
{
   datetime t=iTime(_Symbol,_Period,0);
   if(t==0) return false;
   if(g_last_bar_time==0){ g_last_bar_time=t; return false; }
   if(t!=g_last_bar_time){ g_last_bar_time=t; return true; }
   return false;
}

double Mean(const double &arr[], int start_shift, int count)
{
   double sum=0.0;
   for(int i=start_shift;i<start_shift+count;i++) sum+=arr[i];
   return sum/count;
}

double StdDev(const double &arr[], int start_shift, int count)
{
   double m=Mean(arr,start_shift,count), s=0.0;
   for(int i=start_shift;i<start_shift+count;i++){ double d=arr[i]-m; s+=d*d; }
   return MathSqrt(s/MathMax(count-1,1));
}

double CalcATR(const MqlRates &rates[], int start_shift, int period)
{
   double sum_tr=0.0;
   for(int i=start_shift;i<start_shift+period;i++)
   {
      double high=rates[i].high, low=rates[i].low, prev_close=rates[i+1].close;
      double tr1=high-low, tr2=MathAbs(high-prev_close), tr3=MathAbs(low-prev_close);
      double tr=MathMax(tr1,MathMax(tr2,tr3));
      sum_tr+=tr;
   }
   return sum_tr/period;
}

double GetPercentileFromArray(const double &arr[], int count, double q)
{
   if(count<=0) return 0.0;
   if(q<=0.0) q=0.0;
   if(q>=1.0) q=1.0;

   double tmp[];
   ArrayResize(tmp,count);
   for(int i=0;i<count;i++) tmp[i]=arr[i];
   ArraySort(tmp);

   double pos=q*(count-1);
   int lo=(int)MathFloor(pos), hi=(int)MathCeil(pos);
   if(lo==hi) return tmp[lo];
   double w=pos-lo;
   return tmp[lo]*(1.0-w)+tmp[hi]*w;
}

bool BuildFeatureVector(matrixf &features, double &atr14)
{
   MqlRates rates[]; ArraySetAsSeries(rates,true);
   if(CopyRates(_Symbol,_Period,0,80,rates)<40) return false;

   double closes[];
   ArrayResize(closes,ArraySize(rates));
   ArraySetAsSeries(closes,true);
   for(int i=0;i<ArraySize(rates);i++) closes[i]=rates[i].close;

   int s=1;
   double ret_1=(closes[s]/closes[s+1])-1.0;
   double ret_3=(closes[s]/closes[s+3])-1.0;
   double ret_5=(closes[s]/closes[s+5])-1.0;
   double ret_10=(closes[s]/closes[s+10])-1.0;

   double one_bar_returns[];
   ArrayResize(one_bar_returns,30);
   for(int i=0;i<30;i++) one_bar_returns[i]=(closes[s+i]/closes[s+i+1])-1.0;

   double vol_10=StdDev(one_bar_returns,0,10);
   double vol_20=StdDev(one_bar_returns,0,20);

   double sma_10=Mean(closes,s,10), sma_20=Mean(closes,s,20);
   if(sma_10==0.0 || sma_20==0.0) return false;

   double dist_sma_10=(closes[s]/sma_10)-1.0;
   double dist_sma_20=(closes[s]/sma_20)-1.0;
   double mean_20=Mean(closes,s,20), std_20=StdDev(closes,s,20), zscore_20=0.0;
   if(std_20>0.0) zscore_20=(closes[s]-mean_20)/std_20;

   atr14=CalcATR(rates,s,14);

   features.Resize(1,FEATURE_COUNT);
   features[0][0]=(float)ret_1;
   features[0][1]=(float)ret_3;
   features[0][2]=(float)ret_5;
   features[0][3]=(float)ret_10;
   features[0][4]=(float)vol_10;
   features[0][5]=(float)vol_20;
   features[0][6]=(float)dist_sma_10;
   features[0][7]=(float)dist_sma_20;
   features[0][8]=(float)zscore_20;
   features[0][9]=(float)atr14;
   return true;
}

bool SetupHandle(long &handle, const uchar &buffer[])
{
   handle = OnnxCreateFromBuffer(buffer, ONNX_DEFAULT);
   if(handle == INVALID_HANDLE) return false;
   if(!OnnxSetInputShape(handle,0,EXT_INPUT_SHAPE)) return false;
   if(!OnnxSetOutputShape(handle,0,EXT_LABEL_SHAPE)) return false;
   if(!OnnxSetOutputShape(handle,1,EXT_PROBA_SHAPE)) return false;
   return true;
}

bool RunOneModel(long handle, const matrixf &x, double &pSell, double &pFlat, double &pBuy)
{
   long predicted_label[1];
   matrixf probs;
   probs.Resize(1,CLASS_COUNT);

   if(!OnnxRun(handle,0,x,predicted_label,probs)) return false;

   pSell=probs[0][0];
   pFlat=probs[0][1];
   pBuy=probs[0][2];
   return true;
}

bool PredictClassProbabilities(double &pSell, double &pFlat, double &pBuy, double &atr14)
{
   matrixf x;
   if(!BuildFeatureVector(x,atr14)) return false;

   double s1=0.0,f1=0.0,b1=0.0;
   double s2=0.0,f2=0.0,b2=0.0;
   double s3=0.0,f3=0.0,b3=0.0;

   if(g_w_mlp>0.0 && !RunOneModel(g_mlp_handle,x,s1,f1,b1)) return false;
   if(g_w_lgbm>0.0 && !RunOneModel(g_lgbm_handle,x,s2,f2,b2)) return false;
   if(g_w_hgb>0.0 && !RunOneModel(g_hgb_handle,x,s3,f3,b3)) return false;

   pSell = g_w_mlp*s1 + g_w_lgbm*s2 + g_w_hgb*s3;
   pFlat = g_w_mlp*f1 + g_w_lgbm*f2 + g_w_hgb*f3;
   pBuy  = g_w_mlp*b1 + g_w_lgbm*b2 + g_w_hgb*b3;
   return true;
}

SignalDirection SignalFromProbabilities(double pSell, double pFlat, double pBuy)
{
   double best=pFlat, second=-1.0;
   SignalDirection signal=SIGNAL_FLAT;

   if(pBuy>=pSell && pBuy>best)
   {
      second=MathMax(best,pSell);
      best=pBuy;
      signal=SIGNAL_BUY;
   }
   else if(pSell>pBuy && pSell>best)
   {
      second=MathMax(best,pBuy);
      best=pSell;
      signal=SIGNAL_SELL;
   }
   else
   {
      second=MathMax(pBuy,pSell);
      signal=SIGNAL_FLAT;
   }

   double gap=best-second;

   if(signal==SIGNAL_BUY)
   {
      if(!InpAllowLong) return SIGNAL_FLAT;
      if(pBuy<InpEntryProbThreshold || gap<InpMinProbGap) return SIGNAL_FLAT;
      return SIGNAL_BUY;
   }

   if(signal==SIGNAL_SELL)
   {
      if(!InpAllowShort) return SIGNAL_FLAT;
      if(pSell<InpEntryProbThreshold || gap<InpMinProbGap) return SIGNAL_FLAT;
      return SIGNAL_SELL;
   }

   return SIGNAL_FLAT;
}

bool GetTrendFilterValues(double &htf_close_1, double &ema_1, double &ema_2)
{
   if(!InpUseTrendFilter) return true;
   if(g_trend_ma_handle==INVALID_HANDLE) return false;

   htf_close_1=iClose(_Symbol,InpTrendTF,1);
   if(htf_close_1==0.0) return false;

   double ema_buf[];
   ArraySetAsSeries(ema_buf,true);
   if(CopyBuffer(g_trend_ma_handle,0,1,2,ema_buf)<2) return false;

   ema_1=ema_buf[0];
   ema_2=ema_buf[1];
   return true;
}

bool TrendAllows(SignalDirection signal)
{
   if(!InpUseTrendFilter || signal==SIGNAL_FLAT) return true;

   double htf_close_1=0.0, ema_1=0.0, ema_2=0.0;
   if(!GetTrendFilterValues(htf_close_1,ema_1,ema_2)) return false;

   bool slope_up=(ema_1>ema_2), slope_down=(ema_1<ema_2);
   double distance_pct=0.0;
   if(ema_1!=0.0) distance_pct=MathAbs(htf_close_1-ema_1)/ema_1;

   if(InpUseTrendDistanceFilter && distance_pct<InpTrendMinDistancePct) return false;

   if(signal==SIGNAL_BUY)
   {
      if(htf_close_1<=ema_1) return false;
      if(InpTrendRequireSlope && !slope_up) return false;
      return true;
   }

   if(signal==SIGNAL_SELL)
   {
      if(htf_close_1>=ema_1) return false;
      if(InpTrendRequireSlope && !slope_down) return false;
      return true;
   }

   return true;
}

bool AtrVolatilityAllows(double current_atr14)
{
   if(!InpUseAtrVolFilter) return true;

   MqlRates rates[];
   ArraySetAsSeries(rates,true);

   int need_bars=InpAtrVolLookback+20;
   if(CopyRates(_Symbol,_Period,0,need_bars,rates)<need_bars) return false;

   double atr_values[];
   ArrayResize(atr_values,InpAtrVolLookback);

   int s=1;
   for(int i=0;i<InpAtrVolLookback;i++) atr_values[i]=CalcATR(rates,s+i,14);

   double atr_min=GetPercentileFromArray(atr_values,InpAtrVolLookback,InpAtrMinPercentile);
   double atr_max=GetPercentileFromArray(atr_values,InpAtrVolLookback,InpAtrMaxPercentile);

   if(current_atr14<atr_min || current_atr14>atr_max) return false;
   return true;
}

bool HasOpenPosition(long &pos_type, double &pos_price)
{
   if(!PositionSelect(_Symbol)) return false;
   if((long)PositionGetInteger(POSITION_MAGIC)!=InpMagic) return false;

   pos_type=(long)PositionGetInteger(POSITION_TYPE);
   pos_price=PositionGetDouble(POSITION_PRICE_OPEN);
   return true;
}

void CloseOpenPosition()
{
   if(PositionSelect(_Symbol) && (long)PositionGetInteger(POSITION_MAGIC)==InpMagic)
      trade.PositionClose(_Symbol);
}

void PushClosedTradeProfit(double value)
{
   int size=ArraySize(g_recent_closed_profits);
   ArrayResize(g_recent_closed_profits,size+1);
   g_recent_closed_profits[size]=value;

   if(ArraySize(g_recent_closed_profits)>InpKillSwitchLookbackTrades)
   {
      for(int i=1;i<ArraySize(g_recent_closed_profits);i++)
         g_recent_closed_profits[i-1]=g_recent_closed_profits[i];
      ArrayResize(g_recent_closed_profits,InpKillSwitchLookbackTrades);
   }
}

void ActivateKillSwitch(string reason)
{
   if(!InpUseKillSwitch) return;
   g_kill_switch_active=true;
   g_kill_switch_pause_remaining=InpKillSwitchPauseBars;
   if(InpKillSwitchFlatOnActivate) CloseOpenPosition();
}

void DecrementKillSwitchPause()
{
   if(!g_kill_switch_active) return;
   if(g_kill_switch_pause_remaining>0) g_kill_switch_pause_remaining--;
   if(g_kill_switch_pause_remaining<=0)
   {
      g_kill_switch_active=false;
      g_consecutive_losses=0;
      ArrayResize(g_recent_closed_profits,0);
   }
}

void EvaluateKillSwitch()
{
   if(!InpUseKillSwitch || g_kill_switch_active) return;

   if(g_consecutive_losses>=InpKillSwitchConsecutiveLosses)
   {
      ActivateKillSwitch("");
      return;
   }

   int n=ArraySize(g_recent_closed_profits);
   if(n<InpKillSwitchLookbackTrades) return;

   int wins=0;
   double gross_profit=0.0, gross_loss_abs=0.0;

   for(int i=0;i<n;i++)
   {
      double p=g_recent_closed_profits[i];
      if(p>0.0){ wins++; gross_profit+=p; }
      else if(p<0.0){ gross_loss_abs+=MathAbs(p); }
   }

   double win_rate=(double)wins/(double)n;
   double profit_factor=(gross_loss_abs>0.0 ? gross_profit/gross_loss_abs : 999.0);

   if(win_rate<InpKillSwitchMinWinRate){ ActivateKillSwitch(""); return; }
   if(profit_factor<InpKillSwitchMinProfitFactor){ ActivateKillSwitch(""); return; }
}

void UpdateClosedTradeStats()
{
   if(!InpUseKillSwitch) return;
   if(!HistorySelect(0,TimeCurrent())) return;

   int total=HistoryDealsTotal();
   if(total<=g_last_history_deals_total) return;

   for(int i=g_last_history_deals_total;i<total;i++)
   {
      ulong deal_ticket=HistoryDealGetTicket(i);
      if(deal_ticket==0) continue;

      string symbol=HistoryDealGetString(deal_ticket,DEAL_SYMBOL);
      long magic=HistoryDealGetInteger(deal_ticket,DEAL_MAGIC);
      long entry=HistoryDealGetInteger(deal_ticket,DEAL_ENTRY);

      if(symbol!=_Symbol || magic!=InpMagic || entry!=DEAL_ENTRY_OUT) continue;

      double profit=HistoryDealGetDouble(deal_ticket,DEAL_PROFIT);
      double swap=HistoryDealGetDouble(deal_ticket,DEAL_SWAP);
      double commission=HistoryDealGetDouble(deal_ticket,DEAL_COMMISSION);
      double net=profit+swap+commission;

      PushClosedTradeProfit(net);

      if(net<0.0) g_consecutive_losses++;
      else if(net>0.0) g_consecutive_losses=0;
   }

   g_last_history_deals_total=total;
   EvaluateKillSwitch();
}

void OpenTrade(SignalDirection signal, double atr14)
{
   double ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);
   double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);

   double min_stop=(double)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*point;
   double sl_dist=MathMax(atr14*InpStopAtrMultiple,min_stop);
   double tp_dist=MathMax(atr14*InpTakeAtrMultiple,min_stop);

   double sl=0.0, tp=0.0;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);

   bool ok=false;
   if(signal==SIGNAL_BUY)
   {
      if(InpUseAtrStops){ sl=ask-sl_dist; tp=ask+tp_dist; }
      ok=trade.Buy(InpLots,_Symbol,ask,sl,tp,"Weighted Ensemble buy");
      if(ok) g_bars_in_trade=0;
   }
   else if(signal==SIGNAL_SELL)
   {
      if(InpUseAtrStops){ sl=bid+sl_dist; tp=bid-tp_dist; }
      ok=trade.Sell(InpLots,_Symbol,bid,sl,tp,"Weighted Ensemble sell");
      if(ok) g_bars_in_trade=0;
   }
}

void ManageExistingPosition(SignalDirection signal)
{
   long pos_type;
   double pos_price;
   if(!HasOpenPosition(pos_type,pos_price)) return;

   g_bars_in_trade++;
   bool should_close=false;

   if(InpCloseOnOppositeSignal)
   {
      if(pos_type==POSITION_TYPE_BUY && signal==SIGNAL_SELL) should_close=true;
      if(pos_type==POSITION_TYPE_SELL && signal==SIGNAL_BUY) should_close=true;
   }

   if(!should_close && g_bars_in_trade>=InpMaxBarsInTrade)
      should_close=true;

   if(should_close) CloseOpenPosition();
}

int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);

   if(!NormalizeWeights())
      return INIT_PARAMETERS_INCORRECT;

   if(g_w_mlp>0.0 && !SetupHandle(g_mlp_handle,ExtModelMLP)) return INIT_FAILED;
   if(g_w_lgbm>0.0 && !SetupHandle(g_lgbm_handle,ExtModelLGBM)) return INIT_FAILED;
   if(g_w_hgb>0.0 && !SetupHandle(g_hgb_handle,ExtModelHGB)) return INIT_FAILED;

   if(InpUseTrendFilter)
   {
      g_trend_ma_handle=iMA(_Symbol,InpTrendTF,InpTrendMAPeriod,0,MODE_EMA,PRICE_CLOSE);
      if(g_trend_ma_handle==INVALID_HANDLE) return INIT_FAILED;
   }

   if(HistorySelect(0,TimeCurrent()))
      g_last_history_deals_total=HistoryDealsTotal();
   else
      g_last_history_deals_total=0;

   ArrayResize(g_recent_closed_profits,0);
   g_consecutive_losses=0;
   g_kill_switch_active=false;
   g_kill_switch_pause_remaining=0;

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_mlp_handle!=INVALID_HANDLE) OnnxRelease(g_mlp_handle);
   if(g_lgbm_handle!=INVALID_HANDLE) OnnxRelease(g_lgbm_handle);
   if(g_hgb_handle!=INVALID_HANDLE) OnnxRelease(g_hgb_handle);
   if(g_trend_ma_handle!=INVALID_HANDLE) IndicatorRelease(g_trend_ma_handle);
}

void OnTick()
{
   if(!IsNewBar()) return;

   UpdateClosedTradeStats();
   DecrementKillSwitchPause();
   if(g_kill_switch_active) return;

   double pSell=0.0, pFlat=0.0, pBuy=0.0, atr14=0.0;
   if(!PredictClassProbabilities(pSell,pFlat,pBuy,atr14)) return;

   if(!AtrVolatilityAllows(atr14))
   {
      ManageExistingPosition(SIGNAL_FLAT);
      return;
   }

   SignalDirection raw_signal=SignalFromProbabilities(pSell,pFlat,pBuy);
   SignalDirection filtered_signal=raw_signal;

   if(!TrendAllows(raw_signal))
      filtered_signal=SIGNAL_FLAT;

   ManageExistingPosition(filtered_signal);

   long pos_type;
   double pos_price;
   if(HasOpenPosition(pos_type,pos_price)) return;

   if(filtered_signal==SIGNAL_BUY || filtered_signal==SIGNAL_SELL)
      OpenTrade(filtered_signal,atr14);
}
